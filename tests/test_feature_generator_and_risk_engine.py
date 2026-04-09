from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from backtester.feature_generator import FeatureGenerator, FeatureGeneratorConfig
from backtester.loaders.market_events import (
    MarketEventsLoadDeps,
    MarketEventsLoadRequest,
    PyArrowModules,
    load_market_events,
)
from backtester.simulation.risk_engine import RiskEvaluator, RiskLimits, RiskState
from utils.dataframes import filter_by_time_range, prepare_timestamp_index


def _book_events(base_time: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": "m1",
                "token_id": "unknown",
                "data": {
                    "asset_id": "yes",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": base_time + timedelta(seconds=1),
                "event_type": "book",
                "market_id": "m1",
                "token_id": "no",
                "data": {
                    "asset_id": "no",
                    "bids": [{"price": "0.48", "size": "80"}],
                    "asks": [{"price": "0.52", "size": "70"}],
                },
            },
        ]
    ).set_index("ts_event")


def test_feature_generator_cache_signature_and_invalidate(tmp_path) -> None:
    cache_dir = tmp_path / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    generator_a = FeatureGenerator(
        FeatureGeneratorConfig(cache_dir=cache_dir, cache_computation_signature="sig_a")
    )
    generator_b = FeatureGenerator(
        FeatureGeneratorConfig(cache_dir=cache_dir, cache_computation_signature="sig_b")
    )

    assert generator_a.cache_signature() != generator_b.cache_signature()

    (cache_dir / "features_x_key_a.parquet").write_text("x", encoding="utf-8")
    (cache_dir / "features_x_key_b.parquet").write_text("x", encoding="utf-8")

    removed = generator_a.invalidate_cache(cache_key="key_a")
    assert removed == 1
    assert (cache_dir / "features_x_key_b.parquet").exists()


def test_feature_generator_postprocess_stale_book_state() -> None:
    generator = FeatureGenerator(
        FeatureGeneratorConfig(book_state_max_age=pd.Timedelta(seconds=1))
    )
    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "market_id": "m1",
                "token_id": "yes",
                "_source_event_type": "book",
                "imbalance_1": 0.0,
                "imbalance_5": 0.0,
                "bid_depth_1": 10.0,
                "ask_depth_1": 20.0,
                "bid_depth_5": 50.0,
                "ask_depth_5": 40.0,
                "bid_ask_ratio": 1.0,
            },
            {
                "ts_event": base_time + timedelta(seconds=5),
                "market_id": "m1",
                "token_id": "yes",
                "_source_event_type": "price_change",
                "imbalance_1": 0.0,
                "imbalance_5": 0.0,
                "bid_depth_1": None,
                "ask_depth_1": None,
                "bid_depth_5": None,
                "ask_depth_5": None,
                "bid_ask_ratio": None,
            },
        ]
    )

    post = generator._postprocess_orderbook_features(frame)
    stale_row = post.iloc[1]
    assert pd.isna(stale_row["bid_depth_1"])
    assert pd.isna(stale_row["ask_depth_1"])


def test_feature_generator_process_and_generate_orderbook_features() -> None:
    generator = FeatureGenerator()
    base_time = datetime(2024, 1, 1, tzinfo=UTC)

    book_events = _book_events(base_time)
    processed = generator._process_book_events(book_events)
    assert len(processed) == 2
    assert processed[0]["token_id"] == "yes"

    generated = generator.generate_orderbook_features(book_events)
    assert not generated.empty
    assert {"spread", "mid_price", "imbalance_1"}.issubset(generated.columns)

    empty = generator.generate_orderbook_features(pd.DataFrame())
    assert empty.empty


def test_feature_generator_accepts_numpy_array_book_levels() -> None:
    generator = FeatureGenerator()
    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    events = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": "m1",
                "token_id": "yes",
                "data": {
                    "asset_id": "yes",
                    "bids": np.array([["0.49", "100"]], dtype=object),
                    "asks": np.array([["0.51", "100"]], dtype=object),
                },
            }
        ]
    ).set_index("ts_event")

    generated = generator.generate_orderbook_features(events)
    assert not generated.empty
    assert generated.iloc[0]["market_id"] == "m1"


def test_feature_generator_cached_paths(monkeypatch, tmp_path) -> None:
    cache_dir = tmp_path / "feature_cache"
    generator = FeatureGenerator(FeatureGeneratorConfig(cache_dir=cache_dir))
    events = _book_events(datetime(2024, 1, 1, tzinfo=UTC))

    first = generator.generate_orderbook_features_cached(events, cache_key="abc", use_cache=True)
    assert not first.empty

    original_read_parquet = pd.read_parquet

    def _raise_read(*args, **kwargs):
        raise RuntimeError("forced read failure")

    monkeypatch.setattr(pd, "read_parquet", _raise_read)
    second = generator.generate_orderbook_features_cached(events, cache_key="abc", use_cache=True)
    assert not second.empty
    monkeypatch.setattr(pd, "read_parquet", original_read_parquet)

    original_to_parquet = pd.DataFrame.to_parquet

    def _raise_write(self, *args, **kwargs):
        raise RuntimeError("forced write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _raise_write)
    third = generator.generate_orderbook_features_cached(events, cache_key="def", use_cache=False)
    assert not third.empty
    monkeypatch.setattr(pd.DataFrame, "to_parquet", original_to_parquet)


def test_feature_generator_trade_and_joined_dataset(monkeypatch) -> None:
    generator = FeatureGenerator()
    base_time = datetime(2024, 1, 1, tzinfo=UTC)

    market_events = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "event_type": "last_trade_price",
                "market_id": "m1",
                "token_id": "yes",
                "data": {"price": "0.5", "size": "2", "side": "BUY"},
            },
            {
                "ts_event": base_time + timedelta(seconds=30),
                "event_type": "last_trade_price",
                "market_id": "m1",
                "token_id": "yes",
                "data": {"price": "0.6", "size": "1", "side": "SELL"},
            },
            {
                "ts_event": base_time + timedelta(seconds=60),
                "event_type": "book",
                "market_id": "m1",
                "token_id": "yes",
                "data": {
                    "asset_id": "yes",
                    "bids": [{"price": "0.49", "size": "20"}],
                    "asks": [{"price": "0.51", "size": "20"}],
                },
            },
        ]
    ).set_index("ts_event")

    trade_features = generator.generate_trade_features(market_events, window="5min")
    assert not trade_features.empty
    assert {"buy_volume", "sell_volume", "trade_count"}.issubset(trade_features.columns)

    orderbook = pd.DataFrame(
        [
            {
                "ts_event": base_time + timedelta(seconds=60),
                "market_id": "m1",
                "token_id": "yes",
                "mid_price": 0.5,
                "spread": 0.02,
                "bid_depth_1": 20.0,
                "ask_depth_1": 20.0,
                "bid_depth_5": 20.0,
                "ask_depth_5": 20.0,
                "imbalance_1": 0.0,
                "imbalance_5": 0.0,
                "bid_ask_ratio": 1.0,
            }
        ]
    ).set_index("ts_event")

    monkeypatch.setattr(generator, "generate_orderbook_features", lambda _: orderbook)
    monkeypatch.setattr(generator, "generate_trade_features", lambda *_args, **_kwargs: trade_features)

    crypto_prices = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "symbol": "BTC",
                "data": {"price": 100.0},
            },
            {
                "ts_event": base_time + timedelta(seconds=60),
                "symbol": "BTC",
                "data": {"price": 101.0},
            },
        ]
    ).set_index("ts_event")

    combined = generator.build_feature_dataset_from_frames(
        market_events,
        crypto_prices,
        crypto_window="5min",
    )
    assert not combined.empty
    assert "crypto_price_change_pct" in combined.columns

    no_crypto = generator.build_feature_dataset_from_frames(
        market_events,
        pd.DataFrame(),
        crypto_window="5min",
    )
    assert not no_crypto.empty

    empty_market = generator.build_feature_dataset_from_frames(
        pd.DataFrame(),
        crypto_prices,
    )
    assert empty_market.empty


def test_load_market_events_with_row_limit_and_prefix(tmp_path) -> None:
    market_path = tmp_path / "pmxt"
    market_path.mkdir(parents=True, exist_ok=True)
    base_time = datetime(2024, 1, 1, tzinfo=UTC)

    raw = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": "btc-prefix-1",
                "token_id": "t1",
                "data": {"asset_id": "t1"},
            },
            {
                "ts_event": base_time + timedelta(seconds=1),
                "event_type": "book",
                "market_id": "eth-prefix-1",
                "token_id": "t2",
                "data": {"asset_id": "t2"},
            },
        ]
    )
    raw.to_parquet(market_path / "events.parquet", index=False)

    request = MarketEventsLoadRequest(
        market_path=market_path,
        start=None,
        end=None,
        limit_files=1,
        max_rows_per_file=1,
        market_slug_prefix="btc-prefix",
        is_pmxt_mode=False,
        mapping_path=tmp_path / "mapping",
    )
    deps = MarketEventsLoadDeps(
        load_condition_ids_for_slug_prefix_fn=lambda _: set(),
        normalize_market_events_schema_fn=lambda frame: frame,
        prepare_timestamp_index_fn=prepare_timestamp_index,
        filter_by_time_range_fn=filter_by_time_range,
    )
    loaded = load_market_events(
        request,
        deps=deps,
        modules=PyArrowModules(pa=None, pc=None, ds=None, pq=None),
    )
    assert len(loaded) == 1
    assert loaded.iloc[0]["market_id"].startswith("btc-prefix")


def test_risk_engine_gates_and_state_transitions() -> None:
    evaluator = RiskEvaluator(
        RiskLimits(
            max_drawdown_pct=0.10,
            max_daily_loss=5.0,
            max_concentration_pct=0.40,
            max_active_positions=1,
            max_gross_exposure=10.0,
        )
    )
    state = RiskState(
        gross_exposure=9.0,
        market_exposure={"m1": 9.0},
        active_positions={"m2"},
        realized_pnl_by_day={"2024-01-01": -10.0},
        equity=80.0,
        peak_equity=100.0,
    )
    now = datetime(2024, 1, 1, tzinfo=UTC)

    allowed, events = evaluator.evaluate_entry(
        timestamp=now,
        market_id="m1",
        requested_notional=2.0,
        state=state,
    )
    assert not allowed
    reasons = {event.reason for event in events}
    assert "gross_exposure_limit_exceeded" in reasons
    assert "active_positions_limit_exceeded" in reasons
    assert "market_concentration_limit_exceeded" in reasons
    assert "drawdown_limit_exceeded" in reasons
    assert "daily_loss_limit_exceeded" in reasons
    assert "gate" in events[0].to_dict()

    state2 = RiskState()
    evaluator2 = RiskEvaluator(RiskLimits())
    evaluator2.register_fill(market_id="m3", notional=4.0, state=state2)
    assert state2.gross_exposure == 4.0
    assert "m3" in state2.active_positions

    evaluator2.register_close(
        market_id="m3",
        resolved_at=now,
        net_pnl=-1.0,
        state=state2,
    )
    assert "m3" not in state2.active_positions
    assert state2.gross_exposure == 0.0

    evaluator2.register_realized_pnl(resolved_at=now, net_pnl=2.0, state=state2)
    assert state2.equity == 1.0
    assert state2.peak_equity >= state2.equity
