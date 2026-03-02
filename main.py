import asyncio
import contextlib
import logging
import os
import signal
from datetime import UTC, datetime

from config import STREAM_FEEDS_DIR
from data.historical import OHLCVParams, fetch_historical_data
from data.reference import get_updown_asset_ids_with_slug
from data.stream import StreamStorageSink, stream_crypto_prices, stream_polymarket_data
from utils import datetime_to_timestamp_ms, setup_application_logging, validate_crypto_price_data

# Configure logging
setup_application_logging()

logger = logging.getLogger(__name__)


def get_historical() -> None:
    logger.info("Starting historical data fetch...")
    tzinfo = UTC
    start = datetime(2021, 1, 30, 0, 0, 0, tzinfo=tzinfo)
    end = datetime(2021, 12, 30, 23, 59, 59, tzinfo=tzinfo)
    params = OHLCVParams(
        symbol="BTC/USDT",
        start_ms=datetime_to_timestamp_ms(start),
        end_ms=datetime_to_timestamp_ms(end),
        timeframe="1s",
        limit=1000,
        exchanges=["binance"]
    )
    data = fetch_historical_data(params)
    if data is not None:
        logger.info("Fetched %d rows of data for %s.", len(data), params.symbol)
    else:
        logger.warning("No data fetched for %s.", params.symbol)


async def stream_crypto_feeds(storage_sink: StreamStorageSink) -> None:
    """Stream live crypto prices from RTDS."""
    logger.info("Starting crypto price stream...")
    source = "chainlink"
    symbols = ["btc/usd"]

    def on_crypto_event(data: dict | list) -> None:
        """Handle incoming crypto price events from the stream."""
        try:
            if not storage_sink:
                logger.warning("Storage sink is None, cannot write crypto price")
                return

            price_data = validate_crypto_price_data(data)
            if price_data is None:
                return

            logger.debug("Writing crypto price: symbol=%s, price=%s, timestamp=%s",
                        price_data["symbol"], price_data["price"], price_data["timestamp"])
            storage_sink.write_crypto_price(price_data, source=source)

        except Exception:
            logger.exception("Unexpected error in on_crypto_event")

    try:
        await stream_crypto_prices(
            symbols=symbols,
            source=source,
            callback=on_crypto_event,
        )
    except asyncio.CancelledError:
        logger.info("Crypto stream cancelled")
        raise
    except Exception:
        logger.exception("Error in crypto stream")
        raise


def _handle_market_event(
    data: dict | list,
    storage_sink: StreamStorageSink,
    stream_task: asyncio.Task | None,
    market_slug: str | None = None,
) -> None:
    """Handle incoming market events and store them.

    Parameters
    ----------
    data : dict | list
        Market event data from websocket.
    storage_sink : StreamStorageSink
        Storage sink for persisting events.
    stream_task : asyncio.Task | None
        Current stream task to cancel on market resolution.
    market_slug : str, optional
        Market slug for organizing orderbook data.
    """
    if not isinstance(data, dict):
        return

    # Store the event (with market slug for proper file organization)
    if storage_sink:
        storage_sink.write_market_event(data, market_slug=market_slug)

    # Detect market resolution and handle it
    if data.get("event_type") == "market_resolved":
        market_id = data.get("market")
        resolved_slug = data.get("slug", market_slug)
        logger.info(
            "Market resolved: %s (ID: %s, slug: %s). "
            "Flushing market buffer and cancelling stream...",
            data.get("id", "unknown"),
            market_id,
            resolved_slug,
        )
        # Flush the market's buffer to ensure all data is persisted
        if storage_sink and market_id:
            storage_sink.handle_market_resolved(market_id, market_slug=resolved_slug)
        # Cancel the current stream to fetch new asset IDs
        if stream_task:
            stream_task.cancel()


async def _fetch_and_stream_market_assets(
    resolution: str, storage_sink: StreamStorageSink
) -> None:
    """Fetch asset IDs and stream market data."""
    try:
        while True:
            utctime = int(datetime.now(tz=UTC).timestamp())
            market_slug, asset_ids = get_updown_asset_ids_with_slug(
                utctime=utctime, resolution=resolution
            )

            if not asset_ids:
                logger.error("No asset IDs found for resolution %s", resolution)
                await asyncio.sleep(5)
                continue

            logger.info("Streaming market asset IDs for slug %s: %s", market_slug, asset_ids)

            # Use a list to hold the task so the callback can access it
            task_holder: list[asyncio.Task | None] = [None]

            def on_market_event(
                data: dict | list,
                holder: list = task_holder,
                slug: str = market_slug,
            ) -> None:
                """Handle incoming market events."""
                _handle_market_event(data, storage_sink, holder[0], market_slug=slug)

            stream_task = asyncio.create_task(
                stream_polymarket_data(
                    asset_ids=asset_ids,
                    enable_custom_features=True,
                    callback=on_market_event,
                )
            )
            task_holder[0] = stream_task

            try:
                await stream_task
            except asyncio.CancelledError:
                logger.info("Market stream cancelled")
                raise
    except asyncio.CancelledError:
        logger.info("Market asset refresh cancelled")
        raise


async def stream_market_feeds(storage_sink: StreamStorageSink, resolution: str = "5m") -> None:
    """Stream Polymarket orderbook data."""
    logger.info("Starting market data stream (resolution: %s)...", resolution)
    try:
        await _fetch_and_stream_market_assets(resolution, storage_sink)
    except asyncio.CancelledError:
        logger.info("Market stream cancelled")
        raise
    except Exception:
        logger.exception("Error in market stream")
        raise


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    hard_exit_requested = {"value": False}

    def _request_shutdown(signal_name: str) -> None:
        if stop_event.is_set() or hard_exit_requested["value"]:
            logger.warning("Hard exit requested (%s)", signal_name)
            os._exit(1)
        logger.info("Shutdown requested (%s)", signal_name)
        stop_event.set()
        hard_exit_requested["value"] = True

    def _signal_handler(signum: int, _frame: object) -> None:
        signal_name = signal.Signals(signum).name
        loop.call_soon_threadsafe(_request_shutdown, signal_name)

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _request_shutdown, sig.name)
            continue
        with contextlib.suppress(ValueError):
            signal.signal(sig, _signal_handler)


def _raise_if_exception(task: asyncio.Task[object]) -> None:
    exc = task.exception()
    if exc is not None:
        raise exc


async def _drain_tasks(tasks: set[asyncio.Task[object]]) -> None:
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


async def _run_until_shutdown(
    tasks: list[asyncio.Task[object]],
    stop_event: asyncio.Event,
) -> None:
    shutdown_task: asyncio.Task[bool] = asyncio.create_task(stop_event.wait())
    pending: set[asyncio.Task[object]] = set(tasks)
    pending.add(shutdown_task)

    while pending:
        done, pending = await asyncio.wait(
            pending,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if shutdown_task in done:
            break

        for task in done:
            _raise_if_exception(task)
            stop_event.set()
            break

    for task in pending:
        task.cancel()
    await _drain_tasks(pending)


async def stream_all_feeds(resolution: str = "5m") -> None:
    """Stream both crypto and market feeds concurrently."""
    logger.info("=== Starting data feed streams (crypto + market) ===")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop, stop_event)

    # Initialize single storage sink for both feeds
    storage_sink = StreamStorageSink(
        base_path=STREAM_FEEDS_DIR, partition_by="hour", buffer_size=50, crypto_buffer_size=5
    )

    try:
        # Create tasks for both streams
        crypto_task: asyncio.Task[object] = asyncio.create_task(stream_crypto_feeds(storage_sink))
        market_task: asyncio.Task[object] = asyncio.create_task(
            stream_market_feeds(storage_sink, resolution)
        )
        await _run_until_shutdown([crypto_task, market_task], stop_event)

    except asyncio.CancelledError:
        logger.info("All streams cancelled")
        raise
    except Exception:
        logger.exception("Error in data feed streams")
    finally:
        if storage_sink:
            logger.info("Closing storage sink...")
            storage_sink.close()


if __name__ == "__main__":
    # Main operation: Stream both crypto and market feeds
    asyncio.run(stream_all_feeds(resolution="5m"))
