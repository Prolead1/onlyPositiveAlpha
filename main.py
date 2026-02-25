import asyncio
import logging
from datetime import UTC, datetime

from data.historical import OHLCVParams, fetch_historical_data
from data.reference import get_updown_asset_ids
from data.stream import stream_crypto_prices, stream_polymarket_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


def get_historical() -> None:
    logger.info("Starting historical data fetch...")
    tzinfo = UTC
    start = datetime(2021, 1, 30, 0, 0, 0, tzinfo=tzinfo)
    end = datetime(2021, 12, 30, 23, 59, 59, tzinfo=tzinfo)
    params = OHLCVParams(
        symbol="BTC/USDT",
        start_ms=int(start.timestamp() * 1000),
        end_ms=int(end.timestamp() * 1000),
        timeframe="1s",
        limit=1000,
        exchanges=["binance"]
    )
    data = fetch_historical_data(params)
    if data is not None:
        logger.info("Fetched %d rows of data for %s.", len(data), params.symbol)
    else:
        logger.warning("No data fetched for %s.", params.symbol)


def custom_crypto_callback(data: dict | list) -> None:
    pass

def get_crypto_stream() -> None:
    logger.info("Starting live Bitcoin price stream from Polymarket RTDS...")
    source = "chainlink"
    symbols = ["btc/usd"]

    asyncio.run(stream_crypto_prices(
        symbols=symbols,
        source=source
        ))


def custom_market_callback(data: dict | list) -> None:
    pass


async def stream_bitcoin_updown(resolution: str = "5m") -> None:
    try:
        logger.info("Starting Polymarket Bitcoin up-down market stream...")
        utctime = int(datetime.now(tz=UTC).timestamp())
        asset_ids = get_updown_asset_ids(utctime=utctime, resolution=resolution)

        if not asset_ids:
            logger.error("No asset IDs found for resolution %s", resolution)
            return

        logger.info("Streaming asset IDs: %s", asset_ids)

        await stream_polymarket_data(
            asset_ids=asset_ids,
            enable_custom_features=True
        )
    except KeyboardInterrupt:
        logger.info("Stream interrupted by user")
    except Exception:
        logger.exception("Error in market stream")


if __name__ == "__main__":
    # Options for streaming data:
    # 1. Fetch historical data: get_historical()
    # 2. Stream crypto prices from RTDS: get_crypto_stream()
    # 3. Stream Polymarket orderbook data (current):
    asyncio.run(stream_bitcoin_updown(resolution="5m"))

