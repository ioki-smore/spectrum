import signal
import sys

import polars as pl
import typer

from config import AppConfig, init_config
from core.discovery import IntervalDiscovery
from core.pipeline import Pipeline
from core.reporting import ReportHandler
from core.service import AnomalyDetectionService
from utils.errors import ErrorCode, handle_exception, status_to_exit_code
from utils.logger import get_logger

logger = get_logger("main")

app = typer.Typer(help="RueAI: Anomaly Detection System", no_args_is_help=True, add_completion=False,
                  context_settings={"help_option_names": ["-h", "--help"]})


@app.callback()
def prepare_config(ctx: typer.Context, config_path: str = typer.Option("config/config.yaml", "--config", "-c",
                                                                       help="Path to configuration file"), ):
    # Initialize config and store in context
    # NOTE: Exceptions will be propagated to the main handler
    res = init_config(config_path)
    if res.is_err():
        return res.err_value

    ctx.obj = res.unwrap()
    return None


@app.command()
def start(ctx: typer.Context):
    """Start the long-running service"""
    config: AppConfig = ctx.obj

    service = AnomalyDetectionService(config)

    # Register Signal Handlers
    def signal_handler(sig, _frame):
        signal_name = signal.Signals(sig).name
        logger.info(f"Received {signal_name}. Stopping...")
        service.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        service.run()
    except Exception as err:
        logger.error(f"Service crashed: {err}")
        return 1

    return 0


@app.command()
def train(ctx: typer.Context,
          interval: str = typer.Option("all", "--interval", help="Interval to train (default: all)")):
    """Trigger manual training"""
    config: AppConfig = ctx.obj
    discovery = IntervalDiscovery(config)
    intervals = discovery.discover()

    logger.info(f"Triggering manual training for interval: {interval}")

    if interval == 'all':
        results = {}
        for i in intervals:
            pipeline = Pipeline(i, config)
            res = pipeline.train()
            results[i] = res.err_value.value if res.is_err() else 0

        # Return 0 (OK) if at least one succeeded
        if any(code == 0 for code in results.values()):
            return 0
        return ErrorCode.MODEL_TRAIN_FAILED
    elif interval in intervals:
        pipeline = Pipeline(interval, config)
        res = pipeline.train()
        if res.is_err():
            logger.error(f"Training failed for {interval}: {res.err_value.message}")
            return res.err_value
        return 0
    else:
        logger.error(f"Interval '{interval}' not found. Available: {intervals}")
        return ErrorCode.CONFIG_INVALID


@app.command()
def detect(ctx: typer.Context,
           interval: str = typer.Option("all", "--interval", help="Interval to detect (default: all)")):
    """Trigger manual detection"""
    config: AppConfig = ctx.obj
    discovery = IntervalDiscovery(config)
    report_handler = ReportHandler(config)
    intervals = discovery.discover()

    logger.info(f"Triggering manual detection for interval: {interval}")

    events = []
    code = 0

    if interval == 'all':
        for i in intervals:
            pipeline = Pipeline(i, config)
            res = pipeline.detect()
            if res.is_ok():
                events.extend(res.unwrap())
            else:
                logger.error(f"Detection failed for {i}: {res.err_value}")
    elif interval in intervals:
        pipeline = Pipeline(interval, config)
        res = pipeline.detect()
        if res.is_ok():
            events = res.unwrap()
        else:
            logger.error(f"Detection failed for {interval}: {res.err_value}")
            code = res.err_value
    else:
        logger.error(f"Interval '{interval}' not found. Available: {intervals}")
        return ErrorCode.CONFIG_INVALID

    if events:
        # Save to summary file
        report_handler.append(events)

        # Configure polars to display all columns and wide output
        with pl.Config(tbl_rows=100, tbl_cols=10, tbl_width_chars=120):
            df = pl.DataFrame(events)
            print("\n=== Detection Results ===")
            print(df)
            print("=========================\n")
    else:
        print("\nNo anomalies detected.\n")

    return code


if __name__ == "__main__":
    try:
        # standalone_mode=False allows exceptions to propagate to our handler
        # and returns the value from the command function
        result = app(standalone_mode=False)

        exit_code = 0
        if isinstance(result, ErrorCode):
            exit_code = status_to_exit_code(result)
        elif isinstance(result, int):
            exit_code = result
        elif result is not None:
            logger.warning(f"Unknown result type: {type(result)}")

        sys.exit(exit_code)

    except BaseException as e:
        sys.exit(handle_exception(e, logger=logger))
