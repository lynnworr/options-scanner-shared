from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
STOCK_PRICE_DIR = RAW_DIR / "stock_prices"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = BASE_DIR / "outputs"
BOARD_DIR = OUTPUT_DIR / "boards"
REPORT_DIR = OUTPUT_DIR / "reports"

PRICE_HISTORY_PERIOD = "2y"
PRICE_INTERVAL = "1d"

MIN_PRICE = 10.00
MAX_PRICE = 1000.00

MIN_AVG_DOLLAR_VOLUME = 25_000_000

BULLISH_SCORE_THRESHOLD = 70
BEARISH_SCORE_THRESHOLD = 70

# ============================================================
# OPTIONS SCANNER DTE SETTINGS
# ============================================================
# Short-term mode:
# 1 to 7 DTE means the scanner will look for trades expiring
# within the next week.
#
# I am intentionally not using 0DTE here because those trades
# behave very differently and need separate rules.
TARGET_DTE_MIN = 1
TARGET_DTE_MAX = 7