"""
Process a series of XRD images using the SeriesProcessor pipeline.
"""

import logging
from pathlib import Path
from xrd_decosmic.core.series_processor import SeriesProcessor, SeriesConfig

# Parameters
RAW_FOLDER = "data/raw"           # Raw .tif images folder
SAVE_FOLDER = "data/processed"    # Output folder
SAVE_NAME_PREFIX = "popc"         # Output file prefix

TH_DONUT = 15
TH_MASK = 0.05
TH_STREAK = 3
WIN_STREAK = 3
EXP_DONUT = 9
EXP_STREAK = 3

LOG_LEVEL = logging.INFO           # Set to logging.DEBUG for verbose output

# Logging setup
logging.basicConfig(level=LOG_LEVEL, format='%(message)s')
logger = logging.getLogger("xrd_decosmic.scripts.process")

# Input checks
raw_path = Path(RAW_FOLDER).resolve()
if not raw_path.exists() or not raw_path.is_dir():
    logger.error(f"Raw folder not found: {RAW_FOLDER}")
    exit(1)

raw_files = sorted(list(raw_path.glob("*.tif")))
if not raw_files:
    logger.error(f"No .tif files found in: {RAW_FOLDER}")
    exit(1)

# Processing configuration
series_config = SeriesConfig(
    th_donut=TH_DONUT,
    th_mask=TH_MASK,
    th_streak=TH_STREAK,
    win_streak=WIN_STREAK,
    exp_donut=EXP_DONUT,
    exp_streak=EXP_STREAK
)

# Initialize processor
logger.info(f"Initializing processor with first file: {raw_files[0]}")
processor = SeriesProcessor(
    str(raw_files[0]),
    series_config,
)

# Run processing pipeline
logger.info("Processing image series...")
series_result = processor.process_series()

# Save results
logger.info(f"Saving results to: {SAVE_FOLDER} (prefix: {SAVE_NAME_PREFIX})")
series_result.save(SAVE_FOLDER, SAVE_NAME_PREFIX)
logger.info("Processing complete.")