"""
Process a series of XRD images using the SeriesProcessor pipeline (script version).
"""

import logging
from pathlib import Path
import fabio
from xrd_decosmic.core.series_processor import SeriesProcessor, SeriesConfig

# Parameters
INPUT_FILE = "data/raw/first_image.tif"   # Path to the first image in the series
OUTPUT_DIR = "data/processed"             # Output directory
OUTPUT_PREFIX = "popc"                    # Output file prefix
USER_MASK = None                          # Path to user mask file (optional)
USE_FABIO = False                         # Use fabio for image loading

TH_DONUT = 15
TH_MASK = 0.05
TH_STREAK = 3
WIN_STREAK = 3
EXP_DONUT = 9
EXP_STREAK = 3

LOG_LEVEL = logging.INFO                  # Set to logging.DEBUG for verbose output

# Logging setup
logging.basicConfig(level=LOG_LEVEL, format='%(message)s')
logger = logging.getLogger("xrd_decosmic.scripts.main")

# Input checks
input_path = Path(INPUT_FILE).resolve()
if not input_path.exists() or not input_path.is_file():
    logger.error(f"Input file not found: {INPUT_FILE}")
    exit(1)

output_path = Path(OUTPUT_DIR).resolve()
output_path.mkdir(parents=True, exist_ok=True)

user_mask = None
if USER_MASK:
    mask_path = Path(USER_MASK).resolve()
    if not mask_path.exists() or not mask_path.is_file():
        logger.error(f"User mask file not found: {USER_MASK}")
        exit(1)
    user_mask = fabio.open(str(mask_path)).data.astype(bool)

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
logger.info(f"Initializing processor with input file: {input_path}")
processor = SeriesProcessor(
    str(input_path),
    series_config,
    user_mask,
    USE_FABIO
)

# Run processing pipeline
logger.info("Processing image series...")
series_result = processor.process_series()

# Save results
logger.info(f"Saving results to: {output_path} (prefix: {OUTPUT_PREFIX})")
series_result.save(str(output_path), OUTPUT_PREFIX)
logger.info("Processing complete.") 