# Script to process a series of XRD images using the SeriesProcessor pipeline
from pathlib import Path
from xrd_decosmic.core.series_processor import SeriesProcessor, SeriesConfig

# === Configuration ===
# Folder containing raw .tif images to process
RAW_FOLDER = "data/raw"
# Folder where processed results will be saved
SAVE_FOLDER = "data/processed"
# Prefix for output file names
SAVE_NAME_PREFIX = "popc"

# === Discover input files ===
# Find all .tif files in the raw folder, sorted alphabetically
raw_files = sorted(list(Path(RAW_FOLDER).expanduser().glob("*.tif")))

# === Set up processing configuration ===
# SeriesConfig extends SingleConfig and adds th_mask for mask thresholding
series_config = SeriesConfig(
    th_donut=15,      # Threshold for donut feature removal
    th_mask=0.05,     # Threshold for binary average mask (fraction of frames)
    th_streak=3,      # Threshold for streak feature removal
    win_streak=3,     # Window size for streak detection
    exp_donut=9,      # Expansion size for donut mask
    exp_streak=3      # Expansion size for streak mask
)

# === Initialize processor ===
# Only the first file is needed to initialize; the processor will find the rest
processor = SeriesProcessor(
    str(raw_files[0]),   # Path to the first image in the series
    series_config,       # Processing configuration
)

# === Run the processing pipeline ===
# This will perform averaging, masking, cleaning, and variance calculations
series_result = processor.process_series()

# === Save results ===
# All result arrays will be saved as TIFF files in the output directory
series_result.save(SAVE_FOLDER, SAVE_NAME_PREFIX)