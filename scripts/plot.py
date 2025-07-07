"""
Plot a series of processed SAXS images as PNGs with percentile-based contrast.
"""

import matplotlib.pyplot as plt
import tifffile
import numpy as np
from pathlib import Path
import logging

# Parameters
OUTPUT_DIR = "data/processed"             # Output directory
OUTPUT_PREFIX = "test"                    # Output file prefix

LOG_LEVEL = logging.INFO                  # Set to logging.DEBUG for verbose output
MIN_PERCENTILE = 10
MAX_PERCENTILE = 90
COLORMAP = 'inferno'

# Logging setup
logging.basicConfig(level=LOG_LEVEL, format='%(message)s')
logger = logging.getLogger(__name__)

output_path = Path(OUTPUT_DIR).resolve()
logger.info(f"Looking for files in: {output_path} (prefix: {OUTPUT_PREFIX})")
# Find all tif files with the given prefix
# (e.g., test_0000.tif, test_0001.tif, ...)
tif_files = sorted(output_path.glob(f"{OUTPUT_PREFIX}_*.tif"))

# Check if any files were found
if not tif_files:
    logger.warning(f"No files found in {output_path} with prefix '{OUTPUT_PREFIX}_'")
else:
    logger.info(f"Found {len(tif_files)} files. Generating plots...")
    # Loop over each tif file and plot
    for tif_file in tif_files:
        logger.info(f"Processing: {tif_file.name}")
        img = tifffile.imread(tif_file).astype(np.float64)
        # Compute 10th and 90th percentiles for clever vmin/vmax (contrast)
        vmin = np.percentile(img, MIN_PERCENTILE)
        vmax = np.percentile(img, MAX_PERCENTILE)
        plt.figure()
        plt.imshow(img, cmap=COLORMAP, vmin=vmin, vmax=vmax)
        plt.title(tif_file.name)
        plt.colorbar()
        # Save as PNG with same name as tif, but .png extension
        png_name = tif_file.with_suffix('.png').name
        png_path = output_path / png_name
        plt.savefig(png_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {png_path}")
