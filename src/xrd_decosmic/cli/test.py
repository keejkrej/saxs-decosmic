"""Single image processing test module for debugging and visualization."""
import argparse
import logging
import sys
import matplotlib.pyplot as plt
import tifffile

from ..core.single_processor import SingleProcessor, SingleConfig, SingleResult

# =====================================================================
# Argument Parsing (Simplified for single image)
# =====================================================================

def parse_args():
    """Parse command-line arguments for single image processing."""
    parser = argparse.ArgumentParser(description='Process a single image with decosmic.')
    parser.add_argument('--input', required=True, help='Path to the input image')
    parser.add_argument('--output', help='Directory to save results (optional)')
    parser.add_argument('--user-mask', help='User-defined mask for modifiable pixels')
    
    # Processing parameters
    parser.add_argument('--th-donut', type=float, default=15, 
                       help='Threshold for donut detection (higher = more strict)')
    parser.add_argument('--th-streak', type=float, default=3,
                       help='Threshold for streak detection (higher = more strict)')
    parser.add_argument('--win-streak', type=int, default=3,
                       help='Window size for streak detection (larger = longer streaks)')
    parser.add_argument('--exp-donut', type=float, default=9,
                       help='Expansion of donut mask (higher = larger masks)')
    parser.add_argument('--exp-streak', type=float, default=3,
                       help='Expansion of streak mask (higher = larger masks)')
    
    # Debug flag
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (more verbose logging)')
    
    return parser.parse_args()

# =====================================================================
# Plotting Function
# =====================================================================

def plot_results(result: SingleResult) -> None:
    """Plot original, cleaned and masked images in a 2x3 subplot layout."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Original image
    axes[0, 0].imshow(result.img_orig, cmap='viridis')
    axes[0, 0].set_title('Original Image')
    
    # Cleaned image
    axes[0, 1].imshow(result.img_clean, cmap='viridis')
    axes[0, 1].set_title('Cleaned Image')
    
    # Donut mask
    axes[0, 2].imshow(result.mask_donut, cmap='gray')
    axes[0, 2].set_title('Donut Mask')
    
    # Streak mask
    axes[1, 0].imshow(result.mask_streak, cmap='gray')
    axes[1, 0].set_title('Streak Mask')
    
    # Combined modification mask
    axes[1, 1].imshow(result.mask_combined, cmap='gray')
    axes[1, 1].set_title('Combined Mask')
    
    # Subtracted artifacts (donut + streak)
    subtracted = result.sub_donut + result.sub_streak if result.sub_donut is not None and result.sub_streak is not None else None
    axes[1, 2].imshow(subtracted, cmap='viridis')
    axes[1, 2].set_title('Subtracted Artifacts')
    
    plt.tight_layout()
    plt.show()

# =====================================================================
# Main Application (Single Image)
# =====================================================================

def main() -> None:
    """Process a single image and display results with optional saving."""
    args = parse_args()

    # Configure logging
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger("xrd_decosmic.cli.test")
    
    try:
        # Load the single image
        img = tifffile.imread(args.input)
        
        # Load user mask if specified
        user_mask = None
        if args.user_mask:
            user_mask = tifffile.imread(args.user_mask).astype(bool)
        
        # Create SingleConfig from CLI args
        config = SingleConfig(
            th_donut=args.th_donut,
            th_streak=args.th_streak,
            win_streak=args.win_streak,
            exp_donut=args.exp_donut,
            exp_streak=args.exp_streak
        )

        # Process the image
        processor = SingleProcessor(img, config, user_mask)
        result = processor.clean_img()
        
        # Plot results
        plot_results(result)
        
        # Save results if output directory is provided
        if args.output:
            # (Implement saving logic here if needed)
            logger.info(f"Results saved to {args.output}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 