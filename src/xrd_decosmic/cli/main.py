"""
Main entry point for the CLI application.
This module provides the command-line interface for the XRD cosmic ray removal
tool. It handles argument parsing, parameter loading, and orchestrates the
processing of image series.
"""
import argparse
import logging
import sys
import fabio
from ..core import SeriesProcessor
# =====================================================================
# Argument Parsing
# =====================================================================
def parse_args():
    """Create and configure the argument parser.
    This function sets up the argument parser with all required and optional
    command-line arguments needed for the XRD cosmic ray removal tool.
    Returns:
        argparse.ArgumentParser: Configured parser with all arguments defined
    """
    parser = argparse.ArgumentParser(description='Remove cosmic background from XRD 2D images')
    # Required arguments
    parser.add_argument('--input', required=True, help='First image in the series')
    parser.add_argument('--output', required=True, help='Directory to save results')
    # Optional arguments
    parser.add_argument('--user-mask', help='User-defined mask for modifiable pixels')
    parser.add_argument('--use-fabio', action='store_true', 
                       help='Use fabio to load image series (more efficient for certain formats)')
    # Processing parameters (optional, will use defaults if not specified)
    parser.add_argument('--th-donut', type=float, default=15, 
                       help='Threshold for donut detection (higher = more strict)')
    parser.add_argument('--th-mask', type=float, default=0.05,
                       help='Threshold for mask creation (0-1, higher = larger masks)')
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
# Main Application
# =====================================================================
def main() -> None:
    """Main entry point for the CLI application.
    This function orchestrates the entire processing workflow:
    1. Parse command-line arguments
    2. Load and validate parameters
    3. Load user mask if specified
    4. Process the image series
    5. Save results
    6. Clean up resources
    The function handles errors gracefully and provides appropriate
    error messages to the user.
    """
    args = parse_args()
    # Configure logging
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    # Set package logger level before getting logger instances
    logging.getLogger("xrd_decosmic").setLevel(logging.DEBUG if args.debug else logging.INFO)
    # Get the logger instance with explicit package path
    logger = logging.getLogger("xrd_decosmic.cli.main")
    try:
        # Load user mask if specified
        if args.user_mask:
            user_mask = fabio.open(args.user_mask).data
            user_mask = user_mask.astype(bool)
        else:
            user_mask = None
        # Process images
        processor = SeriesProcessor(args.input,
                                    th_donut=args.th_donut,
                                    th_mask=args.th_mask,
                                    th_streak=args.th_streak,
                                    win_streak=args.win_streak,
                                    exp_donut=args.exp_donut,
                                    exp_streak=args.exp_streak,
                                    user_mask=user_mask,
                                    use_fabio=args.use_fabio)
        processor.process()
        # Save results
        processor.save_results(args.output)
        logger.info(f'Results saved to {args.output}')
        # Free up memory
        processor.cleanup()
    except Exception as e:
        logger.error(f'Error: {e}')
        sys.exit(1)
if __name__ == '__main__':
    main() 