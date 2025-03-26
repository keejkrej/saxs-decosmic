"""
Main entry point for the CLI application.
"""
import argparse
import json
import sys

from ..core.processing_params import ProcessingParams
from ..core.series_processor import SeriesProcessor

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with parameter descriptions."""
    parser = argparse.ArgumentParser(description='Remove cosmic background from XRD 2D images')
    parser.add_argument('input_file', help='First image in the series')
    parser.add_argument('--output-dir', required=True, help='Directory to save results')
    parser.add_argument('--params-file', help='JSON file containing processing parameters')
    
    # Add parameter arguments without defaults - they'll use ProcessingParams defaults if not specified
    parser.add_argument('--th-donut', help='Threshold for donut detection')
    parser.add_argument('--th-mask', help='Threshold for mask creation (0-1)')
    parser.add_argument('--th-streak', help='Threshold for streak detection')
    parser.add_argument('--win-streak', help='Window size for streak detection')
    parser.add_argument('--exp-donut', help='Exponent for donut detection')
    parser.add_argument('--exp-streak', help='Exponent for streak detection')
    
    return parser

def get_params(args: argparse.Namespace) -> ProcessingParams:
    """
    Get parameters from either file or command line arguments.
    Starts with ProcessingParams defaults and modifies as needed.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        ProcessingParams instance with validated parameters
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Start with default ProcessingParams instance
    params = ProcessingParams()
    
    if args.params_file:
        # Load parameters from file
        with open(args.params_file, 'r') as f:
            file_params = json.load(f)
            # Update params with file values
            for key, value in file_params.items():
                setattr(params, key, value)
    else:
        # Check each CLI argument and update if specified
        if args.th_donut is not None:
            params.th_donut = args.th_donut
        if args.th_mask is not None:
            params.th_mask = args.th_mask
        if args.th_streak is not None:
            params.th_streak = args.th_streak
        if args.win_streak is not None:
            params.win_streak = args.win_streak
        if args.exp_donut is not None:
            params.exp_donut = args.exp_donut
        if args.exp_streak is not None:
            params.exp_streak = args.exp_streak
    
    # Validate parameters before returning
    params.validate()
    return params

def main():
    """Main entry point for the CLI application."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Get parameters with validation
        params_model = get_params(args)
        
        # Process images
        processor = SeriesProcessor(args.input_file)
        processor.load_params(params_model)
        processor.avg_clean_img(lambda i: print(f'Progress: {int(i/processor.img_num*100)}%'))
        
        # Save results
        processor.save_results(args.output_dir)
        print(f'Results saved to {args.output_dir}')
        
    except Exception as e:
        print(f'Error: {str(e)}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 