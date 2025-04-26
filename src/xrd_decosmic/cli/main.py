"""
Main entry point for the CLI application.

This module provides the command-line interface for the XRD cosmic ray removal
tool. It handles argument parsing, parameter loading, and orchestrates the
processing of image series.
"""
import argparse
import json
import sys
import fabio
from typing import Optional, Dict, Any

from ..core import ProcessingParams, SeriesProcessor

# =====================================================================
# Argument Parsing
# =====================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with parameter descriptions.
    
    This function creates and configures the argument parser with all
    necessary command-line arguments for the application.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(description='Remove cosmic background from XRD 2D images')
    
    # Required arguments
    parser.add_argument('input_file', help='First image in the series')
    parser.add_argument('--output-dir', required=True, help='Directory to save results')
    
    # Optional arguments
    parser.add_argument('--params-file', help='JSON file containing processing parameters')
    parser.add_argument('--user-mask', help='User-defined mask for modifiable pixels')
    parser.add_argument('--use-fabio', action='store_true', 
                       help='Use fabio to load image series (more efficient for certain formats)')
    
    # Processing parameters (optional, will use defaults if not specified)
    parser.add_argument('--th-donut', help='Threshold for donut detection (higher = more strict)')
    parser.add_argument('--th-mask', help='Threshold for mask creation (0-1, higher = larger masks)')
    parser.add_argument('--th-streak', help='Threshold for streak detection (higher = more strict)')
    parser.add_argument('--win-streak', help='Window size for streak detection (larger = longer streaks)')
    parser.add_argument('--exp-donut', help='Expansion of donut mask (higher = larger masks)')
    parser.add_argument('--exp-streak', help='Expansion of streak mask (higher = larger masks)')
    
    return parser

# =====================================================================
# Parameter Handling
# =====================================================================

def get_params(args: argparse.Namespace) -> ProcessingParams:
    """Get parameters from either file or command line arguments.
    
    This function retrieves processing parameters from either a JSON file
    or command-line arguments. It starts with default values from
    ProcessingParams and modifies them based on the provided arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        ProcessingParams: Instance with validated parameters
        
    Raises:
        ValueError: If parameters are invalid
        FileNotFoundError: If params_file is specified but not found
        json.JSONDecodeError: If params_file contains invalid JSON
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
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Get parameters with validation
        params_model = get_params(args)

        # Load user mask if specified
        if args.user_mask:
            user_mask = fabio.open(args.user_mask).data
            user_mask = user_mask.astype(bool)
        else:
            user_mask = None
        
        # Process images
        processor = SeriesProcessor(args.input_file, user_mask=user_mask, use_fabio=args.use_fabio)
        processor.load_params(params_model)
        processor.avg_clean_img(lambda p: print(f'Progress: {p}%'))
        
        # Save results
        processor.save_results(args.output_dir)
        print(f'Results saved to {args.output_dir}')
        
        # Free up memory
        del processor
        
    except Exception as e:
        print(f'Error: {str(e)}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 