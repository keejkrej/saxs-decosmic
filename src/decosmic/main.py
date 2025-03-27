"""
Main entry point for the XRD Decosmic package.
"""

def main():
    """
    Display usage information for the XRD Decosmic package.
    """
    print("XRD Decosmic - Cosmic background removal from XRD 2d images")
    print("\nUsage:")
    print("  To run the GUI application:")
    print("    python -m decosmic.gui")
    print("    xrd-decosmic-gui")
    print("\n  To run the command-line interface:")
    print("    python -m decosmic.cli input_file.edf --output-dir ./output [options]")
    print("    xrd-decosmic-cli input_file.edf --output-dir ./output [options]")
    print("\nOptions:")
    print("  --params-file FILE    Load parameters from JSON file")
    print("  --th-donut VALUE      Threshold for donut detection (default: 15)")
    print("  --th-mask VALUE       Threshold for ring mask creation (default: 0.05)")
    print("  --th-streak VALUE     Threshold for streak detection (default: 3)")
    print("  --win-streak VALUE    Window size for streak detection (default: 3)")
    print("  --exp-donut VALUE     Expansion of donut mask (default: 9)")
    print("  --exp-streak VALUE    Expansion of streak mask (default: 3)")
    print("\nFor more information, see the README or documentation.")

if __name__ == "__main__":
    main()