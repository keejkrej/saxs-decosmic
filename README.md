# XRD Decosmic

A tool for removing cosmic background from XRD 2D images.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/xrd-decosmic.git
cd xrd-decosmic

# Install in development mode
pip install -e .

# Install with test dependencies
pip install -e ".[test]"

# Install with development tools
pip install -e ".[dev]"
```

## Usage

Running the main package will display usage information:

```bash
python -m decosmic
xrd-decosmic
```

### GUI Application

Run the GUI application using any of these methods:

```bash
# As a module
python -m decosmic.gui

# Using the dedicated script
xrd-decosmic-gui
```

In the GUI:
1. File > Load Data - Select first image in the series
2. File > Load Parameters (Optional)
3. Click "Start" to process the images
4. Select plot type
5. Click "Plot" to visualize the results
6. File > Save Data - Select folder to save the processed data
7. File > Save Parameters (Optional) - Save your parameter settings

### Command Line Interface

Run the CLI using any of these methods:

```bash
# As a module
python -m decosmic.cli input_file.edf --output-dir ./output

# Using the dedicated script
xrd-decosmic-cli input_file.edf --output-dir ./output

# With custom parameters
python -m decosmic.cli input_file.edf --output-dir ./output --th-donut 20 --th-mask 0.2
```

### Notebook

Example notebook is available at:
```
notebook/example.ipynb
```

When using the notebook, change the file name accordingly, just specify the first image in series:

Windows:
```python
fileName = "P:/user/Your-Name/Your-Project/Your-Image.tif"
```

Jupyter Hub:
```python
fileName = "/project/ag-nickel/user/Your-Name/Your-Project/Your-Image.tif"
```

## Project Structure

```
xrd-decosmic/
├── src/                    # Source code
│   └── decosmic/          # Main package
│       ├── cli/           # Command line interface
│       ├── gui/           # GUI components
│       ├── models/        # Core processing models
│       └── ui/            # UI definition files
├── tests/                 # Test suite
├── notebook/              # Jupyter notebooks
├── pyproject.toml         # Project metadata and dependencies
├── setup.py               # Setup script for backward compatibility
└── README.md              # This file
```

## Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/decosmic
```