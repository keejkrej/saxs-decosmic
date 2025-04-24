# XRD Decosmic GUI

This module implements the graphical user interface for the XRD Decosmic application using a Model-View-Controller (MVC) architecture with PySide6 widgets.

## Architecture Overview

### Model-View-Controller

The application follows the MVC pattern:

- **Models**: Data structures and business logic located in `models/`
- **Views**: User interface components using PySide6 widgets in `views/`
- **Controllers**: Logic to connect models and views in `controllers/`

### Widget-based User Interface with PySide6

The GUI uses Qt's widget system with PySide6, which provides:

- Native look and feel across platforms
- Efficient and responsive UI
- Rich set of pre-built widgets
- Extensive customization options
- Compatibility with Qt6, the latest Qt release
- Permissive LGPL licensing

## Directory Structure

```
gui/
├── __init__.py
├── __main__.py
├── main.py               # Entry point for the GUI application
├── controllers/          # Controller components
│   ├── __init__.py
│   └── main_controller.py
├── models/              # Model components
│   ├── __init__.py
│   └── processing_model.py
└── views/               # View components
    ├── __init__.py
    ├── main_view.py     # Main window widget
    ├── plot_view.py     # Plot visualization widget
    └── file_dialog.py   # File dialog helper
```

## Integration Design

The application uses a straightforward widget-based approach:

1. The `MainView` class creates and manages the main window UI
2. Qt signals and slots handle communication between components
3. Direct method calls for synchronous operations
4. Custom widgets for specialized functionality (e.g., plotting)

## Key Components

- **MainView**: Main window widget that creates and manages the UI
- **PlotView**: Matplotlib-based plotting widget
- **FileDialog**: Helper class for file operations
- **ProcessingModel**: Core data processing and state management
- **MainController**: Connects UI events to model operations

## PySide6 Integration

This application uses PySide6 for several advantages:

1. **License**: PySide6 is available under the more permissive LGPL license
2. **Latest Qt**: PySide6 provides access to Qt6, the latest version of Qt
3. **API**: Uses a more Pythonic naming convention (e.g., `Signal` instead of `pyqtSignal`)
4. **Performance**: Native widgets provide optimal performance
5. **Portability**: Works consistently across platforms

## Usage

To run the GUI application:

```python
from decosmic.gui import main

if __name__ == "__main__":
    main.main()
```

Or use the module entry point:

```
python -m decosmic.gui
```

## Dependencies

- PySide6: Qt for Python
- Matplotlib: For plotting
- NumPy: For numerical operations 