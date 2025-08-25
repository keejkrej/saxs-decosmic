"""
PySide6 GUI to run the SeriesProcessor pipeline (feature parity with scripts/process.py).

Features:
- Pick INPUT_FILE (first image in series)
- Pick OUTPUT_DIR
- Set OUTPUT_PREFIX
- Pick USER_MASK (optional)
- Configure USE_FABIO, AVG_CLEAN_ONLY
- Configure thresholds: TH_DONUT, TH_MASK, TH_STREAK, WIN_STREAK, EXP_DONUT, EXP_STREAK
- Optional: Plot saved .tif results to .png using 10/90 percentiles

Run executes the same pipeline and saves results to OUTPUT_DIR with OUTPUT_PREFIX.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import multiprocessing as mp

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import fabio
from saxs_decosmic.core.series_processor import SeriesProcessor, SeriesConfig

# When PyInstaller builds a windowed app (console disabled), sys.stdout and
# sys.stderr may be None. Some libraries try to write to them and cause
# "'NoneType' object has no attribute 'write'". Coerce them to a harmless
# file object pointing to the OS null device if they're None.
import sys
import os
# Keep references to devnull file objects so we can close them on exit
_devnull_stdout = None
_devnull_stderr = None
if getattr(sys, 'stdout', None) is None:
    _devnull_stdout = open(os.devnull, 'w')
    sys.stdout = _devnull_stdout
if getattr(sys, 'stderr', None) is None:
    _devnull_stderr = open(os.devnull, 'w')
    sys.stderr = _devnull_stderr


class ProcessingWorker(QThread):
    """Background worker that runs the processing pipeline to keep UI responsive."""

    progressed: Signal = Signal(str)
    finished_ok: Signal = Signal(str)
    failed: Signal = Signal(str)

    def __init__(
        self,
        input_file: str,
        output_dir: str,
        output_prefix: str,
        user_mask_path: str | None,
        use_fabio: bool,
        avg_clean_only: bool,
        th_donut: int,
        th_mask: float,
        th_streak: int,
        win_streak: int,
        exp_donut: int,
        exp_streak: int,
        plot_after: bool,
    ) -> None:
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.user_mask_path = user_mask_path
        self.use_fabio = use_fabio
        self.avg_clean_only = avg_clean_only
        self.th_donut = th_donut
        self.th_mask = th_mask
        self.th_streak = th_streak
        self.win_streak = win_streak
        self.exp_donut = exp_donut
        self.exp_streak = exp_streak
        self.plot_after = plot_after

    def run(self) -> None:
        try:
            self.progressed.emit("Preparing configuration and IO paths ...")

            # Resolve IO
            input_path = Path(self.input_file).resolve()
            if not input_path.exists() or not input_path.is_file():
                raise FileNotFoundError(f"Input file not found: {self.input_file}")

            output_path = Path(self.output_dir).resolve()
            output_path.mkdir(parents=True, exist_ok=True)

            # Load user mask if any
            mask_array: np.ndarray | None = None
            if self.user_mask_path:
                mask_path = Path(self.user_mask_path).resolve()
                if not mask_path.exists() or not mask_path.is_file():
                    raise FileNotFoundError(f"User mask file not found: {self.user_mask_path}")
                mask_array = fabio.open(str(mask_path)).data.astype(bool)

            # Build config
            series_config = SeriesConfig(
                th_donut=self.th_donut,
                th_mask=self.th_mask,
                th_streak=self.th_streak,
                win_streak=self.win_streak,
                exp_donut=self.exp_donut,
                exp_streak=self.exp_streak,
            )

            self.progressed.emit("Initializing processor ...")
            with SeriesProcessor(
                str(input_path),
                series_config,
                mask_array,
                self.use_fabio,
            ) as processor:
                self.progressed.emit("Processing image series ...")
                series_result = processor.process_series()

                self.progressed.emit("Saving results ...")
                series_result.save(str(output_path), self.output_prefix, avg_clean_only=self.avg_clean_only)

            if self.plot_after:
                self._plot_saved_tifs(output_path, self.output_prefix)

            self.finished_ok.emit(f"Done. Results saved to: {output_path} (prefix: {self.output_prefix})")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

    def _plot_saved_tifs(self, output_path: Path, prefix: str) -> None:
        """Plot all saved tif files with given prefix using 10/90 percentiles and save as PNG.

        This mirrors scripts/plot.py behavior but limits to the given prefix.
        """
        try:
            from matplotlib import pyplot as plt  # type: ignore
            import matplotlib  # type: ignore
            import tifffile  # type: ignore

            # Use non-GUI backend to avoid conflicts with Qt event loop
            matplotlib.use("Agg", force=True)

            tif_files = sorted(output_path.glob(f"{prefix}_*.tif"))
            if not tif_files:
                self.progressed.emit(f"No .tif files found in {output_path} with prefix '{prefix}_'")
                return

            self.progressed.emit(f"Plotting {len(tif_files)} files ...")
            for tif_file in tif_files:
                img = tifffile.imread(tif_file).astype(np.float64)
                vmin = float(np.percentile(img, 1))
                vmax = float(np.percentile(img, 99))
                plt.figure()
                plt.imshow(img, cmap="inferno", vmin=vmin, vmax=vmax)
                plt.title(tif_file.name)
                plt.colorbar()
                png_name = tif_file.with_suffix('.png').name
                png_path = output_path / png_name
                plt.savefig(png_path, bbox_inches='tight')
                plt.close()
                self.progressed.emit(f"Saved plot: {png_path}")
        except Exception as exc:  # noqa: BLE001
            # Do not fail the whole run on plotting errors; report and continue
            self.progressed.emit(f"Plotting failed: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SAXS Decosmic GUI")
        self.resize(400, 650)

        central = QWidget(self)
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        # IO group
        io_group = QGroupBox("I/O Paths")
        io_form = QFormLayout()

        self.input_file_edit = QLineEdit()
        btn_browse_input = QPushButton("Browse...")
        btn_browse_input.clicked.connect(self._browse_input_file)
        io_form.addRow(QLabel("INPUT_FILE"), self._hbox(self.input_file_edit, btn_browse_input))

        self.output_dir_edit = QLineEdit()
        btn_browse_output = QPushButton("Browse...")
        btn_browse_output.clicked.connect(self._browse_output_dir)
        io_form.addRow(QLabel("OUTPUT_DIR"), self._hbox(self.output_dir_edit, btn_browse_output))

        self.output_prefix_edit = QLineEdit()
        io_form.addRow(QLabel("OUTPUT_PREFIX"), self.output_prefix_edit)

        self.user_mask_edit = QLineEdit()
        btn_browse_mask = QPushButton("Browse...")
        btn_browse_mask.clicked.connect(self._browse_user_mask)
        io_form.addRow(QLabel("USER_MASK"), self._hbox(self.user_mask_edit, btn_browse_mask))

        io_group.setLayout(io_form)
        layout.addWidget(io_group)

        # Options group
        opts_group = QGroupBox("Options")
        opts_form = QFormLayout()

        self.use_fabio_cb = QCheckBox("USE_FABIO")
        self.use_fabio_cb.setChecked(False)
        opts_form.addRow(self.use_fabio_cb)

        self.avg_clean_only_cb = QCheckBox("AVG_CLEAN_ONLY")
        self.avg_clean_only_cb.setChecked(True)
        opts_form.addRow(self.avg_clean_only_cb)

        self.plot_after_cb = QCheckBox("PLOT_AFTER")
        self.plot_after_cb.setChecked(True)
        opts_form.addRow(self.plot_after_cb)

        # Parameters
        params_group = QGroupBox("Parameters")
        params_form = QFormLayout()

        self.th_donut_spin = QSpinBox()
        self.th_donut_spin.setValue(15)
        params_form.addRow("TH_DONUT", self.th_donut_spin)

        self.th_mask_spin = QDoubleSpinBox()
        self.th_mask_spin.setDecimals(2)
        self.th_mask_spin.setSingleStep(0.01)
        self.th_mask_spin.setValue(0.05)
        params_form.addRow("TH_MASK", self.th_mask_spin)

        self.th_streak_spin = QSpinBox()
        self.th_streak_spin.setValue(3)
        params_form.addRow("TH_STREAK", self.th_streak_spin)

        self.win_streak_spin = QSpinBox()
        self.win_streak_spin.setValue(3)
        params_form.addRow("WIN_STREAK", self.win_streak_spin)

        self.exp_donut_spin = QSpinBox()
        self.exp_donut_spin.setValue(9)
        params_form.addRow("EXP_DONUT", self.exp_donut_spin)

        self.exp_streak_spin = QSpinBox()
        self.exp_streak_spin.setValue(3)
        params_form.addRow("EXP_STREAK", self.exp_streak_spin)

        opts_group.setLayout(opts_form)
        layout.addWidget(opts_group)

        params_group.setLayout(params_form)
        layout.addWidget(params_group)

        # Run controls
        run_box = QHBoxLayout()
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._on_run)
        run_box.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        run_box.addWidget(self.cancel_btn)

        layout.addLayout(run_box)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # Log output
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text, stretch=1)

        # (Status bar removed)

        self.worker: ProcessingWorker | None = None

    @staticmethod
    def _hbox(*widgets: QWidget) -> QWidget:
        w = QWidget()
        l = QHBoxLayout(w)
        l.setContentsMargins(0, 0, 0, 0)
        for wd in widgets:
            l.addWidget(wd)
        return w

    def _browse_input_file(self) -> None:
        fname, _ = QFileDialog.getOpenFileName(self, "Select first image", "", "Images (*.tif *.tiff *.edf *.cbf *.img *.mccd *.h5 *.*)")
        if fname:
            self.input_file_edit.setText(fname)

    def _browse_output_dir(self) -> None:
        dname = QFileDialog.getExistingDirectory(self, "Select output directory")
        if dname:
            self.output_dir_edit.setText(dname)

    def _browse_user_mask(self) -> None:
        fname, _ = QFileDialog.getOpenFileName(self, "Select user mask (optional)", "", "Images (*.tif *.tiff *.edf *.cbf *.img *.mccd *.h5 *.*)")
        if fname:
            self.user_mask_edit.setText(fname)

    def _append_log(self, text: str) -> None:
        self.log_text.append(text)
        # status bar removed; logs only to the text widget

    def _on_run(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Processing is already running.")
            return

        input_file = self.input_file_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        output_prefix = self.output_prefix_edit.text().strip()
        user_mask = self.user_mask_edit.text().strip() or None
        use_fabio = self.use_fabio_cb.isChecked()
        avg_clean_only = self.avg_clean_only_cb.isChecked()
        plot_after = self.plot_after_cb.isChecked()

        if not input_file:
            QMessageBox.critical(self, "Error", "Please select INPUT_FILE.")
            return
        if not output_dir:
            QMessageBox.critical(self, "Error", "Please select OUTPUT_DIR.")
            return
        if not output_prefix:
            # Allow empty but warn
            result = QMessageBox.question(self, "Confirm", "OUTPUT_PREFIX is empty. Continue?", QMessageBox.Yes | QMessageBox.No)
            if result != QMessageBox.Yes:
                return

        # Parameters
        th_donut = int(self.th_donut_spin.value())
        th_mask = float(self.th_mask_spin.value())
        th_streak = int(self.th_streak_spin.value())
        win_streak = int(self.win_streak_spin.value())
        exp_donut = int(self.exp_donut_spin.value())
        exp_streak = int(self.exp_streak_spin.value())

        self.worker = ProcessingWorker(
            input_file,
            output_dir,
            output_prefix,
            user_mask,
            use_fabio,
            avg_clean_only,
            th_donut,
            th_mask,
            th_streak,
            win_streak,
            exp_donut,
            exp_streak,
            plot_after,
        )
        self.worker.progressed.connect(self._append_log)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished_ok.connect(self._on_finished_ok)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self._append_log("Started processing ...")
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        # show indeterminate progress
        self.progress_bar.setVisible(True)

    def _on_failed(self, msg: str) -> None:
        self._append_log(f"Failed: {msg}")
        QMessageBox.critical(self, "Processing Failed", msg)

    def _on_finished_ok(self, msg: str) -> None:
        self._append_log(msg)
        QMessageBox.information(self, "Done", msg)

    def _on_finished(self) -> None:
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        # hide progress
        self.progress_bar.setVisible(False)

    def _on_cancel(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            # QThread has no safe cooperative cancel here; best effort is to ask politely.
            # Since the underlying processing loop is compute bound, we avoid hard termination.
            QMessageBox.information(self, "Not supported", "Cancel is not supported during processing.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    mp.freeze_support()
    mp.set_start_method('spawn')
    app = QApplication([])
    win = MainWindow()
    win.show()
    exit_code = app.exec()

    # Close any devnull file handles we opened to avoid leaving resources open
    try:
        if '_devnull_stdout' in globals() and globals()['_devnull_stdout'] is not None:
            globals()['_devnull_stdout'].close()
        if '_devnull_stderr' in globals() and globals()['_devnull_stderr'] is not None:
            globals()['_devnull_stderr'].close()
    except Exception:
        pass

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()


