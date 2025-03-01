from PyQt5.QtCore import QObject, pyqtSignal

class ProgressModel(QObject):
    progressSignal = pyqtSignal(str)

    def __init__(self, max: int, step: int):
        super().__init__()
        self.max = max
        self.step = step
        self.accumulator = 0

    def update(self, idx):
        pct = idx / self.max * 100
        if pct > self.accumulator:
            self.progressSignal.emit(f"{self.accumulator:02d}%")
            self.accumulator += self.step
