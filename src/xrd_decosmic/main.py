import os
import sys
import json
import numpy as np
import fabio
from matplotlib.pyplot import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, 
    NavigationToolbar2QT as NavigationToolbar
)
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QSizePolicy, QVBoxLayout, QWidget
from PyQt5.QtCore import pyqtSlot
from .ui.main_window_ui import Ui_MainWindow
from .models.data_model import DataModel
from .models.params_model import ParamsModel

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(8, 8))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

    def plot(self, img: np.ndarray):
        self.ax.clear()
        self.ax.imshow(img, cmap='hot')
        self.draw()

class MainWindow(QMainWindow, Ui_MainWindow):
    dataModel: DataModel

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setupExtra()
        self.setupPlot()

        self.dataModel = None
        self.paramsModel = ParamsModel('15', '0.1', '3', '3')
        self.paramsEdit = {
            'thDonut': self.thDonutText,
            'thMask': self.thMaskText,
            'thStrike': self.thStrikeText,
            'winStrike': self.winStrikeText
        }

        self.setupController()
        self.updateParams()

    def setupExtra(self):
        self.setWindowTitle("XRD Decosmic")
        # self.setFixedSize(1000, 600)
        self.logText.setReadOnly(True)
        self.comboBox.addItems(['Avg', 'CleanAvg', 'Diff', 'SubDonut', 'SubStrike', 'Mask'])

    def setupPlot(self):
        plotLayout = QVBoxLayout()
        self.plotCanvas = PlotCanvas()
        self.plotToolBar = NavigationToolbar(self.plotCanvas, self.plotArea)
        plotLayout.addWidget(self.plotToolBar)
        plotLayout.addWidget(self.plotCanvas)
        self.plotArea.setLayout(plotLayout)

    def setupController(self):
        self.actionLoadData.triggered.connect(self.loadData)
        self.actionLoadParameters.triggered.connect(self.loadParams)
        self.actionSaveData.triggered.connect(self.saveData)
        self.actionSaveParameters.triggered.connect(self.saveParams)
        for name, edit in self.paramsEdit.items():
            edit.textChanged.connect(lambda text, name=name: self.onParamsEditChanged(name, text))
            getattr(self.paramsModel, f"{name}Changed").connect(lambda text, name=name: self.onParamsDataChanged(name, text))
        self.startButton.clicked.connect(self.startCleaning)
        self.plotButton.clicked.connect(self.plotData)

    def logPrintLn(self, text):
        self.logText.appendPlainText(text)
        QApplication.processEvents()

    def loadData(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Open first file in image series', './', '*.tif')
        self.logPrintLn(f"Loading {os.path.basename(fileName)} ...")
        try:
            self.dataModel = DataModel(fileName)
            self.dataModel.actionSignal.connect(self.logPrintLn)
            self.dataModel.progressSignal.connect(self.logPrintLn)
            self.logPrintLn(f"Loading successful!")
        except Exception as e:
            self.logPrintLn(f"Loading failed!")
            print(e)

    def loadParams(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Open parameter file', './', '*.json')
        self.logPrintLn(f"Loading {os.path.basename(fileName)} ...")
        try:
            self.paramsJson = json.load(open(fileName, 'r'))
            for name, text in self.paramsJson.items():
                setattr(self.paramsModel, name, text)
            self.logPrintLn(f"Loading successful!")
        except Exception as e:
            self.logPrintLn(f"Loading failed!")
            print(e)

    def saveParams(self):
        fileName, _ = QFileDialog.getSaveFileName(self, 'Save parameter file', './', '*.json')
        self.logPrintLn(f"Saving {os.path.basename(fileName)} ...")
        try:
            self.paramsJson = {name: getattr(self.paramsModel, name) for name in self.paramsEdit.keys()}
            with open(fileName, 'w') as f:
                json.dump(self.paramsJson, f, indent=4)
            self.logPrintLn(f"Saving successful!")
        except Exception as e:
            self.logPrintLn(f"Saving failed!")
            print(e)

    def startCleaning(self):
        self.logPrintLn("Start cleaning ...")
        try:
            self.dataModel.loadParams(self.paramsModel)
            self.dataModel.avgCleanImg()
            self.logPrintLn("Cleaning successful!")
        except Exception as e:
            self.logPrintLn("Cleaning failed!")
            print(e)

    def plotData(self):
        plotType = self.comboBox.currentText()
        self.logPrintLn(f"Plotting {plotType} ...")
        try:
            if plotType == 'Avg':
                img = self.dataModel.imgAvg
            elif plotType == 'CleanAvg':
                img = self.dataModel.imgCleanAvg
            elif plotType == 'Diff':
                img = self.dataModel.imgDiffAvg
            elif plotType == 'SubDonut':
                img = self.dataModel.subDonutAvg
            elif plotType == 'SubStrike':
                img = self.dataModel.subStrikeAvg
            elif plotType == 'Mask':
                img = self.dataModel.combinedMask
            self.plotCanvas.plot(img)
            self.logPrintLn("Plotting successful!")
        except Exception as e:
            self.logPrintLn("Plotting failed!")
            print(e)

    def saveData(self):
        folderName = QFileDialog.getExistingDirectory(self, 'Save data and plot to the folder', './')
        self.logPrintLn(f"Saving data and plot to {os.path.basename(folderName)} ...")
        try:
            self.saveTif(self.dataModel.imgAvg, os.path.join(folderName, 'avg.tif'))
            self.saveTif(self.dataModel.imgCleanAvg, os.path.join(folderName, 'clean_avg.tif'))
            self.saveTif(self.dataModel.imgDiffAvg, os.path.join(folderName, 'diff_avg.tif'))
            self.logPrintLn(f"Saving successful!")
        except Exception as e:
            self.logPrintLn(f"Saving failed!")
            print(e)

    def saveTif(self, img, fileName):
        fabio.tifimage.TifImage(data=img).write(fileName)

    def updateParams(self):
        for name, edit in self.paramsEdit.items():
            edit.setText(getattr(self.paramsModel, name))

    @pyqtSlot(str, str)
    def onParamsEditChanged(self, name, text):
        setattr(self.paramsModel, name, text)

    @pyqtSlot(str, str)
    def onParamsDataChanged(self, name, text):
        if self.paramsEdit[name].text() != text:
            self.paramsEdit[name].setText(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())