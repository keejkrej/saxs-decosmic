import fabio
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from .params_model import ParamsModel
from .image_model import ImageModel
from .progress_model import ProgressModel

class DataModel(QObject):
    firstFilename: str
    imgSeries: any
    imgNum: int
    imgShape: tuple[int, int]
    imgDType: np.dtype
    thDonut: int
    thMask: float
    thStrike: int
    winStrike: int

    actionSignal = pyqtSignal(str)
    progressSignal = pyqtSignal(str)

    def __init__(self, firstFilename: str, userMask: np.ndarray[bool]=True):
        super().__init__()
        self.firstFilename = firstFilename
        self.imgSeries = fabio.open_series(first_filename=self.firstFilename)
        self.imgNum = self.imgSeries.nframes
        self.imgShape = self.imgSeries.get_frame(0).data.shape
        self.imgDType = self.imgSeries.get_frame(0).data.dtype
        self.userMask = userMask

    def loadParams(self, paramsModel: ParamsModel):
        thDonut = int(paramsModel.thDonut)
        thMask = float(paramsModel.thMask)
        thStrike = int(paramsModel.thStrike)
        winStrike = int(paramsModel.winStrike)
        if thDonut < 0:
            raise ValueError('Threshold of donut should be a non-negative integer')
        if thMask < 0 or thMask > 1:
            raise ValueError('Threshold of mask should be a float between 0 and 1')
        if thStrike < 0 and thStrike > winStrike**2:
            raise ValueError('Threshold of strike should be a non-negative integer and less than window size')
        if winStrike < 0:
            raise ValueError('Window size of strike should be a non-negative integer')
        self.thDonut = thDonut
        self.thMask = thMask
        self.thStrike = thStrike
        self.winStrike = winStrike

    def getImg(self, idx: int) -> np.ndarray:
        img = self.imgSeries.get_frame(idx).data.astype(np.int32)
        img = np.nan_to_num(img, nan=-1)
        img = np.clip(img, 0, None)
        return img
    
    def avgImg(self) -> np.ndarray:
        imgSum = np.zeros(self.imgShape, dtype=np.float64)
        imgBinarySum = np.zeros(self.imgShape, dtype=np.float64)
        self.actionSignal.emit('Averaging images')
        avgProgress = ProgressModel(self.imgNum, 10)
        avgProgress.progressSignal.connect(self.progressSignal.emit)

        self.progressSignal.emit('\n')
        for i in range(self.imgNum):
            img = self.getImg(i)
            imgSum += img
            imgBinary = img > 0
            imgBinarySum += imgBinary
            avgProgress.update(i)
        self.progressSignal.emit('100')
        
        self.imgAvg = imgSum / self.imgNum
        self.imgBinaryAvg = imgBinarySum / self.imgNum

    def maskImg(self) -> np.ndarray:
        self.ringMask = self.imgBinaryAvg < self.thMask
        self.combinedMask = self.ringMask & self.userMask

    def avgCleanImg(self) -> np.ndarray:
        self.avgImg()
        self.maskImg()
        imgCleanSum = np.zeros(self.imgShape, dtype=np.float64)
        imgCleanNum = np.ones(self.imgShape, dtype=np.int32) * self.imgNum
        subDonutSum = np.zeros(self.imgShape, dtype=np.float64)
        subStrikeSum = np.zeros(self.imgShape, dtype=np.float64)
        self.actionSignal.emit('Cleaning images')
        cleanProgress = ProgressModel(self.imgNum, 10)
        cleanProgress.progressSignal.connect(self.progressSignal.emit)

        self.progressSignal.emit('\n')
        for i in range(self.imgNum):
            img = self.getImg(i)
            imgModel = ImageModel(img, self.combinedMask)
            imgModel.loadParams(self.thDonut, self.thStrike, self.winStrike)
            imgModel.cleanImg()
            imgCleanSum += imgModel.imgClean
            subDonutSum += imgModel.subDonut
            subStrikeSum += imgModel.subStrike
            imgCleanNum -= imgModel.modMask
            cleanProgress.update(i)
        self.progressSignal.emit('100')
        
        self.imgCleanAvg = imgCleanSum / imgCleanNum
        self.subDonutAvg = subDonutSum / self.imgNum
        self.subStrikeAvg = subStrikeSum / self.imgNum
        self.imgDiffAvg = self.imgAvg - self.imgCleanAvg
