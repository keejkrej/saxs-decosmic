import numpy as np
from scipy.ndimage import convolve, maximum_filter

class ImageModel:

    def __init__(self, img: np.ndarray[np.int32], mask: np.ndarray[bool]):
        self.img = img
        self.mask = mask
        self.imgClean = None
        self.modMask = None
        self.subDonut = None
        self.subStrike = None

    def loadParams(self, thDonut: int, thStrike: int, winStrike: int, expDonut: int, expStrike: int) -> None:
        self.thDonut = thDonut
        self.thStrike = thStrike
        self.winStrike = winStrike
        self.expDonut = expDonut
        self.expStrike = expStrike

    def deDonut(self, img: np.ndarray[np.int32], mask: np.ndarray[bool]) -> tuple[np.ndarray[np.int32], np.ndarray[bool]]:
        imgCopy = img.copy()
        donutMask = imgCopy > self.thDonut
        donutMaskExpanded = maximum_filter(donutMask, size=self.expDonut)
        modMask = donutMaskExpanded & mask
        imgCopy[modMask] = 0
        return imgCopy, modMask
    
    def deStrike(self, img: np.ndarray[np.int32], mask: np.ndarray[bool]) -> tuple[np.ndarray[np.int32], np.ndarray[bool]]:
        imgCopy = np.copy(img)
        imgBinary = (imgCopy > 0).astype(np.int32)
        imgBinary = imgBinary * mask
        convKernel = np.ones((self.winStrike, self.winStrike), dtype=np.int32)
        imgConv = convolve(imgBinary, convKernel, mode='constant', cval=0)
        imgConv = imgConv * imgBinary
        strikeMask = imgConv > self.thStrike
        strikeMaskExpanded = maximum_filter(strikeMask, size=self.expStrike)
        modMask = strikeMaskExpanded & mask
        imgCopy[modMask] = 0
        return imgCopy, modMask
    
    def cleanImg(self) -> None:
        imgCopy = np.copy(self.img)
        self.imgDonut, self.maskDonut = self.deDonut(imgCopy, self.mask)
        self.imgStrike, self.maskStrike = self.deStrike(self.imgDonut, self.mask)
        self.imgClean = np.copy(self.imgStrike)
        self.modMask = self.maskDonut | self.maskStrike
        self.subDonut = imgCopy - self.imgDonut
        self.subStrike = self.imgDonut - self.imgStrike
