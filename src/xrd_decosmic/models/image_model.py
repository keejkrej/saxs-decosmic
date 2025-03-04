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

    def loadParams(self, thDonut: int, thStrike: int, winStrike: int) -> None:
        self.thDonut = thDonut
        self.thStrike = thStrike
        self.winStrike = winStrike        

    def deDonut(self, img: np.ndarray[np.int32], mask: np.ndarray[bool]) -> tuple[np.ndarray[np.int32], np.ndarray[bool]]:
        imgCopy = img.copy()
        donutMask = imgCopy > self.thDonut
        donutMaskExpanded = maximum_filter(donutMask, size=3)
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
        strikeMaskExpanded = maximum_filter(strikeMask, size=3)
        modMask = strikeMaskExpanded & mask
        imgCopy[modMask] = 0
        return imgCopy, modMask
    
    def cleanImg(self) -> None:
        imgCopy = np.copy(self.img)
        imgDonut, maskDonut = self.deDonut(imgCopy, self.mask)
        imgStrike, maskStrike = self.deStrike(imgDonut, self.mask)
        self.imgClean = np.copy(imgStrike)
        self.modMask = maskDonut | maskStrike
        self.subDonut = imgCopy - imgDonut
        self.subStrike = imgDonut - imgStrike