from PyQt5.QtCore import QObject, pyqtSignal

class ParamsModel(QObject):
    thDonut: str
    thMask: str
    thStrike: str
    winStrike: str

    thDonutChanged = pyqtSignal(str)
    thMaskChanged = pyqtSignal(str)
    thStrikeChanged = pyqtSignal(str)
    winStrikeChanged = pyqtSignal(str)

    def __init__(self, thDonut, thMask, thStrike, winStrike):
        super().__init__()
        self._thDonut = thDonut
        self._thMask = thMask
        self._thStrike = thStrike
        self._winStrike = winStrike

    @property
    def thDonut(self):
        return self._thDonut
    
    @thDonut.setter
    def thDonut(self, text):
        if self._thDonut != text:
            self._thDonut = text
            self.thDonutChanged.emit(text)

    @property
    def thMask(self):
        return self._thMask
    
    @thMask.setter
    def thMask(self, text):
        if self._thMask != text:
            self._thMask = text
            self.thMaskChanged.emit(text)

    @property
    def thStrike(self):
        return self._thStrike
    
    @thStrike.setter
    def thStrike(self, text):
        if self._thStrike != text:
            self._thStrike = text
            self.thStrikeChanged.emit(text)

    @property
    def winStrike(self):
        return self._winStrike
    
    @winStrike.setter
    def winStrike(self, text):
        if self._winStrike != text:
            self._winStrike = text
            self.winStrikeChanged.emit(text)
    