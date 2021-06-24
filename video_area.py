from PyQt5.QtWidgets import QApplication, QTabWidget, QWidget, QFormLayout, QLineEdit, QPlainTextEdit, QGridLayout, QPushButton
from PyQt5.QtWidgets import QLabel, QFileDialog, QScrollArea, QSpinBox
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QStatusBar
from PyQt5.QtWidgets import QToolBar
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDir, Qt
from PyQt5 import QtGui, QtCore, QtWidgets
import cv2
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import time
from inception_res_net import InceptionResNetV1
import os
from scipy.spatial import distance

from video_thread import VideoThread
from image_area import ImageArea
from constants import *
from pathlib import Path


class VideoArea(ImageArea):
    def __init__(self, face_detector, parent=None):
        super().__init__(face_detector, parent= parent)

    def open_from_file(self, file_name):
        if file_name:
            self.thread = VideoThread()
            self.thread.file_name = file_name
            self.thread.on_frame_read = lambda cv_image, frame: self.update_image(cv_image,  self.thread.file_name,  "{0}/{0}_{1}.png".format(Path(self.thread.file_name).stem,frame), False,)
            self.thread.start()

    def open(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Wybierz plik","", "Filmy(*.mp4)")
        self.open_from_file(file_name)
    
    def on_change_treshold(self):
         pass