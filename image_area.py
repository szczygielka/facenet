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
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import time
from inception_res_net import InceptionResNetV1
import os
from scipy.spatial import distance
from constants import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import * 
import copy 
from image_manager import *
import os

class ImageArea(QtWidgets.QWidget):
    def __init__(self, facenet_manager, max_image_size = DEFAULT_MAX_IMAGE_SIZE,  parent=None):
        super().__init__(parent)

        self.original_image = None
        self.layout = QVBoxLayout()
        self.max_image_size = max_image_size
        self.image = QLabel()
        self.layout.addWidget(self.image)
        self.setLayout(self.layout)
        self.facenet_manager = facenet_manager
        self.file_name = None
        self.out_file_name = None
    
    def reload_image(self):
        if self.original_image is not None:
            self.update_image(copy.copy(self.original_image), False)
    
    def update_image(self, cv_image, file_name = None, out_file_name = None, set_ori = True):
        self.cv_image = cv_image
        if file_name:
            self.file_name = file_name
        if file_name:
            self.out_file_name = out_file_name
        if set_ori:
            self.original_image = copy.copy(cv_image)

        self.cv_image, _ = self.facenet_manager.process_image(self.cv_image, self.out_file_name, self.file_name)
        height, width, channels = self.cv_image.shape
        
        # resize image to square size x size
        if height > width:  
            coef = self.max_image_size/height
            self.cv_image = cv2.resize(self.cv_image, (int(coef * width), self.max_image_size), interpolation=cv2.INTER_LINEAR)
        else:
            coef = self.max_image_size/width
            self.cv_image = cv2.resize(self.cv_image, (self.max_image_size, int(coef * height)), interpolation=cv2.INTER_LINEAR)

        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        qformat = QImage.Format_Indexed8
        if len(self.cv_image.shape) == 3:
            if(self.cv_image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        self.cv_image = QImage(self.cv_image, self.cv_image.shape[1], self.cv_image.shape[0], self.cv_image.strides[0], qformat)

        # self.cv_image = QtGui.QImage(self.cv_image.data, self.cv_image.shape[1], self.cv_image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()

        self.image.setPixmap(QtGui.QPixmap.fromImage(self.cv_image))

    
    def open_from_file(self, file_name):
        if file_name:
            cv_image = ImageManager.load_image(file_name)
            self.update_image(cv_image, file_name, os.path.basename(file_name))

    def open(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Wybierz plik","", "Obrazy(*.png *.jpg *.gif *.jpeg)")
        self.open_from_file(file_name)