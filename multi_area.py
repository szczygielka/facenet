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
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider
from PyQt5.QtGui import * 
import copy 


class MultiArea(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.scroll_area = QScrollArea(self)
        self.layout.addWidget(self.scroll_area)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget(self.scroll_area)
        self.scroll_layout = QVBoxLayout(self.scroll_content) 
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)

    def clear(self):
        for i in reversed(range(self.scroll_layout.count())): 
            self.scroll_layout.itemAt(i).widget().deleteLater()

    def add_widget(self, widget):
        self.scroll_layout.addWidget(widget)




