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
from data_area import *
from image_area import *
from video_area import *
from constants import *
from facenet_manager import *  
from multi_area import * 
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import time

class ProcessThread(QThread):
    facenet_manager = None
    callback = None
    parent = None
    def run(self):
        def add(data):
            print("jest tutaj")
            image, detections = data
            for d in detections:
                filename, output_path, outname, closest_distance, face_name = d
                self.callback(FaceImageWidget(image, filename, output_path, outname, closest_distance, face_name))
            time.sleep(0.001)

        data = self.facenet_manager.process(self.path, add)

class FaceImageWidget(QtWidgets.QWidget):
    def __init__(self, image, filename, output_path, outname, closest_distance, face_name, small_image_size = 100,parent = None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.image = QLabel()
        self.layout.addWidget(self.image, alignment=Qt.AlignLeft)
        self.label = QLabel()
        self.layout.addWidget(self.label, alignment=Qt.AlignLeft)

        self.label.setText("{}; {:.2f} ; {}\n".format(filename, closest_distance, face_name))

        self.cv_image = cv2.resize(image, (small_image_size, small_image_size))
        self.cv_image = QtGui.QImage(self.cv_image.data, self.cv_image.shape[1], self.cv_image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image.setPixmap(QtGui.QPixmap.fromImage(self.cv_image))


    def clear(self):
        for i in reversed(range(self.scroll_layout.count())): 
            self.scroll_layout.itemAt(i).widget().deleteLater()

    def addImage(self, widget):
        self.scroll_layout.addWidget(widget)

class FacenetGui(QMainWindow):
    def __init__(self, dataset_path = None, threshold = 1, log_filename = None, output_path = None, test_path = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Facenet')
        
        self.facenet_manager = FacenetManager()
        
        self.centralWidget = QWidget()
        self.data_widget = DataArea(self.on_dataset_load_button_clicked_function,
            self.on_threshold_change_function, self.on_log_file_change_function, self.on_multi_path_change_function,
            self.on_multi_reset_path_button_clicked, 
            self.on_output_path_change_function, threshold)

        self.main_layout = QHBoxLayout()
        self.centralWidget.setLayout(self.main_layout)
        self.setCentralWidget(self.centralWidget)

        self.createMenu()
        self.createTabs()
        self.createImageMenu()
        self.data_widget.set_dataset_path(dataset_path)
        self.data_widget.set_log_path(log_filename)
        self.data_widget.set_outpath(output_path)
        self.data_widget.set_threshold(threshold)

    def on_output_path_change_function(self, val):
        self.facenet_manager.set_output_path(val)

    def on_dataset_load_button_clicked_function(self, val):
        self.facenet_manager.load_dataset_to_face_detector_manager(val)
        self.data_widget.update_loaded_images(len(self.facenet_manager.face_detector.calculates_images))

    def on_threshold_change_function(self, val):
        self.facenet_manager.set_threshold(val)
        self.image_tab.reload_image()

    def on_log_file_change_function(self, val):
        self.facenet_manager.set_log_filename(val)

    def on_multi_reset_path_button_clicked(self):
        self.multi_tab.clear()     

    def on_multi_path_change_function(self, path):
        data = self.facenet_manager.process(path)
        for image, detections in data:
            for d in detections:
                print(d)
                filename, output_path, outname, closest_distance, face_name = d
                self.multi_tab.add_widget(FaceImageWidget(image, filename, output_path, outname, closest_distance, face_name))

    def createImageMenu(self):
        self.imageMenu = self.menu.addMenu("Image")
        self.imageMenu.addAction('Open', self.image_tab.open, shortcut="Ctrl+G")
        self.videoMenu = self.menu.addMenu("Video")
        self.videoMenu.addAction('Open', self.video_tab.open, shortcut="Ctrl+H")

    def createMenu(self):
        self.menu = self.menuBar()
        self.fileMenu = self.menu.addMenu("File")
        self.fileMenu.addAction('Exit', self.close)

    
    def createTabs(self):
        self.tabs = QTabWidget()
        
        self.image_tab = ImageArea(self.facenet_manager)
        self.video_tab = VideoArea(self.facenet_manager)
        self.multi_tab = MultiArea()
        
        self.tabs.addTab(self.image_tab, "Image")        
        self.tabs.addTab(self.video_tab, "Video")        
        self.tabs.addTab(self.multi_tab, "Multi")        
        
        self.main_layout.addWidget(self.tabs)
        self.main_layout.addWidget(self.data_widget, alignment=Qt.AlignTop)

    def set_tab(self, tab_index = 0):
        self.tabs.setCurrentIndex(tab_index)

