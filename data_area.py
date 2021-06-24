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


class DataArea(QtWidgets.QWidget):
    def __init__(self, on_dataset_load_button_clicked_function, 
                        on_threshold_change_function, 
                        on_log_file_change_function, 
                        on_multi_path_change_function,
                        on_multi_reset_path_button_clicked,
                        on_output_path_change_function,
                        threshold = 1, parent = None):
        super().__init__(parent)

        self.on_dataset_load_button_clicked_function = on_dataset_load_button_clicked_function
        self.on_threshold_change_function = on_threshold_change_function
        self.on_log_file_change_function = on_log_file_change_function
        self.on_output_path_change_function = on_output_path_change_function
        self.on_multi_path_change_function = on_multi_path_change_function
        self.on_multi_reset_path_button_clicked = on_multi_reset_path_button_clicked
        self.threshold_min = 0
        self.threshold_max = 3
        self.threshold_scale = 100
        self.threshold = threshold
        self.dataset_path = ""
        self.log_filename = ""
        self.multi_path = ""
        self.outpath = None
        
        self.initialize_threshold_gui()
        self.initialize_dataset_gui()
        self.initialize_out_path_gui()
        self.initialize_multi_gui()
        self.initialize_log_gui()
        self.initialize_layout()
        self.updateThresholdLabel()
        self.setMaximumHeight(300)

    def initialize_layout(self):
        self.layout = QGridLayout()
        self.layout.addWidget(self.threshold_label, 0, 0, 1, 2, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.threshold_slider,1, 0, 1, 2)
        self.layout.addWidget(self.dataset_set_path_button,2, 0, 1, 1, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.dataset_load_button,2, 1, 1, 1, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.dataset_path_label,3, 0, 1, 2, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.dataset_loaded_images_label,4, 0, 1, 2, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.log_set_path_button,5, 0, 1, 2, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.log_path_label,6, 0, 1, 2, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.multi_set_path_button,7, 0, 1, 1, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.multi_reset_path_button,7, 1, 1, 2, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.outpath_set_button,8, 0, 1, 2, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.outpath_label,9, 0, 1, 2, alignment=Qt.AlignCenter)
        self.setLayout(self.layout)

    def initialize_dataset_gui(self):
        self.dataset_path_label = QLabel("Path: {}".format(self.dataset_path))
        self.dataset_set_path_button = QPushButton("Set dataset path")
        self.dataset_set_path_button.clicked.connect(self.dataset_set_path_button_clicked)
        self.dataset_loaded_images_label = QLabel("Loaded images: 0")
        self.dataset_load_button = QPushButton("Load dataset")
        self.dataset_load_button.clicked.connect(self.dataset_load_button_clicked)

    def initialize_log_gui(self):
        self.log_path_label = QLabel("Log: {}".format(self.log_filename))
        self.log_set_path_button = QPushButton("Set log file")
        self.log_set_path_button.clicked.connect(self.log_set_path_button_clicked)

    def initialize_out_path_gui(self):
        self.outpath_label = QLabel("Output: {}".format(self.outpath))
        self.outpath_set_button = QPushButton("Select output path")
        self.outpath_set_button.clicked.connect(self.set_outpath_button_clicked)

    def initialize_multi_gui(self):
        self.multi_set_path_button = QPushButton("Select directory to test")
        self.multi_reset_path_button = QPushButton("Reset multi")
        self.multi_set_path_button.clicked.connect(self.multi_set_path_button_clicked)
        self.multi_reset_path_button.clicked.connect(self.on_multi_reset_path_button_clicked)

    def set_multi_path(self, multi_path):
        if multi_path:
            self.multi_path = multi_path
            self.on_multi_path_change_function(multi_path)
    
    def set_outpath(self, outpath):
        if outpath:
            self.outpath = outpath
            self.outpath_label.setText("Output: {}".format(self.outpath))
            self.on_output_path_change_function(self.outpath)

    def multi_set_path_button_clicked(self):
        path_name = QFileDialog.getExistingDirectory(self, 'Select directory to test')
        self.set_multi_path(path_name)

    def set_outpath_button_clicked(self):
        path_name = QFileDialog.getExistingDirectory(self, 'Select output path')
        self.set_outpath(path_name)

    def set_log_path(self, log_filename):
        if log_filename:
            self.log_filename = log_filename
            self.log_path_label.setText("Log: {}".format(self.log_filename))
            self.on_log_file_change_function(log_filename)


    def log_set_path_button_clicked(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self,"Select log file","","Text Files (*.txt, *.log)", options=options)
        self.set_log_path(file_name)


    def initialize_threshold_gui(self):
        self.threshold_label = QLabel("Threshold")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(int(self.threshold_min * self.threshold_scale))
        self.threshold_slider.setMaximum(int(self.threshold_max * self.threshold_scale))
        self.threshold_slider.setValue(self.threshold)
        self.threshold_slider.setTickInterval(1)
        self.threshold_slider.valueChanged.connect(self.on_threshold_slider_value_change)

    def set_dataset_path(self, path_name):
        if path_name:
            self.dataset_path = path_name
            self.dataset_path_label.setText("Path: {}".format(self.dataset_path))

    def dataset_set_path_button_clicked(self):
        path_name = QFileDialog.getExistingDirectory(self, 'Select a directory')
        self.set_dataset_path(path_name)

    def update_loaded_images(self, value):
        self.dataset_loaded_images_label.setText("Loaded images: {}".format(value))

    def dataset_load_button_clicked(self):
        if self.on_dataset_load_button_clicked_function:
            self.on_dataset_load_button_clicked_function(self.dataset_path)

    def set_threshold(self, val):
        self.threshold = val
        self.updateThresholdLabel()
        if self.on_threshold_change_function:
            self.on_threshold_change_function(self.threshold)

    def on_threshold_slider_value_change(self):
        threshold = self.threshold_slider.value() * 1 / self.threshold_scale
        self.set_threshold(threshold)

    def updateThresholdLabel(self):
        self.threshold_label.setText("Threshold: {:.2f}".format(self.threshold))