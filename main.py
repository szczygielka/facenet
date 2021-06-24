import copy
from os.path import isfile, join
from os import listdir
import os
import sys
from constants import * 
import argparse

from face_detector import * 
from cascade_manager import *
from pathlib import Path
from facenet_manager import FacenetManager
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout
from gui import FacenetGui

parser = argparse.ArgumentParser(description='Facenet')
parser.add_argument('-f', help='test path or image/video')
parser.add_argument('-d', help='dataset path')
parser.add_argument('-l', help='log filename')
parser.add_argument('-o', help='output path')
parser.add_argument('-c', help='run in console only',  action='store_true')
parser.add_argument('-t', help='threshold, default: {}'.format(DEFAULT_THRESHOLD) , type=float)
args = parser.parse_args()
if not args.t:
    args.t = DEFAULT_THRESHOLD

if args.c:
    f = FacenetManager()
    f.load_dataset_to_face_detector_manager(args.d)
    f.set_log_filename(args.l)
    f.set_output_path(args.o)
    f.set_threshold(args.t)
    f.process(args.f)
else:
    app = QApplication([])
    win = FacenetGui(args.d, args.t, args.l, args.o)
    win.show()
    app.exec_()