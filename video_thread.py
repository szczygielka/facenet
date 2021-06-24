import cv2
import time
from constants import *
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread


class VideoThread(QThread):
    file_name = ""
    frame_rate = VIDEO_FRAME_RATE
    on_frame_read = None

    def run(self):
        vidcap = cv2.VideoCapture(self.file_name)
        success, image = vidcap.read()
        t1 = time.time()
        frame = 0
        while success:
            success, cv_image = vidcap.read()
            if success:
                self.on_frame_read(cv_image, frame)
                frame += 1
                sleep_time = 1 / self.frame_rate - (time.time() - t1)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    time.sleep(0.001)
                t1 = time.time()