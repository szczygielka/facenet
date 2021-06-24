import cv2
from constants import *

'''
Class use to detect object by CascadeClassifier
'''

class CascadeManager():
    def __init__(self, cascade_paths = [CASCADE_PATH_ALT_2 ], cascade_scale_factor = CASCADE_SCALE_FACTOR, cascade_min_neighbors = CASCADE_MIN_NEIGHBORS):
        self.cascade_paths = cascade_paths
        self.cascade_scale_factor = cascade_scale_factor
        self.cascade_min_neighbors = cascade_min_neighbors

        self.cascades = []
        '''
        Create and load CascadeClassifier from xml file (cascade_paths)
        e.g. haarcascade_frontalface_alt2.xlm
        '''
        for cascade_path in self.cascade_paths:
            self.cascades.append(cv2.CascadeClassifier(cascade_path))
    
    def detect_objects(self, image):
        '''
        Use CascadeClassifier to detect object on image
        return list of detected object (x, y, w, h)
        '''
        objects = []
        for cascade in self.cascades:
            objs = cascade.detectMultiScale(image, scaleFactor = self.cascade_scale_factor, minNeighbors=self.cascade_min_neighbors)
            for o in objs:
                objects.append(o)
        return objects