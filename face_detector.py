import cv2
import numpy as np
import time
from inception_res_net import InceptionResNetV1
import os
from scipy.spatial import distance
from constants import *
from image_manager import *
from utils import * 
from cascade_manager import *
from keras_facenet import FaceNet

'''
Class use to calculate embedings and detect closest face 

'''
class FaceDetector():
    def __init__(self, model_weight_path = INCEPTION_RES_NET_V1_WEIGHTS_PATH, face_size = FACE_SIZE):
        self.model_weight_path = model_weight_path
        self.model = InceptionResNetV1(weights_path=self.model_weight_path)
        # self.embedder = FaceNet()
        self.cascade_manager = CascadeManager()
        self.calculates_images = []
        self.face_size = face_size

    @staticmethod
    def l2_normalize(x, axis=-1, epsilon=1e-10):
        # x / sqrt (sum (square))
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output

    def load_dataset(self, path, images_extensions = IMAGES_EXTENSIONS):
        '''
        Find all images in dataset path and calculate embedings for all images
        All calculated images are collected in calculates_images list variable of CalculatedImage objects
        Steps:
        1. load image
        2. use CascadeManager to gets faces
        3. for all detected faces:
            a. resize image for CNN
            b. use CNN to get embedings
            c. normalize embedings
        4. store calculated data in calculates_images list

        Return number of collected images in calculates_images list
        '''
        files = get_files(indir=path, images_extensions=images_extensions, videos_extensions=[])
        
        for file in files.images_paths:
            cv_image = ImageManager.load_image(file.name, file.path)
            objects = self.cascade_manager.detect_objects(cv_image)
            faces = ImageManager.cut_images(cv_image, objects)
            face_name = file.name.split(".")[0] 
            for face in faces:
                face_resized = np.resize(cv2.resize(face, self.face_size), (1, self.face_size[0], self.face_size[1],3))
                # embedings = self.embedder.embeddings([face])
                embedings = self.model.predict_on_batch(face_resized)
                embedings = self.l2_normalize(embedings)
                self.calculates_images.append(
                    CalculatedImage(
                        file = file,
                        embedings = embedings,
                        face_name = face_name,
                        full_image = face,
                        resized_image = face_resized
                    )
            )
        return len(self.calculates_images)

    def get_embedings(self, image):
        '''
        1. resize image for CNN
        2. use CNN to get embedings
        3. normalize embedings
        '''
        face_resized = np.resize(cv2.resize(image, self.face_size), (1, self.face_size[0], self.face_size[1],3))
        embedings = self.model.predict_on_batch(face_resized)
        embedings = self.l2_normalize(embedings)
        # embedings = self.embedder.embeddings([image])

        return embedings

    @staticmethod
    def calc_dist(image_1_embedings, image_2_embedings):
        '''
        calculated euclidean distance between two embedings 
        '''
        return distance.euclidean(image_1_embedings, image_2_embedings)

    def get_closest(self, cv_face):
        '''
        find closest embedings in data set
        '''
        embedings = self.get_embedings(cv_face)
        closest_image = None
        closest_distance = sys.float_info.max
        for calc_image in self.calculates_images:
            dis = self.calc_dist(calc_image.embedings, embedings)
            if dis <= closest_distance:
                closest_distance = dis
                closest_image = calc_image

        return (closest_distance, closest_image)