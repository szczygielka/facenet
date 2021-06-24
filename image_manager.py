import cv2
from constants import *
import os


class ImageManager():
    def __init__(self):
        pass

    @staticmethod
    def add_texts(image, objects, names, text_margin = (0, -10), font = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0, 0, 0), thickness = 1):
        output_image = image
        for i in range(len(objects)):
            (x, y, w, h) = objects[i]

            output_image = cv2.putText(output_image, names[i], (x + text_margin[0], y + text_margin[1]), font, 
                   fontScale, color, thickness, cv2.LINE_AA)

        return output_image

    @staticmethod
    def add_frames(image, objects, frame_color = (0, 255, 0), frame_thicknes = 2):
        output_image = image
        for i in range(len(objects)):
            (x, y, w, h) = objects[i]

            cv2.rectangle(output_image, (x, y), (x+w, y+h), frame_color, frame_thicknes)
        return output_image

    @staticmethod
    def cut_images(image, objects):
        images = []
        for (x, y, w, h) in objects:
            img = image[y:y+h, x:x+w]
            images.append(img)
        return images

    @staticmethod
    def load_image(name, path = None):
        if path is None:
            return cv2.imread(name)
        else:
            return cv2.imread("{}/{}".format(path, name))

    @staticmethod
    def write_image(image, path, name):
        path =  os.path.dirname(os.path.abspath("{}\{}".format(path, name)))
        name = name.split("/")[-1].split("\\")[-1]
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite("{}\{}".format(path, name), image)