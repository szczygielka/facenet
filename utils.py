import copy
from os.path import isfile, join
from os import listdir
import os
import sys
from constants import * 
import cv2


class CalculatedImage():
    def __init__(self, file, embedings, face_name, full_image, resized_image):
        self.file = file
        self.embedings = embedings
        self.face_name = face_name
        self.full_image = full_image
        self.resized_image = resized_image

class File:
    def __init__(self, path, name):
        self.name = name
        self.path = path

class Files:
    def __init__(self):
        self.images_paths = []
        self.videos_paths = []

class Video:
    def __init__(self, path, file_name):
        self.images = []
        self.path = path
        self.name = file_name

    def process(self):
        vidcap = cv2.VideoCapture("{}/{}".format(self.path, self.name))

        success, image = vidcap.read()
        while success:
            success, cv_image = vidcap.read()
            if success:
                self.images.append(cv_image)

        return self


def get_files(indir, images_extensions = IMAGES_EXTENSIONS, videos_extensions = VIDEOS_EXTENSIONS):
    results = Files()
    if indir is None:
        return results

    def add_file(indir, f):
        extension = f.split(".")[-1]
        if extension in images_extensions:
            results.images_paths.append(File(indir, f))
        if extension in videos_extensions:
            results.videos_paths.append(File(indir, f))

    if isfile(indir):
        path =  os.path.dirname(os.path.abspath(indir))
        name = indir.split("/")[-1].split("\\")[-1]
        add_file(path, name)
    else:
        for f in listdir(indir):
            if isfile("{}/{}".format(indir, f)):
                add_file(indir, f)
            else:
                r = get_files("{}/{}".format(indir, f), images_extensions, videos_extensions)
                results.images_paths += r.images_paths
                results.videos_paths += r.videos_paths

    return results



