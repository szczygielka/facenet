from constants import * 
from face_detector import * 
from cascade_manager import *
from pathlib import Path
'''
Class use to process imageas and videos using CascadeManager and FaceDetector
'''
class FacenetManager():
    def __init__(self, cascade_manager = None, log_filename = None, output_path = None, threshold = DEFAULT_THRESHOLD):
        if cascade_manager is None:
            self.cascade_manager = CascadeManager()
        else:
            self.cascade_manager = cascade_manager
                
        self.face_detector = None
        self.log_filename = log_filename
        self.output_path = output_path


    def set_log_filename(self,log_filename):
        self.log_filename = log_filename

    def set_output_path(self,output_path):
        self.output_path = output_path

    def set_threshold(self,threshold):
        self.threshold = threshold

    def set_face_detector_manager(self, face_detector = None):
        if face_detector is None:
            self.face_detector = FaceDetector()
        else:
            self.face_detector = face_detector
        return self

    def load_dataset_to_face_detector_manager(self, dataset_path = None):
        if self.face_detector is None:
            self.set_face_detector_manager()
        if dataset_path is not None and dataset_path is not "":
            self.face_detector.load_dataset(dataset_path)
        return self

    def process_image(self, image, outname = None, filename = None):
        '''
        Function use to find closest face by CascadeManager and FaceDetector
        and highlight finded element on original image

        Steps:
        1. detected_objects by cascade_manager
        2. cut image to gets faces
        3. for all faces:
            a. find closest using FaceDetector
            b. if distance is smaller than threshold:
                ii. add detected face to list
                iii * if log file is not None save data in log file
        4. write information and draw frame on original image where face are detected
        5. if outputpah is not None then save prepared image on disk

        return process image and detections list of information about detected objects:
                    filename, output_path, outname, closest_distance, face_name
        '''
        detections = []
        if self.face_detector is None:
            return image, detections

        detected_objects = self.cascade_manager.detect_objects(image)
        faces = ImageManager.cut_images(image, detected_objects)
        file = None
        if self.log_filename:
            log_file_path = Path(self.log_filename).parent.absolute() 
            if not os.path.exists(log_file_path):
                os.makedirs(log_file_path)
            file = open(self.log_filename, "a")

        names = []
        objects = []
        for face, obj in zip(faces, detected_objects):
            closest_distance, closest_image = self.face_detector.get_closest(face)
            if closest_distance < self.threshold:
                names.append("{:.2f}-{}".format(closest_distance, closest_image.face_name))
                objects.append(obj)
                if self.log_filename:
                    file.write("{}; {}/{}; {:.2f} ; {}\n".format(filename, self.output_path, outname, closest_distance, closest_image.face_name))
                detections.append((filename, self.output_path, outname, closest_distance, closest_image.face_name))
                print("Distance: {:.2f}  Image: {}".format(closest_distance, closest_image.face_name))
        
        if len(names) > 0:
            image = ImageManager.add_frames(image, objects)
            image = ImageManager.add_texts(image, objects, names)
            if self.output_path and outname:
                ImageManager.write_image(image, self.output_path, outname)
        
        return image, detections
            
    def process(self, path = "", after_file_process_callback= None):
        '''
        Process images and videos in path
        Steps:
        1. load list of images and videos in path (recursive), get_files - returns list of File object (contains images path and videos path)
        2. for all images path:
            a. load image
            b. process image by process_image function
            c. * if after_file_process_callback not None call function with image
        2. for all videos path:
            a. load video
            b. for all frames in video:
                ii. process image by process_image function
                iii. * if after_file_process_callback not None call function with image
        '''
        files = get_files(path)
        results_images = []
        for file in files.images_paths:
            image = ImageManager.load_image(file.name, file.path)
            results_images.append(self.process_image(image, file.name, "{}/{}".format(file.path, file.name)))
            if after_file_process_callback:
                after_file_process_callback(results_images[-1])

        for file in files.videos_paths:
            images = Video(file.path, file.name).process().images
            for index, image in enumerate(images):
                results_images.append(self.process_image(image, "{0}/{0}_{1}.png".format(Path(file.name).stem,index),
                    "{}/{}".format(file.path, file.name)))
                if after_file_process_callback:
                    after_file_process_callback(results_images[-1])
                
                

        return results_images