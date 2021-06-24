# github z kodem ktory testowalem:
#https://github.com/nyoki-mtl/keras-facenet
# opis dzialania róznych rzeczy do wykrywania i detekcji twarzy:
#https://medium.com/clique-org/how-to-create-a-face-recognition-model-using-facenet-keras-fd65c0b092f1
# artykul w bardziej zjadliwy sposob:
#https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/

INCEPTION_RES_NET_V1_WEIGHTS_PATH = "model/keras/facenet_keras_weights.h5"
FACE_IMAGES_PATH = "data/images/"
CASCADE_PATH_ALT_2 = 'model/cv2/haarcascade_frontalface_alt2.xml'
CASCADE_PATH_ALT = 'model/cv2/haarcascade_frontalface_alt.xml'
CASCADE_PATH_TREE = 'model/cv2/haarcascade_frontalface_alt_tree.xml'
CASCADE_SCALE_FACTOR = 1.1
CASCADE_MIN_NEIGHBORS = 3
FACE_SIZE = (160, 160)
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_IMAGE_SIZE = 600
IMAGES_EXTENSIONS = ["png", "jpg", "jpeg"]
VIDEOS_EXTENSIONS = ["mp4"]

VIDEO_FRAME_RATE = 60