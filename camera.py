# import cv2
# cam = cv2.VideoCapture(0)
# s, im = cam.read() # captures image
# for i in range(10000000):
#     cv2.imshow("Test Picture", im) # displays captured image
# cv2.imwrite("test.bmp",im) # writes image test.bmp to disk


import cv2
import os
frame_width =640
frame_height = 480
from PIL import Image
import numpy
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Input
from keras.models import Model,load_model
from keras.models import Sequential
from keras.optimizers import Adamax
from keras.utils import np_utils
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter
from matplotlib.image import imread
from scipy.misc import imresize, imsave
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
print("frame_width=",frame_width,"   ","frame_height=",frame_height)
def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = numpy.array([((i / 255.0) ** invGamma) * 255
      for i in numpy.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outp.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
#capture from camera at location 0
cap = cv2.VideoCapture(0)
#set the width and height, and UNSUCCESSFULLY set the exposure time
cap.set(3,640)
cap.set(4,480)
cap.set(15, 0.1)
new_list=[[]]
new_arr = numpy.empty((1,48,48,3))
while True:
    ret, cv2_img = cap.read()
    cv2_img = cv2.flip(cv2_img, 1)

    gamma = 2.5
    cv2_img = adjust_gamma(cv2_img, gamma=gamma)





    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
    )
    print("[INFO] Found {0} Faces.".format(len(faces)))
    roi_color = None
    for (x, y, w, h) in faces:
                roi_color = cv2_img[int(y):int(y + h), int(x):int(x + w)]
                print("[INFO] Object found. Saving locally.")
                break
    if(roi_color is None):
        cv2.imshow("input", cv2_img)
        print("ok bro")
        key = cv2.waitKey(10)
        if key == 27: # Esc key
            break
        continue
    cv2_img_model = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2_img_model)
    #cv2.imshow("thresholded", imgray*thresh2)

    img = img.resize((48, 48))

    #img =img.convert('L')
    arr = numpy.asarray(img)
    #print(arr.shape)
    new_arr[0]=arr

    #print('a',arr)

    #print(new_arr.shape)
    with tf.Session(graph=tf.Graph()) as sess:
          '''
          tf.saved_model.loader.load(sess, ["serve"], FLAGS.export_path)
          graph = tf.get_default_graph()
          #for i in graph.get_operations():
           # print(i)
          images = graph.get_tensor_by_name("duplicate_input_layer:0")
          #scores = graph.get_tensor_by_name("scores")
          '''
          new_model = load_model('path_to_my_model.h5')
          x=new_arr

          y_model=new_model.predict(x)
          print(y_model)
          #print(str(y_model[0][1]))

          for (x, y, w, h) in faces:
                
                if(y_model[0][1] > 0.5):
                    cv2.rectangle(cv2_img, (int(x), y), (x + int(w), int(y + h)), (0, 0, 255), 2)
                    print("bored")
                else:
                    cv2.rectangle(cv2_img, (int(x), y), (x + int(w), int(y + h)), (0, 255, 0), 2)
                    print("not bored")
                break
          sess.close()
    cv2.imshow("input", cv2_img)
    print("ok bro")
    key = cv2.waitKey(10)
    if key == 27: # Esc key
        break


cv2.destroyAllWindows()
cv2.VideoCapture(0).release()
