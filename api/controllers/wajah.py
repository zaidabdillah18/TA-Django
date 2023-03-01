from rest_framework.response import Response
from rest_framework.decorators import api_view
from tensorflow import keras
from datetime import datetime
from PIL import Image
from io import BytesIO
from numpy import asarray
import tensorflow as tf
import os
import re
import base64
import numpy as np


IMAGE_SHAPE = (224,224)

@api_view(['GET'])
def hello(request):
    return Response({'say':'hello'})


@api_view(['POST'])
def multi_predict(request):
    siamese_model = siamese_architecture()
    cwd = os.getcwd()  
    siamese_model.load_weights(cwd+"/api/models/model_shallow.h5")
    
    imgQuery = request.data['query']
    images = request.data['images']

    imgArr1 = get_duplicate_array_image(imgQuery, len(images))
    imgArr2 = get_multi_array_image(images)

    # print(imgArr1.shape)
    # print(imgArr2.shape)

    result = siamese_model.predict([imgArr1,imgArr2])
    # print(result)

    return Response(result)

@api_view(['POST'])
def predict(request):
    siamese_model = siamese_architecture()
    cwd = os.getcwd()  
    siamese_model.load_weights(cwd+"/api/models/model_shallow.h5")

    imgArr1 = get_array_image(request.data['image1'])
    imgArr2 = get_array_image(request.data['image2'])
    # print(imgArr1.shape)
    # print(imgArr2.shape)
    result = siamese_model.predict([imgArr1,imgArr2])

    # print(result)

    return Response({'predict':result[0][0]})

def get_multi_array_image(multi_base64):
    x = []
    for base in multi_base64:
        image = Image.open(BytesIO(base64.b64decode(base)))
        image = image.resize(IMAGE_SHAPE)
        imgArr = asarray(image)
        x.append(imgArr)
    return np.array(x).astype('float32')  

def get_duplicate_array_image(str_base64, num):
    x = []
    for k in range(num):
        image = Image.open(BytesIO(base64.b64decode(str_base64)))
        image = image.resize(IMAGE_SHAPE)
        imgArr = asarray(image)
        x.append(imgArr)
    return np.array(x).astype('float32')  

def get_array_image(str_base64):
    x = []
    image = Image.open(BytesIO(base64.b64decode(str_base64)))
    image = image.resize(IMAGE_SHAPE)
    imgArr = asarray(image)
    x.append(imgArr)

    return np.array(x).astype('float32')   

# def name_generator(name):
#     today = str(datetime.now())
#     today = today.split('.')
#     today = today[0]
#     today = today.split()
#     today = today[0]+today[1]
#     pattern = r':'
#     today = re.sub(pattern, '', today )
#     return today+"_"+re.sub(r' ', '', name )

def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, keras.backend.epsilon()))

def siamese_architecture():
    input = keras.layers.Input((224, 224, 3))

    x = tf.keras.layers.BatchNormalization()(input)
    x = tf.keras.layers.Conv2D(4, (5, 5), activation="tanh")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(8, (5, 5), activation="tanh")(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Flatten()(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(31, activation="tanh")(x) #10 num class
    embedding_network = keras.Model(input, x)


    input_1 = keras.layers.Input((224, 224, 3))
    input_2 = keras.layers.Input((224, 224, 3))

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = keras.layers.Lambda(euclidean_distance)([tower_1, tower_2])
    normal_layer = keras.layers.BatchNormalization()(merge_layer)
    output_layer = keras.layers.Dense(1, activation="sigmoid")(normal_layer)
    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)
    return siamese

