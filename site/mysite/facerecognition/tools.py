import cv2 
import matplotlib.pyplot as plot
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui
import numpy as np
import glob
import tensorflow as tf
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras import optimizers
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import PIL
from PIL import ImageFont, ImageDraw, Image  

from django.conf import settings

staticPath = settings.STATICFILES_DIRS[0]
faceLabels = ['face', 'non-face']
characterLabels = ['Flandre', 'Marisa', 'Reimu', 'Remilia', 'Sakuya']

def loadModel(modelPath, classes):
    #ResNet50 model for the face classification
    model = Sequential()

    # 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
    #model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights = weightsPath))
    model.add(ResNet50(include_top = False, pooling = 'avg'))

    # 2nd layer as Dense for 2-class classification
    model.add(Dense(classes, activation = 'softmax'))

    sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model.load_weights(modelPath)
    
    return model
    
def getFacePrediction(img, labels):
    
    img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    img = img.reshape((1,) + img.shape)

    predictions = faceModel.predict(img, steps=1)
    #print(predictions)
    #print(verbosePredictions(predictions, labels))
    
    if predictions[0][0] > predictions[0][1]:
        return 1
    else:
        return 0

def getCharacterPrediction(img, labels):
    img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    img = img.reshape((1,) + img.shape)

    predictions = touhouModel.predict(img, steps=1)
    
    #print(predictions)
    #print(verbosePredictions(predictions, labels))
    
    highestPrediction = np.amax(predictions[0])
    predictionPercentage = highestPrediction * 100
    predictionIndex = np.argmax(predictions[0])
    character = labels[predictionIndex]
    
    return character, predictionPercentage
    
def verbosePredictions(predictions, labels):
    predictString = ""
    for i in range(0, len(predictions[0])):
        predictString += "%s-%.2f%% " % (labels[i], predictions[0][i] * 100)
    
    return predictString

def getFaces(inpath, outpath, classifyCharacters):
    #Get the width and height
    img = cv2.imread(inpath)
    markedImg = img.copy()
    height = img.shape[0]
    width = img.shape[1]
    potentialFaces = 0
    actualFaces = 0
    #rectangle width
    rW = 2
    
    faceCascade = cv2.CascadeClassifier(staticPath + "/code/lbpcascade_animeface.xml")
    
    faces = faceCascade.detectMultiScale(img,
                                 # detector options
                                 scaleFactor = 1.01,
                                 minNeighbors = 3,
                                 minSize = (32, 32))

    charactersFound = dict.fromkeys(characterLabels, 0)
    
    for (x, y, w, h) in faces:
        potentialFaces += 1
        #cv2.rectangle(markedImg, (x,y), (x + w, y + h), (0,0,255), rW)

        prediction = 0

        #print(potentialFaces)
        prediction = getFacePrediction(img[y:y+h, x:x+w], faceLabels)     
        #cv2.rectangle(markedImg, (lx,ly), (rx, ry), (255,0,0), rW)
        
        if prediction == 1:
            #print("detected")
            outputImg = img.copy()
            actualFaces += 1
            
            #See which charcter it is if we are going to classify the characters
            if classifyCharacters:
                character, characterPrediction = getCharacterPrediction(outputImg[y:y+h, x:x+w], characterLabels)
                resultString = "%s-%.2f%%" % (character, characterPrediction)
                
                #Increment the counter for how many times the character was found in the image
                charactersFound[character] += 1
                
                fontSize = 40
                font = ImageFont.truetype("arial.ttf", fontSize)
                
                while font.getsize(resultString)[0] > w:
                    fontSize -= 1
                    font = ImageFont.truetype("arial.ttf", fontSize)     
                
                fW, fH = font.getsize(resultString)[0], font.getsize(resultString)[1]  
            
                markedImgHSV = cv2.cvtColor(markedImg, cv2.COLOR_BGR2HSV)
                
                markedImgHSV[y+h-fH:y+h,x:x+w,2] = markedImgHSV[y+h-fH:y+h,x:x+w,2] * 0.5
                markedImg = cv2.cvtColor(markedImgHSV, cv2.COLOR_HSV2BGR)
            
                cv2.rectangle(markedImg, (x,y), (x+w, y+h), (255,255,255), rW, lineType=cv2.LINE_AA)
                cv2.rectangle(markedImg, (x,y), (x+w, y+h), (0,0,0), rW - 1, lineType=cv2.LINE_AA)
                
                tempImg = Image.fromarray(markedImg)
                draw = ImageDraw.Draw(tempImg)
                draw.text((x+rW, y+h-rW), resultString, font=font, anchor='lb')  
                #draw.text((x+rW+1, y+rW+1), str(potentialFaces), font=font, anchor='lt') 
                markedImg = np.asarray(tempImg)
            else:
                cv2.rectangle(markedImg, (x,y), (x+w, y+h), (0,255,0), rW * 2, lineType=cv2.LINE_AA)
        
        #cv2.putText(markedImg, str(potentialFaces), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imwrite(outpath, markedImg) 


    return actualFaces, charactersFound
    
faceModel = loadModel(staticPath + "/code/faceModel.hdf5", 2)
touhouModel = loadModel(staticPath + "/code/touhouModel.hdf5", 5)