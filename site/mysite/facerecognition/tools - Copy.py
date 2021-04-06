import cv2 
import matplotlib.pyplot as plot
from matplotlib import pyplot as plt
from matplotlib import image as image
import easygui
import numpy as np
import glob

from django.conf import settings

staticPath = settings.STATICFILES_DIRS[0]

print("Ok")
model = cv2.ml.SVM_create()
SVM = model.load(staticPath + "/code/model.xml")
    
    
#Create the HOG descriptor
winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = True

hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)
    

def isFace(imgPath):
    img = cv2.imread(imgPath)
    
    print(img.shape)
    img = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)
   
    testDescriptor = hog.compute(img)
    testDescriptor = np.array([testDescriptor])
    prediction = SVM.predict(testDescriptor)[1].ravel()
    
    return prediction
    
def getPrediction(img):
    try:
        img = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)

        testDescriptor = hog.compute(img)
        testDescriptor = np.array([testDescriptor])
        prediction = SVM.predict(testDescriptor)[1].ravel()

        return prediction
    except:
        return 0
        
def getFaces(inpath, outpath):
    #Get the width and height
    img = cv2.imread(inpath)
    markedImg = img.copy()
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height = img.shape[0]
    width = img.shape[1]
    counter = 0
    
    print(width)

    faceCascade = cv2.CascadeClassifier(staticPath + "/code/lbpcascade_animeface.xml")
    
    
    faces = faceCascade.detectMultiScale(grayImg,
                                 # detector options
                                 scaleFactor = 1.01,
                                 minNeighbors = 5,
                                 minSize = (32, 32))

    for (x, y, w, h) in faces:

        prediction = 0
        scale = 1
        cx = x + w/2
        cy = y + h/2
        scaledH = h * scale
        scaledW = w * scale
        lx = x
        ly = y
        
        i = 0
        
        while prediction != 1 and scale > 0.5:
            #Method 1
            rx = int(lx + scaledW)
            ry = int(ly + scaledH)
            prediction = getPrediction(grayImg[ly:ry, lx:rx])

            if prediction == 1:
                cv2.rectangle(markedImg, (lx,ly), (rx, ry), (0,255,0), 2)
                outputImg = img.copy()
                counter += 1
            else:
                #Method 2
                lx = int(cx - scaledW//2)
                ly = int(cy - scaledH//2)
                rx = int(cx + scaledW//2)
                ry = int(cy + scaledH//2)

                prediction = getPrediction(grayImg[ly:ry, lx:rx])

                if prediction == 1:
                    cv2.rectangle(markedImg, (lx,ly), (rx, ry), (0,255,0), 2)
                    outputImg = img.copy()
                    counter += 1
            
            scale = scale - 0.2
            scaledH = h * scale
            scaledW = w * scale
            i+=1
            
            if i > 150:
                break
    
    cv2.imwrite(outpath, markedImg) 

    return counter