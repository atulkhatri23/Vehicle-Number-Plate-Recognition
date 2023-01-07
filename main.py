import numpy as np
import pandas as pd
import cv2
import imutils
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from IPython.display import Image
import argparse
import os
import sys
from time import sleep
import board
from digitalio import DigitalInOut
from adafruit_character_lcd.character_lcd import Character_LCD_Mono

lcd = None
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold

inpWidth = 416     #Width of network's input image
inpHeight = 416     #Height of network's input image







def dist(x1, x2, y1, y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5



# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            # if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            # if scores[classId]>confThreshold:
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    cropped = None
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        # calculate bottom and right
        bottom = top + height
        right = left + width

        # crop the plate out
        cropped = frame[top:bottom, left:right].copy()
    if cropped is not None:
        return cropped

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom, frame):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)


# Match contours to license plate or character template
def find_contours(dimensions, img):
    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    ii = cv2.imread('contour.jpg')

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
            x_cntr_list.append(
                intX)  # stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44, 24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY + intHeight, intX:intX + intWidth]
            char = cv2.resize(char, (20, 40))

            cv2.rectangle(ii, (intX, intY), (intWidth + intX, intY + intHeight), (50, 21, 200), 2)

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy)  # List that stores the character's binary image (unsorted)

    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])  # stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,LP_WIDTH/2,LP_HEIGHT/10,2*LP_HEIGHT/3]
    cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list



# Predicting the output
def fix_dimension(img):
    new_img = np.zeros((28, 28, 3))
    for i in range(3):
        new_img[:, :, i] = img
    return new_img


def show_results(count,char):
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i, c in enumerate(characters):
        dic[i] = c

    output = []
    for i, ch in enumerate(char):  # iterating over the characters
        img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)  # preparing image for the model
        y_ = loaded_model(img)[0]  # predicting the class
        y_ = tf.math.argmax(y_).numpy()
        character = dic[y_]  #
        output.append(character)  # storing the result in a list

    plate_number = ''.join(output)
    if(len(plate_number)>1):
        # display text on LCD display \n = new line
        lcd.message = plate_number
        sleep(3)
        return count
    return count+1

def test(image_inp,net,loaded_model):
    cap = cv2.VideoCapture(image_inp)
    count=0
    while cv2.waitKey(1)<0:
    # get frame from the video
        hasFrame, frame = cap.read() #frame: an image object from cv2
        # cv2.imshow('frame',frame)
        # Stop the program if reached end of video
        if not hasFrame:
            break

        # Create a 4D blob from a frame.
        try:
            blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
        except:
            break

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        cropped = postprocess(frame, outs)
        if cropped is not None:
            # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
            #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

            # Write the frame with the detection boxes
            #plt.imshow(cropped)
            #plt.show()
            char=segment_characters(cropped)
            f=count
            count=show_results(count,char)
            if(count==f):
                break
            elif(count==10):
                break
        else:
            ####
            image = frame
            # Resize the image - change width to 500
            image = imutils.resize(image, width=500)
            img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # RGB to Gray scale conversion
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Noise removal with iterative bilateral filter(removes noise while preserving edges)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)

            # Find Edges of the grayscale image
            edged = cv2.Canny(gray, 170, 200)

            # Find contours based on Edges
            cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
            NumberPlateCnt = None #we currently have no Number plate contour

            # loop over our contours to find the best possible approximate contour of number plate
            for c in cnts:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    if len(approx) == 4:  # Select the contour with 4 corners
                        NumberPlateCnt = approx #This is our approx Number Plate Contour
                        x,y,w,h = cv2.boundingRect(c)
                        ROI = img[y:y+h, x:x+w]
                        break

            idx=0
            m=0
            if NumberPlateCnt is None:
                continue
            for i in range(4):
                if NumberPlateCnt[i][0][1]>m:
                    idx=i
                    m=NumberPlateCnt[i][0][1]
            if idx==0:
                pin=3
            else:
                pin=idx-1
            if idx==3:
                nin=0
            else:
                nin=idx+1

            p=dist(NumberPlateCnt[idx][0][0], NumberPlateCnt[pin][0][0], NumberPlateCnt[idx][0][1], NumberPlateCnt[pin][0][1])
            n=dist(NumberPlateCnt[idx][0][0], NumberPlateCnt[nin][0][0], NumberPlateCnt[idx][0][1], NumberPlateCnt[nin][0][1])

            if p>n:
                if NumberPlateCnt[pin][0][0]<NumberPlateCnt[idx][0][0]:
                    left=pin
                    right=idx
                else:
                    left=idx
                    right=pin
                d=p
            else:
                if NumberPlateCnt[nin][0][0]<NumberPlateCnt[idx][0][0]:
                    left=nin
                    right=idx
                else:
                    left=idx
                    right=nin
                d=n
            left_x=NumberPlateCnt[left][0][0]
            left_y=NumberPlateCnt[left][0][1]
            right_x=NumberPlateCnt[right][0][0]
            right_y=NumberPlateCnt[right][0][1]

            opp=right_y-left_y
            hyp=((left_x-right_x)**2+(left_y-right_y)**2)**0.5
            sin=opp/hyp
            theta=math.asin(sin)*57.2958

            image_center = tuple(np.array(ROI.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, theta, 1.0)
            result = cv2.warpAffine(ROI, rot_mat, ROI.shape[1::-1], flags=cv2.INTER_LINEAR)

            if opp>0:
                h=result.shape[0]-opp//2
            else:
                h=result.shape[0]+opp//2

            result=result[0:h, :]
            char=segment_characters(result)
            f=count
            count=show_results(count,char)
            if(count==f):
                break
            elif(count==10):
                break

        cap.release()
        cv2.destroyAllWindows()

        return 0

if __name__ == "__main__":


    # Modify this if you have a different sized character LCD
    lcd_columns = 16
    lcd_rows = 2

    lcd_rs = DigitalInOut(board.D26)
    lcd_en = DigitalInOut(board.D19)
    lcd_d4 = DigitalInOut(board.D13)
    lcd_d5 = DigitalInOut(board.D6)
    lcd_d6 = DigitalInOut(board.D5)
    lcd_d7 = DigitalInOut(board.D11)

    # Initialise the LCD class
    lcd = Character_LCD_Mono(
        lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7, lcd_columns, lcd_rows
    )










    # Load names of classes
    classesFile = "yolo_utils/classes.names";

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "yolo_utils/darknet-yolov3.cfg";
    modelWeights = "yolo_utils/lapi.weights";


    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Create a new model instance
    loaded_model = Sequential()
    loaded_model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
    loaded_model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
    loaded_model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
    loaded_model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
    loaded_model.add(MaxPooling2D(pool_size=(4, 4)))
    loaded_model.add(Dropout(0.4))
    loaded_model.add(Flatten())
    loaded_model.add(Dense(128, activation='relu'))
    loaded_model.add(Dense(36, activation='softmax'))

    # Restore the weights
    loaded_model.load_weights('checkpoints/my_checkpoint')
    path='images/'+sys.argv[1]
    t=test(path,net,loaded_model)
