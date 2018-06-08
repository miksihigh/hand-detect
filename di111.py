import cv2
import numpy as np
#import sys

#imagePath = sys.argv[1]
imagePath = ('C:/python/data/52.jpg')
hand_cascade = cv2.CascadeClassifier('C:/python/hand.xml')


frame = cv2.imread(imagePath)
#frame = cv2.imread('C:/python/hands1/3ayRRL-XatI.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
hands = hand_cascade.detectMultiScale(gray, 1.1 , 5)
for(x, y, w, h) in hands:
    print(imagePath, x, y, w, h)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('hands', frame)
    cv2.imwrite(imagePath + '.jpg', frame)
    #cv2.imwrite( 'frame.jpg', frame)
    q = cv2.waitKey(33)
    if q == 27:
        break

cv2.waitKey(0)