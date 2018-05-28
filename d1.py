import cv2
import numpy as np


hand_cascade = cv2.CascadeClassifier('D:/python/hand.xml')


frame = cv2.imread("D:/143.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
hands = hand_cascade.detectMultiScale(gray, 1.1 , 5)
for(x, y, w, h) in hands:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('hands', frame)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break


