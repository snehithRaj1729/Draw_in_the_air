import cv2
import numpy as np
import math



cap=cv2.VideoCapture(0)
screen=np.zeros((480,640))
screen_1=np.zeros((1000,1000))
while cap.isOpened():
    _,frame=cap.read()
    cv2.putText(screen_1,"press 's' to start writing",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
    cv2.putText(screen_1, "press 'q' to stop/quit writing", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv2.imshow('instructions',screen_1)
    if cv2.waitKey(1) & 0XFF==ord('s'):
        break

cap.release()
cv2.destroyAllWindows()



cap=cv2.VideoCapture(0)
screen=np.zeros((480,640))
curr=None
prev=None

while cap.isOpened():
        _,frame=cap.read()
        frame=cv2.flip(frame,1)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 48, 0], dtype='uint8')
        upper = np.array([20, 255, 255], dtype='uint8')
        skinregion = cv2.inRange(hsv_frame, lower, upper)
        blur = cv2.GaussianBlur(skinregion, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        ext_top = tuple(c[c[:, :, 1].argmin()][0])
        prev = curr
        curr = ext_top
        if prev and curr:
            cv2.line(screen, prev, curr, (255, 255 , 255), 5)
        cv2.circle(frame, ext_top, 3, (255, 0, 0), -1)
        cv2.imshow('live', frame)
        cv2.imshow('screen', screen)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break



cap.release()
cv2.destroyAllWindows()