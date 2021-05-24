import cv2
import numpy as np

min_width=80
min_height=80 

offset=6 

pos=550 
detected = []
vcs= 0

    
def Get_centre(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('video.mp4')
subtract = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret , frame1 = cap.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtract.apply(blur)
    curr = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    detect = cv2.morphologyEx (curr, cv2. MORPH_CLOSE , kernel)
    detect = cv2.morphologyEx (detect, cv2. MORPH_CLOSE , kernel)
    contour,h=cv2.findContours(detect,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, pos), (1200, pos), (255,127,0), 3) 
    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        valid_contour = (w >= min_width) and (h >= min_height)
        if not valid_contour:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        centre = Get_centre(x, y, w, h)
        detected.append(centre)
        cv2.circle(frame1, centre, 4, (0, 0,255), -1)

        for (x,y) in detected:
            if y<(pos+offset) and y>(pos-offset):
                vcs+=1
                cv2.line(frame1, (25, pos), (1200, pos), (0,127,255), 3)  
                detected.remove((x,y))
                     
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(vcs), (450, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    # cv2.imshow("detected",detect)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()