from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2 as cv

faceDetector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv.VideoCapture(0)
while True:    
    ret, img = cap.read()    
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    faces = faceDetector.detectMultiScale(gray, 1.3, 5)
    #for (x,y,w,h) in faces:
    #    cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        
    rects = dlib.rectangles()
    for (x,y,w,h) in faces:         
        rects.append(dlib.rectangle(x,y,x+w,y+h))

    #roi_gray = gray[y:y+h, x:x+w]
    #roi_color = img[y:y+h, x:x+w]
        #eyes = eyeDetector.detectMultiScale(roi_gray)
        #for (ex,ey,ew,eh) in eyes:
        #    cv.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
    for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	    shape = predictor(gray, rect)
	    shape = face_utils.shape_to_np(shape)

	    # convert dlib's rectangle to a OpenCV-style bounding box
	    # [i.e., (x, y, w, h)], then draw the face bounding box
	    (x, y, w, h) = face_utils.rect_to_bb(rect)
	    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

	    # show the face number
	    cv.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10),
		    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	    # loop over the (x, y)-coordinates for the facial landmarks
	    # and draw them on the image
	    for (x, y) in shape:
		    cv.circle(img, (x, y), 1, (0, 0, 255), -1)

		
	    for i in range(0, 16):
                cv.line(img,(shape[i][0], shape[i][1]),(shape[i+1][0], shape[i+1][1]),(0,255,0),2)

        

    cv.imshow('Output', img)  

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()