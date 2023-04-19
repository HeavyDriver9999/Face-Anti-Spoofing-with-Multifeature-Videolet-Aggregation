import cv2
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

x1=0
y1=0
# Load face detection classifier
face_cascade = cv2.CascadeClassifier("C:/Users/ARNAV TALAN(bennett)/Desktop/haarcascade_frontalface_default.xml")
path = r"C:/Users/ARNAV TALAN(bennett)/Downloads/dataset/sample/"
os.chdir(path)
i = 1
wait = 0
# Load video capture device (webcam)
cap = cv2.VideoCapture(0)

# Initialize liveness flag
liveness = False
model = tf.keras.models.load_model("C:/Users/ARNAV TALAN(bennett)/Desktop/AI_PROJECT/saved_model.h5")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(frame, str(datetime.now()), (10, 40),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2, cv2.LINE_AA)
    wait = wait+100
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Check if a face is detected and update liveness flag
    if len(faces) > 0:
        liveness = True
    else:
        liveness = False

    # Display the resulting frame

# Draw rectangle around detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        x1 = x
        y1 = y
    if wait == 500:
        filename = 'Frame'+'.jpg'
        cv2.imwrite(filename, frame)
        i = i+1
        wait = 0
            

        # Load and preprocess the new image
    img_path = "C:/Users/ARNAV TALAN(bennett)/Downloads/dataset/sample/Frame.jpg"
    img = image.load_img(img_path, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0   # rescale the pixel values to [0, 1]

        # Make a prediction
    val = model.predict(x)
    preds=np.argmax(val)
    if preds == 1:
        cv2.putText(frame, "REAL", (x1+50, y1+175), cv2.FONT_HERSHEY_DUPLEX, 1, (124, 252, 0), 2)
        cv2.imshow('frame', frame)
            
    else:
        cv2.putText(frame, "FAKE", (x1+20, y1+195), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', frame)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Check liveness flag

