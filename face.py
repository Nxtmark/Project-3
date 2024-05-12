import threading

import numpy as np

import mysql.connector
import base64
from PIL import Image
import io

import cv2
from deepface import DeepFace

# mydb = mysql.connector.connect(
#     host="127.0.0.1",
#     user="root",
#     password="",
#     database="kkk"
# )

# # Create a cursor object
# cursor = mydb.cursor()

# # Prepare the query
# query = 'SELECT  `Image` FROM `student` WHERE Student_id=1'

# # Execute the query to get the file
# cursor.execute(query)
# data = cursor.fetchall()

# # The returned data will be a list of list
# image_data = data[0][0]

# # Decode the string
# binary_data = base64.b64decode(image_data)
# image=np.array(binary_data)



# Convert the bytes into a PIL image
#image = Image.open(io.BytesIO(binary_data))

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
 
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

counter = 0

face_match= False

reference_img = cv2.imread("mam.jpg")


def check_face(frame):
    global face_match
    try:
        if (DeepFace.verify(frame, reference_img )['verified']):
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False    


while True:
    ret,frame = cap.read()
    
    if ret:
        if counter % 60 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter +=1
        
        if face_match:
            cv2.putText(frame,"MATCH!",(20,450),cv2.FONT_HERSHEY_SIMPLEX, 2 , (0,255,0) , 3)
        else:
            cv2.putText(frame,"NO MATCH!",(20,450),cv2.FONT_HERSHEY_SIMPLEX, 2 , (0,0,255) , 3)
    
        cv2.imshow("video",frame)
    key=cv2.waitKey(1)
    if key==ord('a'):
        break
    
cv2.destroyAllWindows()