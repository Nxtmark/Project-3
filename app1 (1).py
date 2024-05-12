import cv2
import deepface
from deepface import DeepFace
import deepface.DeepFace
import tkinter as tk
import tkinter.messagebox
from tkinter import messagebox
from flask import Flask, render_template, Response
import time

app = Flask(__name__)

warnings=0

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    

def generate():
    global result,warnings
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(cv2.CAP_PROP_FPS,15)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform face verification using DeepFace
        face_match = False
        try:
            if deepface.DeepFace.verify(frame, "mam.jpg")["verified"]:
                face_match = True
        except ValueError:
            pass

        # Add text to the frame based on the verification result
        if face_match:
            result="MATCH!"
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            result="NO MATCH!"
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            warnings+=1
            root = tk.Tk()
            root.withdraw()
            warn_message="Warning:{}".format(warnings)
            tkinter.messagebox.showwarning(warn_message)
            if warnings==4:
                cap.release()
                break

        # Encode the frame in JPEG format and yield it
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('examt.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5300',debug=False, threaded=False)