import cv2
import dlib
import math
import numpy as np
import base64
from flask import session

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/thisarakavinda/Thisara/Developments/SLIIT/Research/Experiments/Orginal/Computer-Vision-Lip-Reading-2.0/model/face_weights.dat")

def calibrate(message):
    
    if 'determining_lip_distance' not in session:
        session['determining_lip_distance'] = 120       
    if 'lip_distances' not in session:
        session['lip_distances'] = []
    if 'lip_threshhold' not in session:
        session['lip_threshhold'] = 0
    
    if session['lip_threshhold'] == 0:
        img_data = base64.b64decode(message.split(",")[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        if not faces:
            return "No faces detected"
        
        for face in faces: 

            landmarks = predictor(image=gray, box=face)

            # Calculate the distance between the upper and lower lip landmarks
            mouth_top = (landmarks.part(51).x, landmarks.part(51).y)
            mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)
            lip_distance = math.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1])
            
            session['determining_lip_distance'] -= 1
            distance = landmarks.part(58).y - landmarks.part(50).y 

            session['lip_distances'].append(distance)
            if(session['determining_lip_distance'] == 0):
                LIP_DISTANCE_THRESHOLD = sum(session['lip_distances']) / len(session['lip_distances']) + 4
                session['lip_threshhold'] = LIP_DISTANCE_THRESHOLD
                print("LIP_DISTANCE_THRESHOLD", LIP_DISTANCE_THRESHOLD)
                
            return "CALIBRATING_LIPS"
                
    else: 
        print("Calibration complete")
        print("LIP_DISTANCE_THRESHOLD", session['lip_threshhold'])
        if session["lip_threshhold"] < 30:
            return "COME_CLOSER"
        elif session["lip_threshhold"] > 40:
            return "MOVE_AWAY"
        else:
            return "SUCCESS"