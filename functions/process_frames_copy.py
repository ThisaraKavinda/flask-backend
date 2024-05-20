import cv2
import dlib
import math
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import sys
import base64
from constants import *
from constants import TOTAL_FRAMES, VALID_WORD_THRESHOLD, NOT_TALKING_THRESHOLD, PAST_BUFFER_SIZE, LIP_WIDTH, LIP_HEIGHT
from flask import session

label_dict = {12: "මම", 1: "ඔයා", 6: "තියෙනවා", 8: "පාසල", 13: "යනවා", 3: "කොහේද", 4: "ගිවිසුම", 7: "නැන්දා", 11: "බලනවා", 9: "පිළිතුර", 10: "පොත", 2: "කෑම", 5: "ගෙදර", 0: "එනවා"}
count = 0

# Define the input shape
input_shape = (TOTAL_FRAMES, 80, 112, 3)

# restnet_transfer = ResNet50(weights='imagenet', include_top=False,pooling='max')
# restnet_transfer.trainable = False

# # Define the model architecture
# model = tf.keras.Sequential([
#     tf.keras.layers.TimeDistributed(restnet_transfer,input_shape=input_shape),
#     tf.keras.layers.GRU(64),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(len(label_dict), activation='softmax')
# ])

# model.load_weights('/Users/thisarakavinda/Thisara/Developments/SLIIT/Research/Experiments/computer-vision-lip-reading-thiers/Computer-Vision-Lip-Reading-2.0/training/model_weights4.h5', by_name=True)

from tensorflow.keras.models import load_model

model = load_model('/Users/thisarakavinda/Thisara/Developments/SLIIT/Research/Experiments/Sinhala/Demo04/training/model_weights8-1.h5')

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("/Users/thisarakavinda/Thisara/Developments/SLIIT/Research/Experiments/Orginal/Computer-Vision-Lip-Reading-2.0/model/face_weights.dat")

curr_word_frames = []
not_talking_counter = 0
first_word = True
labels = []

past_word_frames = deque(maxlen=PAST_BUFFER_SIZE)

ending_buffer_size = 5
predicted_word_label = None
draw_prediction = False
spoken_already = []

def process_video(message):
    
    img_data = base64.b64decode(message.split(",")[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if not faces:
        return "No faces detected"
    
    for face in faces:
        x1 = face.left()  
        y1 = face.top()  
        x2 = face.right()  
        y2 = face.bottom()  

        landmarks = predictor(image=gray, box=face)

        # Calculate the distance between the upper and lower lip landmarks
        mouth_top = (landmarks.part(51).x, landmarks.part(51).y)
        mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)
        lip_distance = math.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1])

        lip_left = landmarks.part(48).x
        lip_right = landmarks.part(54).x
        lip_top = landmarks.part(50).y
        lip_bottom = landmarks.part(58).y

        # Add padding if necessary to get a 76x110 frame
        width_diff = LIP_WIDTH - (lip_right - lip_left)
        height_diff = LIP_HEIGHT - (lip_bottom - lip_top)
        pad_left = width_diff // 2
        pad_right = width_diff - pad_left
        pad_top = height_diff // 2
        pad_bottom = height_diff - pad_top

        # Ensure that the padding doesn't extend beyond the original frame
        pad_left = min(pad_left, lip_left)
        pad_right = min(pad_right, frame.shape[1] - lip_right)
        pad_top = min(pad_top, lip_top)
        pad_bottom = min(pad_bottom, frame.shape[0] - lip_bottom)

        # Create padded lip region
        lip_frame = frame[lip_top - pad_top:lip_bottom + pad_bottom, lip_left - pad_left:lip_right + pad_right]
        lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))

        
        lip_frame_lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)
        # Apply contrast stretching to the L channel of the LAB image
        l_channel, a_channel, b_channel = cv2.split(lip_frame_lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
        l_channel_eq = clahe.apply(l_channel)

        # Merge the equalized L channel with the original A and B channels
        lip_frame_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
        lip_frame_eq = cv2.cvtColor(lip_frame_eq, cv2.COLOR_LAB2BGR)
        lip_frame_eq= cv2.GaussianBlur(lip_frame_eq, (7, 7), 0)
        lip_frame_eq = cv2.bilateralFilter(lip_frame_eq, 5, 75, 75)
        kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])

        # Apply the kernel to the input image
        lip_frame_eq = cv2.filter2D(lip_frame_eq, -1, kernel)
        lip_frame_eq= cv2.GaussianBlur(lip_frame_eq, (5, 5), 0)
        lip_frame = lip_frame_eq
        
        
        # Draw a circle around the mouth
        # for n in range(48, 61):
        #     x = landmarks.part(n).x
        #     y = landmarks.part(n).y
        #     cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)
        # print(lip_distance, "lip_distance")

        print(lip_distance)
        if 'lip_threshhold' not in session:
            session['lip_threshhold'] = 15
        if lip_distance > session['lip_threshhold']: # person is talking
            print("Talking")
            # cv2.putText(frame, "Talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            global curr_word_frames
            curr_word_frames += [lip_frame.tolist()]
            global not_talking_counter
            not_talking_counter = 0
            global draw_prediction
            draw_prediction = False
            # return "Talking"
            
        else:
            # cv2.putText(frame, "Not talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # print("Not talking")
            not_talking_counter += 1
            if not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE == TOTAL_FRAMES: 

                global past_word_frames
                curr_word_frames = list(past_word_frames) + curr_word_frames

                curr_data = np.array([curr_word_frames[:input_shape[0]]])

                print("*********", curr_data.shape)
                global spoken_already
                print(spoken_already)
                prediction = model.predict(curr_data)

                prob_per_class = []
                for i in range(len(prediction[0])):
                    prob_per_class.append((prediction[0][i], label_dict[i]))
                sorted_probs = sorted(prob_per_class, key=lambda x: x[0], reverse=True)
                for prob, label in sorted_probs:
                    print(f"{label}: {prob:.3f}")
                predicted_class_index = np.argmax(prediction)
                
                # while label_dict[predicted_class_index] in spoken_already:
                    # If the predicted label has already been spoken,
                    # set its probability to zero and choose the next highest probability
                    # prediction[0][predicted_class_index] = 0
                    # predicted_class_index = np.argmax(prediction)
                    
                global predicted_word_label
                predicted_word_label = label_dict[predicted_class_index]
                spoken_already.append(predicted_word_label)
                print("********************************************")
                print("FINISHED!", predicted_word_label)
                print("********************************************")

                draw_prediction = True
                global count
                count = 0

                curr_word_frames = []
                not_talking_counter = 0
            elif not_talking_counter < NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE < TOTAL_FRAMES and len(curr_word_frames) > VALID_WORD_THRESHOLD:
                curr_word_frames += [lip_frame.tolist()]
                not_talking_counter = 0
            elif len(curr_word_frames) < VALID_WORD_THRESHOLD or (not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE > TOTAL_FRAMES):
                curr_word_frames = []

            past_word_frames+= [lip_frame.tolist()]
            if len(past_word_frames) > PAST_BUFFER_SIZE:
                past_word_frames.pop(0)

    if(draw_prediction and count < 20):
        count += 1
        cv2.putText(frame, predicted_word_label, (50 ,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        return predicted_word_label
