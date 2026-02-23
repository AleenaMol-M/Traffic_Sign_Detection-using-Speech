import os
import time
import cv2
import numpy as np
import tensorflow as tf
import threading
from tensorflow.keras.models import load_model
from rag import retrieve_rule 
from voice import speak

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Load Model
model = load_model("traffic_classifier_resnet.h5", compile=False)

class_names = [
    'Green_Light', 'Red_Light', 'Speed_Limit_10', 'Speed_Limit_100', 
    'Speed_Limit_110', 'Speed_Limit_120', 'Speed_Limit_20', 'Speed_Limit_30', 
    'Speed_Limit_40', 'Speed_Limit_50', 'Speed_Limit_60', 'Speed_Limit_70', 
    'Speed_Limit_80', 'Speed_Limit_90', 'Stop'
]

# --- PERSISTENCE VARIABLES ---
last_announced_time = {} 
COOLDOWN_SECONDS = 5
memory_box = None        # Stores (x, y, w, h, label)
ghost_frames = 0         # Counter for how long to keep the "ghost" box
MAX_GHOST_FRAMES = 8     # How many frames to keep the box after it's lost

def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- DETECTION LOGIC ---
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (3,3), 0), cv2.COLOR_BGR2HSV)
    
    # Improved Red Masks (Handles both bright and dark red)
    mask_r1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    mask_g = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([90, 255, 255]))
    mask = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), mask_g)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_this_frame = False

    for cnt in contours:
        if cv2.contourArea(cnt) < 1000: continue 

        x, y, w, h = cv2.boundingRect(cnt)
        crop = frame[max(0, y-15):y+h+15, max(0, x-15):x+w+15]
        if crop.size == 0: continue
        
        # PREDICTION
        test_img = cv2.resize(crop, (224, 224)) / 255.0
        preds = model.predict(np.expand_dims(test_img, axis=0), verbose=0)[0]
        class_id = np.argmax(preds)
        confidence = preds[class_id]
        
        if confidence > 0.65: # Lowered slightly for better persistence
            detected_this_frame = True
            sign_name = class_names[class_id]
            
            # Save to memory and reset ghost counter
            memory_box = (x, y, w, h, sign_name)
            ghost_frames = 0

            # Draw Real-time Box (Bright Green)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(frame, f"{sign_name} {confidence:.2f}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- ANNOUNCEMENT ---
            current_time = time.time()
            if (current_time - last_announced_time.get(sign_name, 0)) > COOLDOWN_SECONDS:
                last_announced_time[sign_name] = current_time
                rule = retrieve_rule(sign_name.replace("_", " "))
                clean_rule = rule.split('.')[0] if rule else ""
                speak_async(f"{sign_name.replace('_', ' ')}. {clean_rule}")

    # --- PERSISTENCE (GHOST) BOX ---
    # If AI misses the sign this frame, draw the last known location from memory
    if not detected_this_frame and memory_box is not None:
        if ghost_frames < MAX_GHOST_FRAMES:
            mx, my, mw, mh, mlabel = memory_box
            # Draw a thinner box to show it's in "memory"
            cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (0, 180, 0), 1)
            cv2.putText(frame, f"Tracking: {mlabel}", (mx, my-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1)
            ghost_frames += 1
        else:
            memory_box = None # Clear memory after enough missed frames

    cv2.imshow("Traffic Assistant", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()