import os

# 1. DISABLE INTERNET FOR HUGGINGFACE (Prevents the Timeout errors)
# os.environ['TRANSFORMERS_OFFLINE'] = "1"
# os.environ['HF_DATASETS_OFFLINE'] = "1"
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from rag import retrieve_rule  # Import your RAG lookup function
from voice import speak        # Import your Speech function

# 2. Load your trained model (Using compile=False to avoid Keras 3 metadata errors)
try:
    model = load_model("traffic_classifier_resnet.h5", compile=False)
except Exception as e:
    print(f"Error loading model: {e}")

class_names = [
    'Green_Light', 'Red_Light', 'Speed_Limit_10', 'Speed_Limit_100', 
    'Speed_Limit_110', 'Speed_Limit_120', 'Speed_Limit_20', 'Speed_Limit_30', 
    'Speed_Limit_40', 'Speed_Limit_50', 'Speed_Limit_60', 'Speed_Limit_70', 
    'Speed_Limit_80', 'Speed_Limit_90', 'Stop'
]


def detect_sign_candidate(image_path):
    img = cv2.imread(image_path)
    if img is None: 
        print(f"Error: Could not find image at {image_path}")
        return None, None, None
    
    # Pre-processing
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Color Masks (Red and Green)
    lower_red1, upper_red1 = np.array([0, 150, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([165, 150, 100]), np.array([180, 255, 255])
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([90, 255, 255])
    
    mask_r = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    mask_g = cv2.inRange(hsv, lower_green, upper_green)
    combined_mask = cv2.bitwise_or(mask_r, mask_g)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_crop = None
    best_coords = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 600: continue 

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        num_corners = len(approx)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h

        # Look for circle/octagon or traffic light rectangle
        if (num_corners >= 5) or (0.3 < aspect_ratio < 0.8):
            best_coords = (x, y, w, h)
            # Add 20% padding
            pw, ph = int(w * 0.2), int(h * 0.2)
            y1, y2 = max(0, y-ph), min(img.shape[0], y+h+ph)
            x1, x2 = max(0, x-pw), min(img.shape[1], x+w+pw)
            best_crop = img[y1:y2, x1:x2]
            break 

    return best_crop, img, best_coords

# --- EXECUTION FLOW ---
image_path = "red2.jpg" # Change to your test file
crop, original_img, coords = detect_sign_candidate(image_path)

if crop is not None:
    # STEP A: CNN CLASSIFICATION
    test_img = cv2.resize(crop, (224, 224)) / 255.0
    pred = model.predict(np.expand_dims(test_img, axis=0), verbose=0)[0]
    class_id = np.argmax(pred)
    sign_name = class_names[class_id]
    
 
    # STEP B: RAG RETRIEVAL (The Brain)
    # Clean the name for better PDF matching
    search_query = sign_name.replace("_", " ")
    rule_found = retrieve_rule(search_query)
    
    # --- IMPORTANT: Filter the rule if RAG returned the whole manual ---
    # We only want the line that matches our sign
    if "Rule:" in rule_found:
        lines = rule_found.split('\n')
        # Look for the specific line containing the rule
        for i, line in enumerate(lines):
            if search_query.lower() in line.lower() or "rule" in line.lower():
                # Take this line and potentially the next one
                rule_found = line if "Rule:" in line else lines[min(i+1, len(lines)-1)]
                break

    # STEP C: VISUAL OVERLAY
    x, y, w, h = coords
    cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(original_img, f"{sign_name}", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

    print(f"\n[DETECTED]: {sign_name}")
    print(f"[RAG RULE]: {rule_found}")

    cv2.imshow("Detection Result", original_img)
    
    # STEP D: VOICE ANNOUNCEMENT (The Mouth)
    speak(f"Detected {search_query}. {rule_found}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Zero candidates found. Check image path or camera lighting.")