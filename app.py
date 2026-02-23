


import streamlit as st

import cv2

import numpy as np

import tensorflow as tf

import tempfile

import time

import threading

from tensorflow.keras.models import load_model

from rag import retrieve_rule

from voice import speak



# --- 1. MODEL LOADING ---

@st.cache_resource

def load_my_model():

    try:

        # Load exactly as main.py

        return load_model("traffic_classifier_resnet.h5", compile=False)

    except Exception as e:

        st.error(f"Model Error: {e}")

        return None



model = load_my_model()

class_names = [

    'Green_Light', 'Red_Light', 'Speed_Limit_10', 'Speed_Limit_100',

    'Speed_Limit_110', 'Speed_Limit_120', 'Speed_Limit_20', 'Speed_Limit_30',

    'Speed_Limit_40', 'Speed_Limit_50', 'Speed_Limit_60', 'Speed_Limit_70',

    'Speed_Limit_80', 'Speed_Limit_90', 'Stop'

]



# --- SESSION STATE ---

if 'last_spoken' not in st.session_state:

    st.session_state.last_spoken = ""

if 'box_memory' not in st.session_state:

    st.session_state.box_memory = None



def speak_async(text):

    

    threading.Thread(target=speak, args=(text,), daemon=True).start()



# --- 2. UI SETUP ---

st.set_page_config(page_title="Traffic AI Assistant", layout="wide")

st.title("🚦 Smart Traffic Sign Assistant")



st.sidebar.header("Settings")

mode = st.sidebar.radio("Input Source:", ("Webcam", "Upload Video"))



uploaded_file = None

if mode == "Upload Video":

    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "mov", "avi"])



conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.75)

run_app = st.sidebar.checkbox("Run System", value=True)



col1, col2 = st.columns([3, 1])

FRAME_WINDOW = col1.image([])

info_placeholder = col2.empty()



# --- 3. VIDEO CAPTURE ---

cap = None

if run_app:

    if mode == "Webcam":

        cap = cv2.VideoCapture(0)

    elif mode == "Upload Video" and uploaded_file:

        # RESET state when a new video is uploaded to catch the first sign

        st.session_state.last_spoken = ""

        tfile = tempfile.NamedTemporaryFile(delete=False)

        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        # Small warm-up delay to let Streamlit UI stabilize

        time.sleep(0.5)



# --- 4. PROCESSING ENGINE ---

if cap:

    while cap.isOpened() and run_app:

        ret, frame = cap.read()

        if not ret: break

       

        display_frame = frame.copy()

        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (3, 3), 0), cv2.COLOR_BGR2HSV)

       

        # Color Masks

        mask_r = cv2.bitwise_or(

            cv2.inRange(hsv, np.array([0, 150, 100]), np.array([10, 255, 255])),

            cv2.inRange(hsv, np.array([165, 150, 100]), np.array([180, 255, 255]))

        )

        mask_g = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([90, 255, 255]))

        mask = cv2.bitwise_or(mask_r, mask_g)

       

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_detection = None



        for cnt in contours:

            if cv2.contourArea(cnt) < 800: continue

            x, y, w, h = cv2.boundingRect(cnt)

           

            crop = frame[max(0, y-10):y+h+10, max(0, x-10):x+w+10]

            if crop.size > 0:

                test_img = cv2.resize(crop, (224, 224)) / 255.0

                preds = model.predict(np.expand_dims(test_img, axis=0), verbose=0)[0]

                idx = np.argmax(preds)

               

                if preds[idx] > conf_threshold:

                    label = class_names[idx]

                    current_detection = (x, y, w, h, label, preds[idx])

                    break



       # --- 5. INSTANT NOTIFICATION ---
        if current_detection:
            x, y, w, h, label, conf = current_detection
            st.session_state.box_memory = (x, y, w, h, label, conf, time.time())

            if label != st.session_state.last_spoken:
                st.session_state.last_spoken = label
                search_term = label.replace("_", " ")
                
                # 1. Fetch the rule from RAG
                rule_text = retrieve_rule(search_term)
                
                # 2. Extract the first sentence for a quick voice announcement
                short_rule = rule_text.split('.')[0] if rule_text else "Proceed with caution."
                
                # 3. Update the UI Text
                with info_placeholder.container():
                    st.markdown(f"## 📢 {search_term}")
                    st.success(f"**Action:** {short_rule}")
                
                # 4. SPEAK BOTH (This was the missing part!)
                speak_async(f"{search_term}. {short_rule}")



        # --- 6. VISUALS ---

        if st.session_state.box_memory:

            mx, my, mw, mh, mlabel, mconf, mtime = st.session_state.box_memory

            if time.time() - mtime < 0.4:

                cv2.rectangle(display_frame, (mx, my), (mx+mw, my+mh), (0, 255, 0), 3)

                cv2.putText(display_frame, f"{mlabel} {mconf:.2f}", (mx, my-10),

                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)



        FRAME_WINDOW.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))



    cap.release()




