import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Set up Streamlit interface
st.title('Hand Gesture Recognition')

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")
offset = 20
imgSize = 300
labels = ["Hello", "I Love You", "Thanks", "Sick", "Okay", "Hurt", "Help", "Washroom", "Angry", "Play", "Home", "You", "No", "Yes", "GoodMorning", "GoodNight", "Book", "Beautiful", "Cute", "Water", "Sleep", "Mother", "School", "Where", "Ugly", "Worst", "Failure", "Victory"]

while True:
    success, img = cap.read()
    if not success:
        st.error("Failed to read frame from camera")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        try:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            if imgCrop.size == 0:
                raise ValueError("Empty cropped image")

            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            confidence = prediction[index] * 100

            label = f"{labels[index]} ({confidence:.2f}%)"
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 500, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

            # Display results on Streamlit
            st.image(imgOutput, channels="BGR")

        except Exception as e:
            st.error(f"Error: {e}")
            
    else:
        st.warning("No hands detected.")

cap.release()
