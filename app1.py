import streamlit as st
import numpy as np
import cv2
import pickle
import base64

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
Conv2D, BatchNormalization, Activation, Add,
GlobalAveragePooling2D, Dense, Dropout,
MaxPooling2D, Input
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_FRAMES = 20
max_len = 47

def residual_block(x, filters):
    shortcut = x

    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

def build_strong_cnn():
    inputs = Input(shape=(224, 224, 3))

    x = Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128)
    x = MaxPooling2D()(x)

    x = residual_block(x, 128)
    x = residual_block(x, 256)
    x = MaxPooling2D()(x)

    x = residual_block(x, 256)
    x = residual_block(x, 512)

    x = GlobalAveragePooling2D()(x)

    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)

    return Model(inputs, x)

@st.cache_resource
def load_all():
    cnn = build_strong_cnn()
    lstm = load_model("best_model.h5")
    return cnn, lstm

    cnn_model, model = load_all()

    with open("w2i.pkl", "rb") as f:
        w2i = pickle.load(f)

    with open("i2w.pkl", "rb") as f:
        i2w = pickle.load(f)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // MAX_FRAMES)

    count = 0
    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

        count += 1

    cap.release()
    return frames

def extract_features(video_path):
    frames = extract_frames(video_path)
    features = []

    for f in frames:
        f = f.astype("float32") / 255.0
        f = np.expand_dims(f, axis=0)

        feat = cnn_model.predict(f, verbose=0)
        features.append(feat.flatten())

    return np.expand_dims(np.array(features), axis=0)

def generate_caption(feature):
    text = "<start>"

    for _ in range(max_len):
        seq = [w2i[w] for w in text.split() if w in w2i]
        seq = pad_sequences([seq], maxlen=max_len)

        yhat = model.predict([feature, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = i2w.get(yhat)
        if word is None:
            break

        text += " " + word

        if word == "<end>":
            break

    return text.replace("<start>", "").replace("<end>", "").strip()

def show_video(path):
    with open(path, 'rb') as video_file:
        video_bytes = video_file.read()

        video_base64 = base64.b64encode(video_bytes).decode()

        video_html = f"""
        <video width="500" controls>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
        """

st.markdown(video_html, unsafe_allow_html=True)

st.title("🎥 Video Captioning Demo")

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file is not None:
    video_bytes = uploaded_file.read()

    with open("temp_video.mp4", "wb") as f:
        f.write(video_bytes)

    st.subheader("🎥 Uploaded Video")
    show_video("temp_video.mp4")

    st.write("Processing...")

    feature = extract_features("temp_video.mp4")
    caption = generate_caption(feature)

    st.success("Caption:")
    st.write(caption)
