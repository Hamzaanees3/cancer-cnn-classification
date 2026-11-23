import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Cancer Image Classification",
    page_icon="🔬",
    layout="centered"
)

MODEL_PATH = Path("cancer_cnn_model.keras")
CLASS_NAMES_PATH = Path("class_names.json")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "cancer_cnn_model.keras was not found. Put it in the same folder as app.py."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "class_names.json was not found. Create it using the exact class order "
            "used when training the model. See class_names.example.json."
        )
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or len(names) != 2:
        raise ValueError("class_names.json must contain a JSON list of exactly 2 class names.")
    return names

def prepare_image(image, model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    height = int(input_shape[1]) if input_shape[1] is not None else 224
    width = int(input_shape[2]) if input_shape[2] is not None else 224

    image = image.convert("RGB").resize((width, height))
    array = np.asarray(image, dtype=np.float32)

    # Assumption: training images were scaled to [0, 1] before being passed
    # to the model. If your training pipeline used a different preprocessing
    # method, change this line to match it exactly.
    array = array / 255.0

    return np.expand_dims(array, axis=0)

st.title("🔬 Cancer Image Classification")
st.write(
    "Upload an image for an educational CNN model prediction. "
    "This application is not a medical diagnostic tool."
)

st.warning(
    "For educational demonstration only. Do not use this result to diagnose, "
    "rule out, or make treatment decisions about cancer. Consult a qualified clinician."
)

try:
    model = load_model()
    class_names = load_class_names()
except Exception as error:
    st.error(str(error))
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Classify image", type="primary"):
            with st.spinner("Running model..."):
                batch = prepare_image(image, model)
                prediction = np.asarray(model.predict(batch, verbose=0))

            # This app expects a binary sigmoid output with shape (1, 1),
            # or a two-class softmax output with shape (1, 2).
            if prediction.ndim == 2 and prediction.shape[1] == 1:
                score = float(prediction[0][0])
                index = 1 if score >= 0.5 else 0
                confidence = score if index == 1 else 1.0 - score
            elif prediction.ndim == 2 and prediction.shape[1] == 2:
                probabilities = prediction[0]
                index = int(np.argmax(probabilities))
                confidence = float(probabilities[index])
            else:
                st.error(
                    f"Unsupported model output shape: {prediction.shape}. "
                    "This app is configured for a binary sigmoid or 2-class softmax model."
                )
                st.stop()

            st.subheader("Model output")
            st.metric("Predicted class", class_names[index])
            st.metric("Model score (confidence)", f"{confidence * 100:.2f}%")
            st.caption(
                "The displayed score is the model's output, not a clinically validated "
                "probability of disease."
            )

    except Exception as error:
        st.error(f"Could not process this image: {error}")
