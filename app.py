import streamlit as st
import numpy as np
from PIL import Image
import json
import tflite_runtime.interpreter as tflite

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

remedies = {
    'rice_bacterial_blight': {'label': '🔴 Rice — Bacterial Blight', 'remedy': 'Apply copper-based bactericides. Remove infected plants.'},
    'rice_blast': {'label': '🔴 Rice — Blast Disease', 'remedy': 'Apply tricyclazole fungicide. Avoid excess nitrogen.'},
    'rice_brown_spot': {'label': '🔴 Rice — Brown Spot', 'remedy': 'Apply mancozeb fungicide. Improve soil nutrition.'},
    'rice_healthy': {'label': '✅ Rice — Healthy', 'remedy': 'Your crop looks healthy! Keep monitoring regularly.'},
    'wheat_healthy': {'label': '✅ Wheat — Healthy', 'remedy': 'Your crop looks healthy! Keep monitoring regularly.'},
    'wheat_septoria': {'label': '🔴 Wheat — Septoria', 'remedy': 'Apply azoxystrobin fungicide. Remove crop debris.'},
    'wheat_stripe_rust': {'label': '🔴 Wheat — Stripe Rust', 'remedy': 'Apply propiconazole fungicide. Use resistant varieties.'},
}

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path='model.tflite')
    interpreter.allocate_tensors()
    return interpreter

st.set_page_config(page_title='Crop Disease Detector', page_icon='🌾')
st.title('🌾 Crop Disease Detector')
st.write('Upload a Rice or Wheat leaf image to detect disease instantly!')

uploaded_file = st.file_uploader('Upload Leaf Image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        interpreter = load_model()
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = img_array[np.newaxis].astype(np.float32)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])

        predicted_class = class_names[np.argmax(pred)]
        confidence = np.max(pred) * 100

        info = remedies.get(predicted_class, {'label': predicted_class, 'remedy': 'No remedy found.'})
        st.subheader(f"Result: {info['label']}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.info(f"**Remedy:** {info['remedy']}")

    except Exception as e:
        st.error(f"Error: {e}")
