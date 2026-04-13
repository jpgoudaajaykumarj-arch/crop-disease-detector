import streamlit as st
import numpy as np
from PIL import Image
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title='Crop Disease Detector', page_icon='🌾')
st.title('🌾 Crop Disease Detector')
st.write('Upload a Rice or Wheat leaf image to detect disease instantly!')

@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model('model.h5', compile=False)

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

remedies = {
    'rice_bacterial_blight': {'label': '🔴 Rice — Bacterial Blight', 'remedy': 'Apply Copper Oxychloride spray.'},
    'rice_blast': {'label': '🔴 Rice — Blast Disease', 'remedy': 'Apply Tricyclazole fungicide.'},
    'rice_brown_spot': {'label': '🔴 Rice — Brown Spot', 'remedy': 'Apply Mancozeb fungicide.'},
    'rice_healthy': {'label': '✅ Rice — Healthy', 'remedy': 'Your crop is healthy!'},
    'rice_leaf_smut': {'label': '🔴 Rice — Leaf Smut', 'remedy': 'Apply Propiconazole fungicide.'},
    'wheat_healthy': {'label': '✅ Wheat — Healthy', 'remedy': 'Your crop is healthy!'},
    'wheat_rust': {'label': '🔴 Wheat — Rust Disease', 'remedy': 'Apply Propiconazole fungicide.'}
}

uploaded_file = st.file_uploader('📷 Upload Leaf Image', type=['jpg','jpeg','png','webp'])

if uploaded_file is not None:
    try:
        model = load_model()
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = round(float(np.max(prediction)) * 100, 2)
        info = remedies.get(predicted_class, {'label': predicted_class, 'remedy': 'No remedy found.'})
        st.subheader(f"Result: {info['label']}")
        st.write(f"Confidence: {confidence}%")
        if 'healthy' in predicted_class:
            st.success(f"✅ {info['remedy']}")
        else:
            st.error(f"💊 {info['remedy']}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
