import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --------------------
# 1. App Configuration and Model Mapping
# --------------------
st.set_page_config(page_title="Multi-Model Image Classifier", layout="centered")
st.title("Experiment with Pre-trained Models")
st.markdown("Choose a model from the dropdown to classify your image. This app uses the Hugging Face `transformers` library to load popular pre-trained models.")
st.write("---")

# A dictionary mapping user-friendly names to Hugging Face model IDs
# The VGG16 and InceptionV3 model names have been corrected.
MODEL_MAP = {
    "ResNet-50": "microsoft/resnet-50",
    "EfficientNetB0": "google/efficientnet-b0",
    "VGG16": "timm/vgg16_bn.tv_in1k",
    "InceptionV3": "timm/inception_v3.tv_in1k",
    "MobileNetV1": "google/mobilenet_v1_1.0_224",
}

# --------------------
# 2. Model Loading Function with Caching
# --------------------
@st.cache_resource(show_spinner=False)
def load_model_and_processor(model_name):
    """
    Loads a pre-trained model and its processor from Hugging Face.
    This function is cached to prevent reloading the model on every interaction.
    """
    try:
        # Get the actual model ID from the map
        model_id = MODEL_MAP.get(model_name)
        if not model_id:
            st.error(f"❌ Unknown model selected: {model_name}")
            return None, None
            
        st.write(f"Loading model: `{model_id}`")
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        return processor, model
    except Exception as e:
        st.error(f"❌ An error occurred while loading the model: {e}")
        st.error("Please ensure the model name is correct and you have an internet connection.")
        return None, None

# --------------------
# 3. User Interface for Model Selection and Image Upload
# --------------------
# Create the dropdown for model selection
selected_model_name = st.selectbox(
    "Select a model to use:",
    options=list(MODEL_MAP.keys())
)

if selected_model_name:
    try:
        with st.spinner(f'Loading {selected_model_name} model...'):
            processor, model = load_model_and_processor(selected_model_name)
        if processor and model:
            st.success(f"✅ Model **{selected_model_name}** loaded successfully!")
        else:
            st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during model loading: {e}")
        st.stop()
        
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to get predictions from the selected model."
)

# --------------------
# 4. Classification Logic
# --------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"Classifying with {selected_model_name}...")

    try:
        # Open and prepare the image.
        image = Image.open(uploaded_file).convert("RGB")
        
        # Preprocess the image for the model.
        inputs = processor(images=image, return_tensors="pt")
        
        # Make the classification.
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the top predictions.
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        top_5_probs, top_5_indices = torch.topk(probabilities, 5)

        # Display the results in a formatted list.
        st.subheader("Top 5 Predictions:")
        for i in range(5):
            predicted_label = model.config.id2label[top_5_indices[i].item()]
            confidence = top_5_probs[i].item()
            st.write(f"**{i + 1}.** **{predicted_label}** with a confidence of {confidence:.2f}")

        st.balloons()

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
