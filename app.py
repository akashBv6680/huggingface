import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import urllib.request
import json

# --------------------
# 1. App Configuration and Model Mapping
# --------------------
st.set_page_config(page_title="Multi-Model Image Classifier", layout="centered")
st.title("Experiment with Pre-trained Models")
st.markdown("Choose a model from the dropdown to classify your image. This app now uses a hybrid approach with `transformers` and `torchvision` for maximum compatibility.")
st.write("---")

# A dictionary mapping user-friendly names to model types and identifiers.
# We'll use this to decide which loading method to use.
MODEL_MAP = {
    "ResNet-50": {"type": "huggingface", "id": "microsoft/resnet-50"},
    "EfficientNetB0": {"type": "huggingface", "id": "google/efficientnet-b0"},
    "VGG16": {"type": "torchvision"},
    "InceptionV3": {"type": "torchvision"},
    "MobileNetV1": {"type": "huggingface", "id": "google/mobilenet_v1_1.0_224"},
}

# --------------------
# 2. Helper Function to get ImageNet labels
# --------------------
@st.cache_data
def get_imagenet_labels():
    """Fetches and caches the ImageNet labels from a known URL."""
    try:
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url) as response:
            labels = json.loads(response.read().decode())
        return labels
    except Exception as e:
        st.error(f"❌ Failed to fetch ImageNet labels: {e}")
        return [f"Label {i}" for i in range(1000)]  # Fallback labels

# --------------------
# 3. Model Loading Function with Caching
# --------------------
@st.cache_resource(show_spinner=False)
def load_model_and_processor(model_name):
    """
    Loads a pre-trained model and its processor using either transformers or torchvision.
    This function is cached to prevent reloading the model on every interaction.
    """
    model_config = MODEL_MAP.get(model_name)
    if not model_config:
        st.error(f"❌ Unknown model selected: {model_name}")
        return None, None

    try:
        st.write(f"Loading model: `{model_name}`")
        
        if model_config["type"] == "huggingface":
            model_id = model_config["id"]
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForImageClassification.from_pretrained(model_id)
            
        elif model_config["type"] == "torchvision":
            if model_name == "VGG16":
                model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
                # Define a standard preprocessing pipeline for torchvision models
                processor = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
            elif model_name == "InceptionV3":
                # InceptionV3 needs a specific input size and requires aux_logits=True
                model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
                processor = transforms.Compose([
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        
        model.eval()  # Set model to evaluation mode
        return processor, model
        
    except Exception as e:
        st.error(f"❌ An error occurred while loading the model: {e}")
        st.error("Please ensure you have an internet connection and the model name is correct.")
        return None, None

# --------------------
# 4. User Interface for Model Selection and Image Upload
# --------------------
selected_model_name = st.selectbox(
    "Select a model to use:",
    options=list(MODEL_MAP.keys())
)

labels = get_imagenet_labels()

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
# 5. Classification Logic
# --------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"Classifying with {selected_model_name}...")

    try:
        image = Image.open(uploaded_file).convert("RGB")
        model_config = MODEL_MAP[selected_model_name]
        
        if model_config["type"] == "huggingface":
            # Preprocess the image with the Hugging Face processor.
            inputs = processor(images=image, return_tensors="pt")
            
        elif model_config["type"] == "torchvision":
            # Preprocess the image with the torchvision processor.
            input_tensor = processor(image)
            inputs = input_tensor.unsqueeze(0) # Add a batch dimension
        
        # Make the classification.
        with torch.no_grad():
            if model_config["type"] == "huggingface":
                 outputs = model(**inputs)
                 logits = outputs.logits
            elif model_config["type"] == "torchvision":
                 outputs = model(inputs)
                 # Handle the special case where InceptionV3 returns a tuple
                 if isinstance(outputs, tuple):
                    logits = outputs[0]
                 else:
                    logits = outputs
             
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        top_5_probs, top_5_indices = torch.topk(probabilities, 5)

        # Display the results.
        st.subheader("Top 5 Predictions:")
        for i in range(5):
            predicted_label = labels[top_5_indices[i].item()]
            confidence = top_5_probs[i].item()
            st.write(f"**{i + 1}.** **{predicted_label}** with a confidence of {confidence:.2f}")

        st.balloons()

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
