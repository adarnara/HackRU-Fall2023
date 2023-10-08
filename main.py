import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import vertexai

from langchain.llms import VertexAI
import numpy as np
import torchvision.transforms as T


st.set_page_config(page_title="Skin.AI", layout = "wide")

PROJECT_ID = "lustrous-baton-401321"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location="us-central1")

result = ""
def generate_response(input_text):
    llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
    )
    return llm(f"How do I cure {input_text}")

model = torch.load("vit_hackru.pt", map_location=torch.device('cpu'))

class_dict = {
    0: "Acne and Rosacea",
    1: "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    2: "Atopic Dermatitis Photos",
    3: "Bullous Disease",
    4: "Cellulitis Impetigo and other Bacterial Infections",
    5: "Eczema",
    6: "Exanthems and Drug Eruptions",
    7: "Hair Loss Photos Alopecia and other Hair Diseases",
    8: "Herpes HPV and other STDs",
    9: "Light Diseases and Disorders of Pigmentation",
    10: "Lupus and other Connective Tissue Diseases",
    11: "Melanoma Skin Cancer Nevi and Moles",
    12: "Nail Fungus and other Nail Disease",
    13: "Poison Ivy Photos and other Contact Dermatitis",
    14: "Psoriasis pictures Lichen Planus and related Diseases",
    15: "Scabies Lyme Disease and other Infestations and Bites",
    16: "Seborrheic Keratoses and other Benign Tumors",
    17: "Systemic Disease",
    18: "Tinea Ringworm Candidiasis and other Fungal Infections",
    19: "Urticaria Hives",
    20: "Vascular Tumors",
    21: "Vasculitis",
    22: "Warts Molluscum and other Viral Infections"
}


def load_image(image_file):
    img = Image.open(image_file)
    return img

def predict_skin_condition(model, image) -> int | str:
    try:

        if isinstance(image, Image.Image):

            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            input_tensor = preprocess(image).unsqueeze(0)

            # Make a prediction
            with torch.no_grad():
                output = model(input_tensor).squeeze(0).softmax(0)

            predicted_class_index = output.argmax().item()

            predicted_class_name = class_dict.get(predicted_class_index, "Sorry, couldn't find a matching disease.")

            return predicted_class_name
        else:
            return "Invalid image format. Please upload a valid image (JPEG or PNG)."
    except Exception as e:
        return f"Error: {str(e)}"

st.title("Skin.AI")
st.write("*Vaishnav Venkat, Vignesh Venkat, Pranav Patil, Adarsh Narayanan*")

photo_upload = st.radio("**Would you like to upload an image or take a photo now:**", ["Upload an image", "Take a photo now"])
if photo_upload == "Upload an image":
    label = "Upload an image of your skin or use the camera to take a photo (only .png and .jpeg files are accepted):"
    file = st.file_uploader(label, type=["png", "jpeg"])
    if file is not None:
        st.image(load_image(file))
        if st.button("Diagnose"):  # Add a button to trigger the diagnosis
            result = predict_skin_condition(model, load_image(file))
            st.header(f"Your Diagnosis:{result}")
            st.write(generate_response(result))
else:
    file = st.camera_input("Take a picture")
    if file:
        if file is not None:
            bytes_data = file.getvalue()
            torch_img = torch.ops.image.decode_image(
            torch.from_numpy(np.frombuffer(bytes_data, np.uint8)), 3)
            transform = T.ToPILImage()
            file = transform(torch_img)
            if st.button("Diagnose"):
                result = predict_skin_condition(model, file)
                st.header(f"Your Diagnosis:{result}")

