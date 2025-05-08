import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA

# Define disease labels (CheXNet is trained for 14 diseases)
disease_labels = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia'
]

# Load CheXNet model from .pth.tar
def load_chexnet_model(pth_tar_path='model.pth.tar', num_classes=14):
    model = models.densenet121(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    checkpoint = torch.load(pth_tar_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)

    # Clean up keys
    new_state_dict = {}
    for k in state_dict:
        nk = k.replace('module.', '')
        if nk.startswith('features.') or nk.startswith('classifier'):
            new_state_dict[nk] = state_dict[k]

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()
    return model

# Apply PCA to compress the image before feeding into model
def pca_compress_image(image_tensor, n_components=50):
    # image_tensor: (C, H, W)
    c, h, w = image_tensor.shape
    compressed_channels = []
    for i in range(c):
        channel = image_tensor[i].numpy().reshape(h, w)
        # Each row is a sample -> so PCA is applied to rows
        pca = PCA(n_components=n_components)
        try:
            compressed = pca.fit_transform(channel)
            decompressed = pca.inverse_transform(compressed)
            compressed_channels.append(torch.tensor(decompressed))
        except Exception as e:
            print(f"PCA compression failed on channel {i}: {e}")
            compressed_channels.append(image_tensor[i])  # fallback

    return torch.stack(compressed_channels)


# Image preprocessing
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Converts to [0,1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image)

# Main Streamlit app
def main():
    st.set_page_config(page_title="CheXNet Lung Disease Prediction", layout="centered")
    st.title("🩺 CheXNet Lung Disease Prediction")
    st.write("Upload a chest X-ray image to detect possible lung conditions using a PCA-compressed DenseNet model.")

    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])
    model = load_chexnet_model()

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        img_tensor = preprocess_image(image)

        with st.spinner("Applying PCA Compression..."):
            compressed_tensor = pca_compress_image(img_tensor, n_components=500)

        input_tensor = compressed_tensor.unsqueeze(0)

        with st.spinner("Predicting..."):
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.sigmoid(outputs).squeeze().numpy()

        # Dictionary of disease:probability
        predictions = {disease_labels[i]: float(probs[i]) for i in range(len(disease_labels))}
        sorted_preds = dict(sorted(predictions.items(), key=lambda x: -x[1]))

        # Medical explanations
        disease_info = {
            'Atelectasis': "Collapsed lung tissue affecting gas exchange.",
            'Cardiomegaly': "Abnormal enlargement of the heart.",
            'Effusion': "Fluid accumulation in pleural space.",
            'Infiltration': "Substance (fluid, pus) in lungs, may suggest pneumonia.",
            'Mass': "Abnormal growth, could be benign or malignant.",
            'Nodule': "Small rounded shadow in lung, possible early tumor.",
            'Pneumonia': "Infection causing inflammation and fluid in lungs.",
            'Pneumothorax': "Air leakage into chest cavity causing lung collapse.",
            'Consolidation': "Lung tissue filled with liquid instead of air.",
            'Edema': "Fluid in lungs due to heart failure or injury.",
            'Emphysema': "Chronic condition destroying air sacs.",
            'Fibrosis': "Scarring of lung tissue reducing elasticity.",
            'Pleural Thickening': "Thickened lining around lungs, post-infection or asbestos.",
            'Hernia': "Diaphragmatic defect allowing organs into chest."
        }

        # Severity helper
        def get_severity(p):
            if p > 0.6:
                return "🔴 High"
            elif p > 0.4:
                return "🟠 Medium"
            else:
                return "🟢 Low"

        st.subheader("📊 Prediction Results with Severity")
        for disease, prob in sorted_preds.items():
            st.markdown(f"**{disease}**: {prob*100:.2f}% — {get_severity(prob)}")
            if disease in disease_info:
                st.caption(f"_Explanation: {disease_info[disease]}_")

        # Bar Chart
        st.subheader("🔬 Visual Risk Distribution")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(list(sorted_preds.keys()), list(sorted_preds.values()), color='skyblue')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        ax.invert_yaxis()
        st.pyplot(fig)

if __name__ == '__main__':
    main()
