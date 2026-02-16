import streamlit as st
from PIL import Image
import os
from datetime import datetime

# --- Session state ---
if "lang" not in st.session_state:
    st.session_state.lang = "english"
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model_loaded_time" not in st.session_state:
    st.session_state.model_loaded_time = None

# --- Translations ---
translations = {
    "app_title": {"english": "Enset Fungal Diseases Detection App", "amharic": "የእንሰት ፈንገስ በሽታ ማወቂያ መተግበሪያ"},
    "upload_image_label": {"english": "Choose an image...", "amharic": "ምስል ይምረጡ..."},
    "uploaded_image_caption": {"english": "Uploaded Image", "amharic": "የተሰቀለ ምስል"},
    "prediction_result_header": {"english": "🔍 Prediction Result", "amharic": "🔍 የተተነበየ ውጤት"},
    "farmer_handbook_header": {"english": "📘 Farmer Handbook", "amharic": "📘 የገበሬ መመሪያ"},
    "upload_image_header": {"english": "📤 Upload Image", "amharic": "📤 ምስል ይስቀሉ"}
}

def get_text(key, lang):
    return translations.get(key, {}).get(lang, f"[{key}]")

@st.cache_data
def generate_handbook(lang):
    handbook = []
    handbook.append(get_text("app_title", lang))
    handbook.append("\n" + "="*50 + "\n")
    handbook.append(get_text("prediction_result_header", lang))
    return "\n".join(handbook)

# --- Sidebar UI ---
selected_lang = st.sidebar.radio("Select Language / ቋንቋ ይምረጡ", ["english", "amharic"])
st.session_state.lang = selected_lang

status_placeholder = st.sidebar.empty()
def update_status():
    if st.session_state.model_loaded:
        status_html = "<span style='color:green; font-weight:bold;'>✅ Model ready</span>"
        if st.session_state.model_loaded_time:
            status_html += f"<br><small>Loaded at {st.session_state.model_loaded_time}</small>"
    else:
        status_html = "<span style='color:orange; font-weight:bold;'>⏳ Model not loaded yet</span>"
    status_placeholder.markdown(status_html, unsafe_allow_html=True)

update_status()

if st.sidebar.button("🔄 Reset Model Status"):
    st.session_state.model_loaded = False
    st.session_state.model_loaded_time = None
    update_status()
    st.sidebar.success("Model status reset. It will reload on next prediction.")

sidebar_handbook = generate_handbook(selected_lang)
st.sidebar.download_button("📥 Download Handbook", sidebar_handbook, file_name="handbook.txt")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='text-align:center; color:gray; font-size:small;'>
    👨‍💻 Developed by <b>Woldekidan Gudelo Dike</b><br>
    🏫 <b>Dilla University</b><br>
    📌 Version 1.0<br>
    📧 <a href="mailto:woldekidan.gudelo@du.edu.et">Contact Developer</a>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Model loading (NO STREAMLIT UI INSIDE) ---
@st.cache_resource
def load_ensemble_model():
    import torch
    import torch.nn as nn
    from torchvision import models
    import timm

    num_classes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class EnsembleModel(nn.Module):
        def __init__(self, num_classes):
            super(EnsembleModel, self).__init__()
            self.vit = models.vit_b_16(weights=None)
            self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
            self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)

        def forward(self, x):
            vit_out = self.vit(x)
            swin_out = self.swin(x)
            return (vit_out + swin_out) / 2

    model = EnsembleModel(num_classes)
    ensemble_model_path = "ensemble_best.pth"

    if os.path.exists(ensemble_model_path):
        checkpoint = torch.load(ensemble_model_path, map_location=device)
        model.vit.load_state_dict(checkpoint['vit'])
        model.swin.load_state_dict(checkpoint['swin'])
    else:
        return None, device

    model = model.to(device)
    model.eval()
    return model, device

# --- Preload model ONCE at startup ---
model, device = load_ensemble_model()

if model is not None and not st.session_state.model_loaded:
    st.session_state.model_loaded = True
    st.session_state.model_loaded_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_status()

def ensemble_predict(image_data):
    import torch
    import torchvision.transforms as transforms

    model, device = load_ensemble_model()
    if model is None:
        st.warning("Model not loaded correctly. Cannot predict.")
        return "Error"

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    img_tensor = eval_tf(image_data).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()

    CLASS_NAMES = ['Corm_Rot', 'Healthy', 'Leaf_Spot', 'Sheath_Rot']
    return CLASS_NAMES[predicted_idx]

# --- Main UI ---
st.title(get_text("app_title", st.session_state.lang))
uploaded_file = st.file_uploader(get_text("upload_image_label", st.session_state.lang), type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=get_text("uploaded_image_caption", st.session_state.lang))
    with st.spinner("Running prediction..."):
        prediction = ensemble_predict(image)
    st.subheader(get_text("prediction_result_header", st.session_state.lang))
    st.write(prediction)
