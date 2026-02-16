import streamlit as st
from PIL import Image
import os
from datetime import datetime
from fpdf import FPDF
from gtts import gTTS

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
}

def get_text(key, lang):
    return translations.get(key, {}).get(lang, f"[{key}]")

# --- Bilingual class names ---
CLASS_NAMES = {
    "english": ['Corm_Rot', 'Healthy', 'Leaf_Spot', 'Sheath_Rot'],
    "amharic": ['የቆርም ብስባሽ', 'ጤናማ', 'የቅጠል ስፖት', 'የሸለቆች ብስባሽ']
}

# --- Bilingual disease info ---
DISEASE_INFO = {
    "english": {
        "Corm_Rot": {
            "symptoms": "Soft, water-soaked corm tissue, foul smell, yellowing and wilting of leaves.",
            "prevention": "Use clean planting material, improve drainage, avoid waterlogging, rotate crops.",
            "treatment": "Remove infected corms, improve drainage, apply copper-based fungicides, and avoid waterlogging."
        },
        "Healthy": {
            "symptoms": "No visible disease symptoms.",
            "prevention": "Maintain field sanitation, monitor regularly, avoid contaminated tools.",
            "treatment": "No treatment needed. Continue good management."
        },
        "Leaf_Spot": {
            "symptoms": "Brown or black circular spots on leaves, yellow halos, premature leaf drying.",
            "prevention": "Improve air circulation, avoid overhead irrigation, remove plant debris.",
            "treatment": "Remove affected leaves, improve air circulation, apply mancozeb or chlorothalonil fungicides."
        },
        "Sheath_Rot": {
            "symptoms": "Brown lesions on leaf sheaths, rotting tissue, foul smell, stunted growth.",
            "prevention": "Avoid excessive moisture, ensure proper spacing, remove infected residues.",
            "treatment": "Remove infected sheaths, avoid overhead irrigation, apply systemic fungicides."
        }
    },

    "amharic": {
        "Corm_Rot": {
            "symptoms": "ቆርሙ ይበላሸጣል፣ ውሃ የተሞላ ይታያል፣ መበስበስ እና ቅጠሎች መረገፍ ይታያል።",
            "prevention": "ንጹህ ዘር ይጠቀሙ፣ ውሃ መቆም ይከላከሉ፣ እርሻ ይቀያይሩ።",
            "treatment": "የተያዙ ቆርሞችን ያስወግዱ፣ የኮፐር መሰረት ያላቸው ፈንገስ መድሀኒቶች ይጠቀሙ።"
        },
        "Healthy": {
            "symptoms": "ምንም የበሽታ ምልክት የለም።",
            "prevention": "ንፁህ እርሻ ይጠብቁ፣ መሳሪያዎችን ያጠብቁ፣ ተክሎችን በተደጋጋሚ ይመልከቱ።",
            "treatment": "ሕክምና አያስፈልግም።"
        },
        "Leaf_Spot": {
            "symptoms": "በቅጠሎች ላይ ቡናማ ወይም ጥቁር ነጠብጣቦች፣ ቢጫ ክብ ያላቸው ቦታዎች።",
            "prevention": "አየር ዝውውር ያሻሽሉ፣ የላይኛ ውሃ ማጠጣት ይቆሙ፣ ተክል ቅሪቶችን ያስወግዱ።",
            "treatment": "የተያዙ ቅጠሎችን ያስወግዱ፣ mancozeb ወይም chlorothalonil ያሉ መድሀኒቶች ይጠቀሙ።"
        },
        "Sheath_Rot": {
            "symptoms": "በሸለቆ ላይ ቡናማ ቦታዎች፣ መበስበስ፣ የእድገት መቆራረጥ።",
            "prevention": "ተክሎችን በትክክል ይርቁ፣ እርጥበት ይቆጠብ፣ ተያዙ ቅሪቶችን ያስወግዱ።",
            "treatment": "የተያዙ ሸለቆችን ያስወግዱ፣ ስስተሚክ ፈንገስ መድሀኒት ይጠቀሙ።"
        }
    }
}

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

# --- Model loading ---
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

# --- Preload model ---
model, device = load_ensemble_model()
if model is not None and not st.session_state.model_loaded:
    st.session_state.model_loaded = True
    st.session_state.model_loaded_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_status()

# --- PDF Generator ---
def generate_pdf(disease_name, info, lang):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    title = "Enset Disease Farmer Guide" if lang=="english" else "የእንሰት በሽታ መመሪያ"
    pdf.cell(0, 10, txt=title, ln=True)

    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Disease: {disease_name}")

    pdf.ln(5)
    pdf.multi_cell(0, 10, txt="Symptoms:" if lang=="english" else "ምልክቶች:")
    pdf.multi_cell(0, 10, txt=info["symptoms"])

    pdf.ln(5)
    pdf.multi_cell(0, 10, txt="Prevention:" if lang=="english" else "መከላከያ:")
    pdf.multi_cell(0, 10, txt=info["prevention"])

    pdf.ln(5)
    pdf.multi_cell(0, 10, txt="Treatment:" if lang=="english" else "ሕክምና:")
    pdf.multi_cell(0, 10, txt=info["treatment"])

    return pdf.output(dest="S").encode("latin1")

# --- Voice Narration ---
def generate_voice(text, lang):
    tts = gTTS(text=text, lang="am" if lang=="amharic" else "en")
    audio_path = "voice_output.mp3"
    tts.save(audio_path)
    return audio_path

# --- Prediction function ---
def ensemble_predict(image_data):
    import torch
    import torchvision.transforms as transforms

    model, device = load_ensemble_model()
    if model is None:
        return "Error", {}

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

    lang = st.session_state.lang
    english_key = CLASS_NAMES["english"][predicted_idx]

    predicted_label = CLASS_NAMES[lang][predicted_idx]
    disease_info = DISEASE_INFO[lang][english_key]

    return predicted_label, disease_info

# --- Main UI ---
st.title(get_text("app_title", st.session_state.lang))
uploaded_file = st.file_uploader(get_text("upload_image_label", st.session_state.lang), type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=get_text("uploaded_image_caption", st.session_state.lang))

    with st.spinner("Running prediction..."):
        pred_label, info = ensemble_predict(image)

    st.subheader(get_text("prediction_result_header", st.session_state.lang))
    st.write(f"### 🏷️ {pred_label}")

    st.markdown("### 🩺 Symptoms")
    st.write(info["symptoms"])

    st.markdown("### 🛡️ Prevention")
    st.write(info["prevention"])

    st.markdown("### 💊 Treatment")
    st.write(info["treatment"])

    # --- PDF Download ---
    pdf_bytes = generate_pdf(pred_label, info, st.session_state.lang)
    st.download_button(
        label="📄 Download Farmer Guide (PDF)",
        data=pdf_bytes,
        file_name="farmer_guide.pdf",
        mime="application/pdf"
    )

    # --- Voice Narration ---
    narration_text = (
        f"{pred_label}. Symptoms: {info['symptoms']}. Prevention: {info['prevention']}. Treatment: {info['treatment']}"
        if st.session_state.lang == "english"
        else f"{pred_label}። ምልክቶች፦ {info['symptoms']}። መከላከያ፦ {info['prevention']}። ሕክምና፦ {info['treatment']}።"
    )

    audio_file = generate_voice(narration_text, st.session_state.lang)
    st.audio(audio_file, format="audio/mp3")
