import streamlit as st
from PIL import Image
import os
from datetime import datetime
from fpdf import FPDF
from gtts import gTTS

# -----------------------------
# SESSION STATE
# -----------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "english"
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model_loaded_time" not in st.session_state:
    st.session_state.model_loaded_time = None

# -----------------------------
# TRANSLATIONS
# -----------------------------
translations = {
    "app_title": {
        "english": "Enset Fungal Diseases Detection App",
        "amharic": "የእንሰት ፈንገስ በሽታ ማወቂያ መተግበሪያ"
    },
    "upload_image_label": {
        "english": "Choose an image...",
        "amharic": "ምስል ይምረጡ..."
    },
    "uploaded_image_caption": {
        "english": "Uploaded Image",
        "amharic": "የተሰቀለ ምስል"
    },
    "prediction_result_header": {
        "english": "🔍 Prediction Result",
        "amharic": "🔍 የተተነበየ ውጤት"
    }
}

def get_text(key, lang):
    return translations[key][lang]

# -----------------------------
# CLASS NAMES (BILINGUAL)
# -----------------------------
CLASS_NAMES = {
    "english": ["Corm_Rot", "Healthy", "Leaf_Spot", "Sheath_Rot"],
    "amharic": ["የቆርም ብስባሽ", "ጤናማ", "የቅጠል ስፖት", "የሸለቆች ብስባሽ"]
}

# -----------------------------
# DISEASE INFO (SYMPTOMS, PREVENTION, TREATMENT)
# -----------------------------
DISEASE_INFO = {
    "english": {
        "Corm_Rot": {
            "symptoms": "Soft, water-soaked corm tissue, foul smell, yellowing and wilting.",
            "prevention": "Use clean planting material, improve drainage, rotate crops.",
            "treatment": "Remove infected corms and apply copper-based fungicides."
        },
        "Healthy": {
            "symptoms": "No visible disease symptoms.",
            "prevention": "Maintain field sanitation and monitor regularly.",
            "treatment": "No treatment needed."
        },
        "Leaf_Spot": {
            "symptoms": "Brown or black circular spots on leaves.",
            "prevention": "Improve air circulation, avoid overhead irrigation.",
            "treatment": "Apply mancozeb or chlorothalonil fungicides."
        },
        "Sheath_Rot": {
            "symptoms": "Brown lesions on leaf sheaths, rotting tissue.",
            "prevention": "Avoid excessive moisture and remove infected residues.",
            "treatment": "Apply systemic fungicides."
        }
    },

    "amharic": {
        "Corm_Rot": {
            "symptoms": "ቆርሙ ይበላሸጣል፣ ውሃ የተሞላ ይታያል፣ መበስበስ ይታያል።",
            "prevention": "ንጹህ ዘር ይጠቀሙ፣ ውሃ መቆም ይከላከሉ።",
            "treatment": "የተያዙ ቆርሞችን ያስወግዱ፣ የኮፐር መድሀኒት ይጠቀሙ።"
        },
        "Healthy": {
            "symptoms": "ምንም የበሽታ ምልክት የለም።",
            "prevention": "ንፁህ እርሻ ይጠብቁ፣ መሳሪያዎችን ያጠብቁ።",
            "treatment": "ሕክምና አያስፈልግም።"
        },
        "Leaf_Spot": {
            "symptoms": "በቅጠሎች ላይ ቡናማ ወይም ጥቁር ነጠብጣቦች።",
            "prevention": "አየር ዝውውር ያሻሽሉ፣ የላይኛ ውሃ ማጠጣት ይቆሙ።",
            "treatment": "mancozeb ወይም chlorothalonil ይጠቀሙ።"
        },
        "Sheath_Rot": {
            "symptoms": "በሸለቆ ላይ ቡናማ ቦታዎች፣ መበስበስ።",
            "prevention": "ተክሎችን በትክክል ይርቁ፣ እርጥበት ይቆጠብ።",
            "treatment": "ስስተሚክ ፈንገስ መድሀኒት ይጠቀሙ።"
        }
    }
}

# -----------------------------
# MODEL LOADING (NO STREAMLIT UI)
# -----------------------------
def load_ensemble_model():
    import torch
    import torch.nn as nn
    from torchvision import models
    import timm

    num_classes = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class EnsembleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = models.vit_b_16(weights=None)
            self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
            self.swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)

        def forward(self, x):
            return (self.vit(x) + self.swin(x)) / 2

    model = EnsembleModel()
    path = "ensemble_best.pth"

    if not os.path.exists(path):
        return None, device

    checkpoint = torch.load(path, map_location=device)
    model.vit.load_state_dict(checkpoint["vit"])
    model.swin.load_state_dict(checkpoint["swin"])

    model = model.to(device)
    model.eval()
    return model, device

# Preload model safely
model, device = load_ensemble_model()
if model and not st.session_state.model_loaded:
    st.session_state.model_loaded = True
    st.session_state.model_loaded_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def ensemble_predict(image_data):
    import torch
    import torchvision.transforms as transforms

    model, device = load_ensemble_model()
    if model is None:
        return None, None

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    img = tf(image_data).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        idx = torch.argmax(out, dim=1).item()

    lang = st.session_state.lang
    english_key = CLASS_NAMES["english"][idx]

    label = CLASS_NAMES[lang][idx]
    info = DISEASE_INFO[lang][english_key]

    return label, info

# -----------------------------
# PDF GENERATION
# -----------------------------
def generate_pdf(label, info, lang):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    title = "Enset Farmer Guide" if lang == "english" else "የእንሰት ገበሬ መመሪያ"
    pdf.cell(0, 10, txt=title, ln=True)

    pdf.set_font("Arial", size=12)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Disease: {label}")

    pdf.ln(5)
    pdf.multi_cell(0, 10, "Symptoms:" if lang=="english" else "ምልክቶች:")
    pdf.multi_cell(0, 10, info["symptoms"])

    pdf.ln(5)
    pdf.multi_cell(0, 10, "Prevention:" if lang=="english" else "መከላከያ:")
    pdf.multi_cell(0, 10, info["prevention"])

    pdf.ln(5)
    pdf.multi_cell(0, 10, "Treatment:" if lang=="english" else "ሕክምና:")
    pdf.multi_cell(0, 10, info["treatment"])

    return pdf.output(dest="S").encode("latin1")

# -----------------------------
# VOICE GENERATION
# -----------------------------
def generate_voice(text, lang):
    tts = gTTS(text=text, lang="am" if lang=="amharic" else "en")
    audio_path = "voice_output.mp3"
    tts.save(audio_path)
    return audio_path

# -----------------------------
# SIDEBAR (RESTORED)
# -----------------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.write("### Language / ቋንቋ")
st.session_state.lang = st.sidebar.radio("", ["english", "amharic"])

st.sidebar.markdown("---")
st.sidebar.write("### Model Status")
if st.session_state.model_loaded:
    st.sidebar.success(f"Model loaded at {st.session_state.model_loaded_time}")
else:
    st.sidebar.warning("Model not loaded")

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Developed by **Woldekidan Gudelo Dike**")
st.sidebar.write("🏫 Dilla University")
st.sidebar.write("📧 woldekidan.gudelo@du.edu.et")

# -----------------------------
# MAIN UI
# -----------------------------
st.title(get_text("app_title", st.session_state.lang))

uploaded_file = st.file_uploader(
    get_text("upload_image_label", st.session_state.lang),
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption=get_text("uploaded_image_caption", st.session_state.lang))

    with st.spinner("Running prediction..."):
        label, info = ensemble_predict(image)

    if label:
        st.subheader(get_text("prediction_result_header", st.session_state.lang))
        st.write(f"### 🏷️ {label}")

        st.markdown("### 🩺 Symptoms")
        st.write(info["symptoms"])

        st.markdown("### 🛡️ Prevention")
        st.write(info["prevention"])

        st.markdown("### 💊 Treatment")
        st.write(info["treatment"])

        # PDF
        pdf_bytes = generate_pdf(label, info, st.session_state.lang)
        st.download_button(
            "📄 Download Farmer Guide (PDF)",
            data=pdf_bytes,
            file_name="farmer_guide.pdf",
            mime="application/pdf"
        )

        # Voice narration
        narration = (
            f"{label}. Symptoms: {info['symptoms']}. Prevention: {info['prevention']}. Treatment: {info['treatment']}"
            if st.session_state.lang == "english"
            else f"{label}። ምልክቶች፦ {info['symptoms']}። መከላከያ፦ {info['prevention']}። ሕክምና፦ {info['treatment']}።"
        )

        audio_file = generate_voice(narration, st.session_state.lang)
        st.audio(audio_file, format="audio/mp3")
