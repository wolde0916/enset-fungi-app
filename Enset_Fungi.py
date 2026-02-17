import streamlit as st
from PIL import Image
import os
from fpdf import FPDF

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Enset Fungal Disease Detection",
    layout="centered"
)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "english"

# -------------------------------------------------
# TRANSLATIONS
# -------------------------------------------------
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

def get_text(key):
    return translations[key][st.session_state.lang]

# -------------------------------------------------
# DISEASE INFO
# -------------------------------------------------
DISEASE_INFO = {
    "english": {
        "Corm_Rot": {
            "name": "Corm Rot",
            "symptoms": "Soft, water-soaked corm tissue, foul smell, yellowing and wilting.",
            "prevention": "Use clean planting material, improve drainage, rotate crops.",
            "treatment": "Remove infected corms and apply copper-based fungicides."
        },
        "Healthy": {
            "name": "Healthy",
            "symptoms": "No visible disease symptoms.",
            "prevention": "Maintain field sanitation and monitor regularly.",
            "treatment": "No treatment needed."
        },
        "Leaf_Spot": {
            "name": "Leaf Spot",
            "symptoms": "Brown or black circular spots on leaves.",
            "prevention": "Improve air circulation, avoid overhead irrigation.",
            "treatment": "Apply mancozeb or chlorothalonil fungicides."
        },
        "Sheath_Rot": {
            "name": "Sheath Rot",
            "symptoms": "Brown lesions on leaf sheaths, rotting tissue.",
            "prevention": "Avoid excessive moisture and remove infected residues.",
            "treatment": "Apply systemic fungicides."
        }
    },
    "amharic": {
        "Corm_Rot": {
            "name": "የቆርም ብስባሽ",
            "symptoms": "ቆርሙ ይበላሸጣል፣ ውሃ የተሞላ ይታያል።",
            "prevention": "ንጹህ ዘር ይጠቀሙ፣ ውሃ መቆም ይከላከሉ።",
            "treatment": "የተያዙ ቆርሞችን ያስወግዱ።"
        },
        "Healthy": {
            "name": "ጤናማ",
            "symptoms": "ምንም የበሽታ ምልክት የለም።",
            "prevention": "ንፁህ እርሻ ይጠብቁ።",
            "treatment": "ሕክምና አያስፈልግም።"
        },
        "Leaf_Spot": {
            "name": "የቅጠል ስፖት",
            "symptoms": "በቅጠሎች ላይ ቡናማ ነጠብጣቦች።",
            "prevention": "አየር ዝውውር ያሻሽሉ።",
            "treatment": "mancozeb ይጠቀሙ።"
        },
        "Sheath_Rot": {
            "name": "የሸለቆች ብስባሽ",
            "symptoms": "ቡናማ ቦታዎች በሸለቆ ላይ።",
            "prevention": "እርጥበት ይቆጠብ።",
            "treatment": "ስስተሚክ ፈንገስ መድሀኒት።"
        }
    }
}

# -------------------------------------------------
# PDF GENERATOR
# -------------------------------------------------
def generate_pdf(disease_name, info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    pdf.cell(0, 10, disease_name, ln=True)
    pdf.ln(5)

    pdf.multi_cell(0, 8, f"Symptoms:\n{info['symptoms']}\n")
    pdf.multi_cell(0, 8, f"Prevention:\n{info['prevention']}\n")
    pdf.multi_cell(0, 8, f"Treatment:\n{info['treatment']}")

    return pdf.output(dest="S").encode("latin1")

# -------------------------------------------------
# MODEL LOADING (CACHED – NO UI HERE)
# -------------------------------------------------
@st.cache_resource
from datetime import datetime
from fpdf import FPDF

# -----------------------------
# FORCE CLEAR ALL STREAMLIT CACHES
# -----------------------------
st.cache_data.clear()
st.cache_resource.clear()

# -----------------------------
# SESSION STATE (only language)
# -----------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "english"

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
# DISEASE INFORMATION (BILINGUAL)
# -----------------------------
DISEASE_INFO = {
    "english": {
        "Corm_Rot": {
            "name": "Corm Rot",
            "symptoms": "Soft, water-soaked corm tissue, foul smell, yellowing and wilting.",
            "prevention": "Use clean planting material, improve drainage, rotate crops.",
            "treatment": "Remove infected corms and apply copper-based fungicides."
        },
        "Healthy": {
            "name": "Healthy",
            "symptoms": "No visible disease symptoms.",
            "prevention": "Maintain field sanitation and monitor regularly.",
            "treatment": "No treatment needed."
        },
        "Leaf_Spot": {
            "name": "Leaf Spot",
            "symptoms": "Brown or black circular spots on leaves.",
            "prevention": "Improve air circulation, avoid overhead irrigation.",
            "treatment": "Apply mancozeb or chlorothalonil fungicides."
        },
        "Sheath_Rot": {
            "name": "Sheath Rot",
            "symptoms": "Brown lesions on leaf sheaths, rotting tissue.",
            "prevention": "Avoid excessive moisture and remove infected residues.",
            "treatment": "Apply systemic fungicides."
        }
    },

    "amharic": {
        "Corm_Rot": {
            "name": "የቆርም ብስባሽ",
            "symptoms": "ቆርሙ ይበላሸጣል፣ ውሃ የተሞላ ይታያል፣ መበስበስ ይታያል።",
            "prevention": "ንጹህ ዘር ይጠቀሙ፣ ውሃ መቆም ይከላከሉ።",
            "treatment": "የተያዙ ቆርሞችን ያስወግዱ፣ የኮፐር መድሀኒት ይጠቀሙ።"
        },
        "Healthy": {
            "name": "ጤናማ",
            "symptoms": "ምንም የበሽታ ምልክት የለም።",
            "prevention": "ንፁህ እርሻ ይጠብቁ፣ መሳሪያዎችን ያጠብቁ።",
            "treatment": "ሕክምና አያስፈልግም።"
        },
        "Leaf_Spot": {
            "name": "የቅጠል ስፖት",
            "symptoms": "በቅጠሎች ላይ ቡናማ ወይም ጥቁር ነጠብጣቦች።",
            "prevention": "አየር ዝውውር ያሻሽሉ፣ የላይኛ ውሃ ማጠጣት ይቆሙ።",
            "treatment": "mancozeb ወይም chlorothalonil ይጠቀሙ።"
        },
        "Sheath_Rot": {
            "name": "የሸለቆች ብስባሽ",
            "symptoms": "በሸለቆ ላይ ቡናማ ቦታዎች፣ መበስበስ።",
            "prevention": "ተክሎችን በትክክል ይርቁ፣ እርጥበት ይቆጠብ።",
            "treatment": "ስስተሚክ ፈንገስ መድሀኒት ይጠቀሙ።"
        }
    }
}

# -----------------------------
from fpdf import FPDF
import os

def generate_pdf(disease_name, info, lang):
    pdf = FPDF()
    pdf.add_page()

    # Load Unicode Ethiopic font
    font_path = os.path.join(os.path.dirname(__file__), "AbyssinicaSIL-Regular.ttf")
    pdf.add_font("Abyss", "", font_path, uni=True)
    pdf.set_font("Abyss", size=16)

    title = "Enset Farmer Guide" if lang == "english" else "የእንሰት ገበሬ መመሪያ"
    pdf.multi_cell(0, 10, txt=title)

    pdf.set_font("Abyss", size=14)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=f"Disease: {disease_name}")

    pdf.set_font("Abyss", size=12)
    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=("Symptoms:" if lang=="english" else "ምልክቶች:"))
    pdf.multi_cell(0, 10, txt=info["symptoms"])

    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=("Prevention:" if lang=="english" else "መከላከያ:"))
    pdf.multi_cell(0, 10, txt=info["prevention"])

    pdf.ln(5)
    pdf.multi_cell(0, 10, txt=("Treatment:" if lang=="english" else "ሕክምና:"))
    pdf.multi_cell(0, 10, txt=info["treatment"])

    return pdf.output(dest="S").encode("latin1")

# SIDEBAR
# -----------------------------
selected_lang = st.sidebar.radio("Select Language / ቋንቋ ይምረጡ", ["english", "amharic"])
st.session_state.lang = selected_lang

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Developed by **Woldekidan Gudelo Dike**")
st.sidebar.write("🏫 Dilla University")
st.sidebar.write("📧 woldekidan.gudelo@du.edu.et")

# -----------------------------
# MODEL LOADING (NO CACHING)
# -----------------------------
def load_ensemble_model():
    import torch
    import torch.nn as nn
    from torchvision import models
    import timm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class EnsembleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = models.vit_b_16(weights=None)
            self.vit.heads.head = nn.Linear(
                self.vit.heads.head.in_features, num_classes
            )
            self.swin = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=False,
                num_classes=num_classes
            )
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
    model.to(device)
    model.eval()

    return model, device

# -------------------------------------------------
# LOAD MODEL (UI OUTSIDE CACHE)
# -------------------------------------------------
with st.spinner("Loading model..."):
    model, device = load_ensemble_model()

if model is None:
    st.error("Model file 'ensemble_best.pth' not found.")
    st.stop()

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
def predict(image):
    import torch
    import torchvision.transforms as transforms

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    img = tf(image).unsqueeze(0).to(device)
    model = model.to(device)
    model.eval()
    return model, device

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def ensemble_predict(image_data):
    import torch
    import torchvision.transforms as transforms

    model, device = load_ensemble_model()  # LOAD FRESH EVERY TIME

    if model is None:
        return None

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

    DISEASE_KEYS = ["Corm_Rot", "Healthy", "Leaf_Spot", "Sheath_Rot"]
    return DISEASE_KEYS[idx]
# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.radio(
    "Select Language / ቋንቋ ይምረጡ",
    ["english", "amharic"],
    key="lang"
)

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Woldekidan Gudelo Dike")
st.sidebar.write("🏫 Dilla University")

# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
st.title(get_text("app_title"))

uploaded_file = st.file_uploader(
    get_text("upload_image_label"),
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
    st.image(image, caption=get_text("uploaded_image_caption"))

    with st.spinner("Running prediction..."):
        disease_key = predict(image)

    info = DISEASE_INFO[st.session_state.lang][disease_key]

    st.subheader(get_text("prediction_result_header"))
    st.markdown(f"### 🏷️ {info['name']}")
    st.write(info["symptoms"])
    st.write(info["prevention"])
    st.write(info["treatment"])

    pdf = generate_pdf(info["name"], info)

    st.download_button(
        "📄 Download Farmer Guide (PDF)",
        data=pdf,
        file_name="farmer_guide.pdf",
        mime="application/pdf"
    )
    st.image(image, caption=get_text("uploaded_image_caption", st.session_state.lang))

    with st.spinner("Running prediction..."):
        disease_key = ensemble_predict(image)

    lang = st.session_state.lang
    info = DISEASE_INFO[lang][disease_key]

    st.subheader(get_text("prediction_result_header", lang))
    st.write(f"### 🏷️ {info['name']}")

    st.markdown("### 🩺 Symptoms")
    st.write(info["symptoms"])

    st.markdown("### 🛡️ Prevention")
    st.write(info["prevention"])

    st.markdown("### 💊 Treatment")
    st.write(info["treatment"])

    # PDF DOWNLOAD
    pdf_bytes = generate_pdf(info["name"], info, lang)
    st.download_button(
        label="📄 Download Farmer Guide (PDF)",
        data=pdf_bytes,
        file_name="farmer_guide.pdf",
        mime="application/pdf"
    )
