import streamlit as st
from PIL import Image
import os
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import timm
import torchvision.transforms as transforms

# --- Session state ---
if "lang" not in st.session_state:
    st.session_state.lang = "english"
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model_loaded_time" not in st.session_state:
    st.session_state.model_loaded_time = None
if "last_prediction_time" not in st.session_state:
    st.session_state.last_prediction_time = None
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0
if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = []  # list of dicts with timestamp + result

# --- Translations ---
translations = {
    "app_title": {"english": "Enset Fungal Diseases Detection App", "amharic": "የእንሰት ፈንገስ በሽታ ማወቂያ መተግበሪያ"},
    "upload_image_label": {"english": "Choose an image...", "amharic": "ምስል ይምረጡ..."},
    "uploaded_image_caption": {"english": "Uploaded Image", "amharic": "የተሰቀለ ምስል"},
    "prediction_result_header": {"english": "🔍 Prediction Result", "amharic": "🔍 የተተነበየ ውጤት"},
    "farmer_handbook_header": {"english": "📘 Farmer Handbook", "amharic": "📘 የገበሬ መመሪያ"},
    "upload_image_header": {"english": "📤 Upload Image", "amharic": "📤 ምስል ይስቀሉ"},
    "symptoms_header": {"english": "Symptoms:", "amharic": "ምልክቶች፡"},
    "treatment_header": {"english": "Treatment:", "amharic": "ሕክምና፡"},
    "select_language": {"english": "Select Language / ቋንቋ ይምረጡ", "amharic": "ቋንቋ ይምረጡ / Select Language"},
    "model_ready": {"english": "✅ Model ready", "amharic": "✅ ሞዴል ዝግጁ ነው"},
    "model_loaded_at": {"english": "Loaded at", "amharic": "በሰዓት የተጫነ"},
    "model_not_loaded": {"english": "⏳ Model not loaded yet", "amharic": "⏳ ሞዴል ገና አልተጫነም"},
    "reset_model_status": {"english": "🔄 Reset Model Status", "amharic": "🔄 የሞዴል ሁኔታን ዳግም አስጀምር"},
    "model_status_reset": {"english": "Model status reset. It will reload on next prediction.", "amharic": "የሞዴል ሁኔታ ዳግም ተጀምሯል፡፡ በሚቀጥለው ትንበያ ይጫናል።"},
    "download_handbook": {"english": "📥 Download Handbook", "amharic": "📥 መመሪያውን ያውርዱ"},
    "download_session_log": {"english": "📊 Download Session Log (CSV)", "amharic": "📊 የክፍለ ጊዜ መዝገብ ያውርዱ (CSV)"},
    "last_prediction_at": {"english": "🕒 Last prediction at", "amharic": "🕒 የመጨረሻ ትንበያ በ"},
    "total_predictions": {"english": "📊 Total predictions this session:", "amharic": "📊 በዚህ ክፍለ ጊዜ ጠቅላላ ትንበያዎች፡"},
    "developed_by": {"english": "👨‍💻 Developed by", "amharic": "👨‍💻 የተዘጋጀው በ"},
    "version": {"english": "📌 Version", "amharic": "📌 ስሪት"},
    "contact_developer": {"english": "📧 Contact Developer", "amharic": "📧 ገንቢውን ያግኙ"},
    "loading_model_weights": {"english": "Loading model weights...", "amharic": "የሞዴል ክብደቶችን በመጫን ላይ..."},
    "loading_vit_swin": {"english": "Loading ViT and Swin models...", "amharic": "ViT እና Swin ሞዴሎችን በመጫን ላይ..."},
    "loading_vit_weights": {"english": "Loading ViT weights...", "amharic": "ViT ክብደቶችን በመጫን ላይ..."},
    "loading_swin_weights": {"english": "Loading Swin weights...", "amharic": "Swin ክብደቶችን በመጫን ላይ..."},
    "model_loaded_success": {"english": "Model loaded successfully!", "amharic": "ሞዴሉ በተሳካ ሁኔታ ተጭኗል!"},
    "error_loading_weights": {"english": "❌ Error loading model weights:", "amharic": "❌ የሞዴል ክብደቶችን በመጫን ላይ ስህተት ተፈጠረ፡"},
    "weights_not_found": {"english": "❌ Ensemble model weights not found at", "amharic": "❌ የተጣመረው ሞዴል ክብደቶች አልተገኙም"},
    "model_not_loaded_correctly": {"english": "Model not loaded correctly. Cannot predict.", "amharic": "ሞዴሉ በትክክል አልተጫነም፡፡ መተንበይ አይቻልም።"},
    "prediction_failed": {"english": "Prediction failed.", "amharic": "ትንበያ አልተሳካም።"},
    "running_prediction": {"english": "Running prediction...", "amharic": "ትንበያ በመስራት ላይ..."},
    "filter_predictions": {"english": "Filter predictions by type:", "amharic": "ትንበያዎችን በአይነት አጣራ፡"},
    "all": {"english": "All", "amharic": "ሁሉም"},
    "session_log": {"english": "📊 Session Log", "amharic": "📊 የክፍለ ጊዜ መዝገብ"}
}

def get_text(key, lang):
    return translations.get(key, {}).get(lang, f"[{key}]")

# --- Disease Class Names Map (for display and logging) ---
DISEASE_CLASS_NAMES_MAP = {
    0: {"english": "Corm_Rot", "amharic": "የሥር መበስበስ"},
    1: {"english": "Healthy", "amharic": "ጤናማ"},
    2: {"english": "Leaf_Spot", "amharic": "ቅጠል ነጠብጣብ"},
    3: {"english": "Sheath_Rot", "amharic": "ግንድ መበስበስ"}
}

# --- Disease Information (Symptoms and Treatment) ---
disease_info = {
    "Corm_Rot": {
        "english": {
            "symptoms": [
                "Soft, watery rot at the base of the plant.",
                "Yellowing and wilting of lower leaves.",
                "Foul odor from affected corm.",
                "Plant eventually collapses."
            ],
            "treatment": [
                "Remove and destroy infected plants.",
                "Improve soil drainage.",
                "Apply fungicides containing metalaxyl or propamocarb.",
                "Practice crop rotation."
            ]
        },
        "amharic": {
            "symptoms": [
                "በተክሉ ሥር ላይ ለስላሳ፣ ውሃማ መበስበስ።",
                "የታችኛው ቅጠሎች ወደ ቢጫነት መቀየር እና መድረቅ።",
                "ከተጎዳው የሥር ክፍል የሚወጣ መጥፎ ሽታ።",
                "ተክሉ በመጨረሻ ይወድቃል።"
            ],
            "treatment": [
                "የተጎዱ ተክሎችን አስወግደው ያጥፉ።",
                "የአፈር ፍሳሽን ያሻሽሉ።",
                "metalaxyl ወይም propamocarb የያዙ ፀረ-ፈንገስ መድኃኒቶችን ይጠቀሙ።",
                "የሰብል ሽክርክርን ይለማመዱ።"
            ]
        }
    },
    "Healthy": {
        "english": {
            "symptoms": ["No visible signs of disease."],
            "treatment": ["Maintain good agricultural practices."]
        },
        "amharic": {
            "symptoms": ["ምንም የበሽታ ምልክቶች አይታዩም።"],
            "treatment": ["ጥሩ የግብርና አሰራሮችን ይቀጥሉ።"]
        }
    }
    ,
    "Leaf_Spot": {
        "english": {
            "symptoms": [
                "Small, circular to irregular dark spots on leaves.",
                "Spots may have a yellow halo.",
                "Severe infection can lead to leaf blight and defoliation."
            ],
            "treatment": [
                "Remove and destroy infected leaves.",
                "Improve air circulation around plants.",
                "Apply copper-based fungicides or mancozeb.",
                "Avoid overhead irrigation."
            ]
        },
        "amharic": {
            "symptoms": [
                "በቅጠሎች ላይ ትናንሽ፣ ክብ ወይም ያልተስተካከሉ ጥቁር ነጠብጣቦች።",
                "ነጠብጣቦቹ ቢጫ ሃሎ ሊኖራቸው ይችላል።",
                "ከባድ ኢንፌክሽን ወደ ቅጠል መበስበስ እና ቅጠሎች መርገፍ ሊያመራ ይችላል።"
            ],
            "treatment": [
                "የተበከሉ ቅጠሎችን አስወግደው ያጥፉ።",
                "በተክሎች ዙሪያ የአየር ዝውውርን ያሻሽሉ።",
                "መዳብ የያዙ ፀረ-ፈንገስ መድኃኒቶችን ወይም mancozeb ይጠቀሙ።",
                "ከላይ የሚደረግ መስኖን ያስወግዱ።"
            ]
        }
    },
    "Sheath_Rot": {
        "english": {
            "symptoms": [
                "Rotting of leaf sheaths, often at the water line.",
                "Discoloration (brown to black) on the sheaths.",
                "Soft, mushy texture of affected sheaths."
            ],
            "treatment": [
                "Remove affected leaf sheaths.",
                "Improve drainage and reduce humidity.",
                "Apply fungicides like benomyl or carbendazim.",
                "Ensure proper plant spacing."
            ]
        },
        "amharic": {
            "symptoms": [
                "የቅጠል ሻጭ መበስበስ፣ ብዙውን ጊዜ በውሃ መስመር ላይ።",
                "በሻጮች ላይ ቀለም መቀየር (ቡናማ እስከ ጥቁር)።",
                "የተጎዱ ሻጮች ለስላሳ፣ ጭቃማ ሸካራነት።"
            ],
            "treatment": [
                "የተጎዱ የቅጠል ሻጮችን ያስወግዱ።",
                "ፍሳሽን ያሻሽሉ እና እርጥበትን ይቀንሱ።",
                "እንደ benomyl ወይም carbendazim ያሉ ፀረ-ፈንገስ መድኃኒቶችን ይጠቀሙ።",
                "ትክክለኛ የእፅዋት ርቀትን ያረጋግጡ።"
            ]
        }
    }
}

def generate_handbook(lang):
    handbook = []
    handbook.append(get_text("app_title", lang))
    handbook.append("\n" + "="*50 + "\n")
    handbook.append(get_text("farmer_handbook_header", lang))
    handbook.append("\n" + "="*50 + "\n")
    for disease_idx in DISEASE_CLASS_NAMES_MAP:
        disease_english_name = DISEASE_CLASS_NAMES_MAP[disease_idx]["english"]
        disease_display_name = DISEASE_CLASS_NAMES_MAP[disease_idx][lang]

        handbook.append(f"Disease: {disease_display_name} ({disease_english_name})")

        if disease_english_name in disease_info:
            current_lang_info = disease_info[disease_english_name][lang]
            handbook.append(get_text("symptoms_header", lang))
            for symptom in current_lang_info["symptoms"]:
                handbook.append(f"- {symptom}")
            handbook.append(get_text("treatment_header", lang))
            for treatment in current_lang_info["treatment"]:
                handbook.append(f"- {treatment}")
        else:
            handbook.append(f"No detailed information available for {disease_display_name}.")
        handbook.append("\n" + "-"*20 + "\n")
    return "\n".join(handbook)

# --- Image Resizing for Display ---
def resize_image_for_display(image: Image.Image, max_dimension=500) -> Image.Image:
    width, height = image.size
    if max(width, height) <= max_dimension:
        return image

    aspect_ratio = width / height
    if width > height:
        new_width = max_dimension
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_dimension
        new_width = int(new_height * aspect_ratio)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

# --- Sidebar UI FIRST ---
lang_options = ["english", "amharic"]
initial_lang_index = lang_options.index(st.session_state.lang)
selected_lang = st.sidebar.radio(
    get_text("select_language", st.session_state.lang),
    lang_options,
    index=initial_lang_index
)
st.session_state.lang = selected_lang

# Status badge placeholder
status_placeholder = st.sidebar.empty()
def update_status():
    if st.session_state.model_loaded:
        status_html = f"<span style='color:green; font-weight:bold;'>{get_text('model_ready', st.session_state.lang)}</span>"
        if st.session_state.model_loaded_time:
            status_html += f"<br><small>{get_text('model_loaded_at', st.session_state.lang)} {st.session_state.model_loaded_time}</small>"
    else:
        status_html = f"<span style='color:orange; font-weight:bold;'>{get_text('model_not_loaded', st.session_state.lang)}</span>"
    status_placeholder.markdown(status_html, unsafe_allow_html=True)

update_status()

# Prediction info placeholders
prediction_time_placeholder = st.sidebar.empty()
prediction_count_placeholder = st.sidebar.empty()

if st.session_state.last_prediction_time:
    prediction_time_placeholder.markdown(
        f"<small>{get_text('last_prediction_at', st.session_state.lang)} {st.session_state.last_prediction_time}</small>",
        unsafe_allow_html=True
    )
prediction_count_placeholder.markdown(
    f"<small>{get_text('total_predictions', st.session_state.lang)} {st.session_state.prediction_count}</small>",
    unsafe_allow_html=True
)

# Reset button
if st.sidebar.button(get_text("reset_model_status", st.session_state.lang)):
    st.session_state.model_loaded = False
    st.session_state.model_loaded_time = None
    st.session_state.last_prediction_time = None
    st.session_state.prediction_count = 0
    st.session_state.prediction_log = []
    update_status()
    prediction_time_placeholder.empty()
    prediction_count_placeholder.empty()
    st.sidebar.success(get_text("model_status_reset", st.session_state.lang))

# Handbook download
sidebar_handbook = generate_handbook(selected_lang)
st.sidebar.download_button(get_text("download_handbook", st.session_state.lang), sidebar_handbook, file_name="farmer_handbook.txt")

# Session log download
if st.session_state.prediction_log:
    df_log = pd.DataFrame(st.session_state.prediction_log)
    csv_log = df_log.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(get_text("download_session_log", st.session_state.lang), csv_log, file_name="prediction_log.csv")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style='text-align:center; color:gray; font-size:small;'>
    {get_text('developed_by', st.session_state.lang)} <b>Woldekidan Gudelo Dike</b><br>
    🏫 <b>Dilla University</b><br>
    {get_text('version', st.session_state.lang)} 1.0<br>
    {get_text('contact_developer', st.session_state.lang)} <a href="mailto:woldekidan.gudelo@du.edu.et">Contact Developer</a>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Lazy imports and model logic ---
def load_ensemble_model():
    if st.session_state.model_loaded:
        return st.session_state.ensemble_model, st.session_state.device

    num_classes = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class EnsembleModel(nn.Module):
        def __init__(self, num_classes):
            super(EnsembleModel, self).__init__()
            # Initialize ViT with pretrained=False because we load weights manually
            self.vit = models.vit_b_16(weights=None)
            self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
            self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)

        def forward(self, x):
            vit_out = self.vit(x)
            swin_out = self.swin(x)
            return (vit_out + swin_out) / 2

    ensemble_model_path = "ensemble_best.pth"
    progress_text = st.empty()
    progress = progress_text.progress(0, text=get_text("loading_model_weights", st.session_state.lang))

    model = EnsembleModel(num_classes)
    progress.progress(30, text=get_text("loading_vit_swin", st.session_state.lang))

    if os.path.exists(ensemble_model_path):
        try:
            checkpoint = torch.load(ensemble_model_path, map_location=device)
            model.vit.load_state_dict(checkpoint['vit'])
            progress.progress(60, text=get_text("loading_vit_weights", st.session_state.lang))
            model.swin.load_state_dict(checkpoint['swin'])
            progress.progress(90, text=get_text("loading_swin_weights", st.session_state.lang))
        except Exception as e:
            st.error(f"{get_text('error_loading_weights', st.session_state.lang)} {e}")
            return None, device
    else:
        st.error(f"{get_text('weights_not_found', st.session_state.lang)} {ensemble_model_path}")
        return None, device

    model = model.to(device)
    model.eval()
    progress.progress(100, text=get_text("model_loaded_success", st.session_state.lang))
    progress_text.empty() # Clear the progress bar after completion
    st.success(get_text("model_loaded_success", st.session_state.lang))

    st.session_state.model_loaded = True
    st.session_state.model_loaded_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.ensemble_model = model
    st.session_state.device = device
    update_status()

    return model, device

def ensemble_predict(image_data):
    model, device = load_ensemble_model()
    if model is None:
        st.warning(get_text("model_not_loaded_correctly", st.session_state.lang))
        return -1 # Return an invalid index for error

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

    # Log the English name for consistency with disease_info dictionary keys
    predicted_english_name = DISEASE_CLASS_NAMES_MAP[predicted_idx]["english"]

    # Update prediction info in sidebar
    st.session_state.last_prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.prediction_count += 1
    st.session_state.prediction_log.append({
        "timestamp": st.session_state.last_prediction_time,
        "result": predicted_english_name # Log the English key
    })

    prediction_time_placeholder.markdown(
        f"<small>{get_text('last_prediction_at', st.session_state.lang)} {st.session_state.last_prediction_time}</small>",
        unsafe_allow_html=True
    )
    prediction_count_placeholder.markdown(
        f"<small>{get_text('total_predictions', st.session_state.lang)} {st.session_state.prediction_count}</small>",
        unsafe_allow_html=True
    )

    return predicted_idx # Return the index

# --- Main UI ---
st.title(get_text("app_title", st.session_state.lang))
uploaded_file = st.file_uploader(get_text("upload_image_label", st.session_state.lang), type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Resize image for display
    display_image = resize_image_for_display(image)
    st.image(display_image, caption=get_text("uploaded_image_caption", st.session_state.lang), width=500)

    with st.spinner(get_text("running_prediction", st.session_state.lang)):
        predicted_class_idx = ensemble_predict(image)

    if predicted_class_idx != -1: # Check for valid prediction index
        prediction_display_name = DISEASE_CLASS_NAMES_MAP[predicted_class_idx][st.session_state.lang]
        prediction_english_key = DISEASE_CLASS_NAMES_MAP[predicted_class_idx]["english"]

        st.subheader(get_text("prediction_result_header", st.session_state.lang))
        st.success(f"✅ {get_text('prediction_result_header', st.session_state.lang)}: {prediction_display_name}")

        # Display Symptoms and Treatment using the ENGLISH KEY for lookup in disease_info
        if prediction_english_key in disease_info:
            current_lang_info = disease_info[prediction_english_key][st.session_state.lang]
            st.markdown(f"### {get_text('symptoms_header', st.session_state.lang)}")
            for symptom in current_lang_info["symptoms"]:
                st.write(f"- {symptom}")

            st.markdown(f"### {get_text('treatment_header', st.session_state.lang)}")
            for treatment in current_lang_info["treatment"]:
                st.write(f"- {treatment}")
        else:
            st.write(f"No detailed information available for {prediction_english_key}.")

        # Display session log table in main panel
        if st.session_state.prediction_log:
            st.subheader(get_text("session_log", st.session_state.lang))

            # Convert log to DataFrame
            df_log = pd.DataFrame(st.session_state.prediction_log)

            # Filter options - display translated names in selectbox, but filter by English keys
            filter_options_display = [get_text("all", st.session_state.lang)]
            # Create a map for display names (English -> Translated)
            display_name_map = {DISEASE_CLASS_NAMES_MAP[i]["english"]: DISEASE_CLASS_NAMES_MAP[i][st.session_state.lang] for i in DISEASE_CLASS_NAMES_MAP.keys()}

            unique_logged_results = df_log["result"].unique().tolist()
            for res_english_key in unique_logged_results:
                filter_options_display.append(display_name_map.get(res_english_key, res_english_key))

            selected_filter_display = st.selectbox(get_text("filter_predictions", st.session_state.lang), filter_options_display)

            if selected_filter_display == get_text("all", st.session_state.lang):
                filtered_df = df_log
            else:
                # Find the English key corresponding to the selected display name
                selected_filter_english_key = next((eng_key for eng_key, disp_name in display_name_map.items() if disp_name == selected_filter_display), None)
                if selected_filter_english_key:
                    filtered_df = df_log[df_log["result"] == selected_filter_english_key]
                else:
                    filtered_df = pd.DataFrame() # Should not happen if logic is correct

            # Translate the 'result' column in the filtered DataFrame for display
            if not filtered_df.empty:
                filtered_df_display = filtered_df.copy()
                filtered_df_display['result'] = filtered_df_display['result'].apply(lambda x: display_name_map.get(x, x))
                st.dataframe(filtered_df_display, use_container_width=True)
            else:
                st.dataframe(filtered_df, use_column_width=True)
    else:
        st.warning(get_text("prediction_failed", st.session_state.lang))
