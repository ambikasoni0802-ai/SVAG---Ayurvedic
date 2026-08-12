import streamlit as st
import streamlit.components.v1 as components
import json, os, subprocess, io, asyncio, base64, math, struct, wave, time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from groq import Groq
import edge_tts

# Male / deep voice for each language (Microsoft Edge neural voices)
LANG_TO_MALE_VOICE = {
    "English": "en-US-GuyNeural",
    "Hindi": "hi-IN-MadhurNeural",
    "Marathi": "mr-IN-ManoharNeural",
    "Tamil": "ta-IN-ValluvarNeural",
    "Telugu": "te-IN-MohanNeural",
    "Kannada": "kn-IN-GaganNeural",
    "Malayalam": "ml-IN-MidhunNeural",
    "Punjabi": "pa-IN-OjasNeural",
    "Bengali": "bn-IN-BashkarNeural",
    "Gujarati": "gu-IN-NiranjanNeural",
    "Odia": "or-IN-SukantNeural",
    "Urdu": "ur-IN-SalmanNeural",
    "Nepali": "ne-NP-SagarNeural",
    "Sinhala": "si-LK-SameeraNeural",
    "Spanish": "es-ES-AlvaroNeural",
    "French": "fr-FR-HenriNeural",
    "German": "de-DE-ConradNeural",
    "Portuguese": "pt-BR-AntonioNeural",
    "Italian": "it-IT-DiegoNeural",
    "Russian": "ru-RU-DmitryNeural",
    "Chinese": "zh-CN-YunxiNeural",
    "Japanese": "ja-JP-KeitaNeural",
    "Korean": "ko-KR-InJoonNeural",
    "Arabic": "ar-SA-HamedNeural",
    "Indonesian": "id-ID-ArdiNeural",
    "Turkish": "tr-TR-AhmetNeural",
    "Vietnamese": "vi-VN-NamMinhNeural",
    "Thai": "th-TH-NiwatNeural",
    "Swahili": "sw-KE-RafikiNeural",
    "Dutch": "nl-NL-MaartenNeural",
}


def generate_chime():
    """Generates a small pleasant two-tone notification chime as WAV bytes."""
    sample_rate = 44100
    notes = [(880, 0.15), (1318, 0.25)]  # A5 then E6 — a simple pleasant "ding-dong"
    frames = []
    for freq, duration in notes:
        n_samples = int(sample_rate * duration)
        for i in range(n_samples):
            t = i / sample_rate
            fade = min(1.0, (n_samples - i) / (sample_rate * 0.05))
            sample = math.sin(2 * math.pi * freq * t) * 0.3 * fade
            frames.append(struct.pack("<h", int(sample * 32767)))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
    buffer.seek(0)
    return buffer.read()


def play_chime_then_speech(speech_audio_bytes):
    """Plays a chime sound, then the speech audio right after it finishes."""
    chime_bytes = generate_chime()
    chime_b64 = base64.b64encode(chime_bytes).decode()
    speech_b64 = base64.b64encode(speech_audio_bytes).decode()
    components.html(
        f"""
        <audio id="svag_chime" autoplay>
            <source src="data:audio/wav;base64,{chime_b64}" type="audio/wav">
        </audio>
        <audio id="svag_speech">
            <source src="data:audio/mp3;base64,{speech_b64}" type="audio/mp3">
        </audio>
        <script>
        const chime = document.getElementById('svag_chime');
        const speech = document.getElementById('svag_speech');
        if (chime) {{
            chime.onended = function() {{
                speech.play().catch(function(e) {{}});
            }};
        }}
        </script>
        """,
        height=0,
    )


def play_audio_hidden(audio_bytes):
    """Autoplays audio invisibly — no visible player box, just plays alongside the text."""
    audio_b64 = base64.b64encode(audio_bytes).decode()
    components.html(
        f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
        """,
        height=0,
    )


def clean_text_for_speech(text):
    """Removes markdown symbols, URLs, and formatting noise so TTS reads natural, clean speech."""
    import re
    cleaned = text

    # Remove markdown links but keep the visible label: [label](url) -> label
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    # Remove raw URLs entirely (they sound broken when read aloud)
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    # Remove markdown bold/italic/code markers
    cleaned = re.sub(r"[*_`#]+", "", cleaned)
    # Remove bullet/list markers at line start (-, •, 1., 2), etc.)
    cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", cleaned, flags=re.MULTILINE)
    # Collapse extra whitespace/newlines into natural pauses
    cleaned = re.sub(r"\n{2,}", ". ", cleaned)
    cleaned = re.sub(r"\n", ". ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def generate_mantra_audio():
    """Generates a slow, deep, dramatic recitation of the Gayatri Mantra."""
    mantra_text = (
        "ॐ भूर्भुवः स्वः तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात्॥"
    )
    try:
        async def _generate():
            audio_bytes = b""
            communicate = edge_tts.Communicate(
                mantra_text, "hi-IN-MadhurNeural", rate="-35%", pitch="-25Hz"
            )
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
            return audio_bytes

        audio_bytes = asyncio.run(_generate())
        return audio_bytes if audio_bytes else None
    except Exception:
        return None


def show_searchbar_trailer_intro():
    """Temporary (5-10s) full-screen black cinematic intro with a dramatic flickering shlok and
    a deep recitation. After it finishes, the screen fully reveals the normal app underneath."""
    mantra_display = (
        "ॐ भूर्भुवः स्वः<br>तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि<br>धियो यो नः प्रचोदयात्॥"
    )
    st.markdown(
        f"""
        <style>
        .svag-cinematic-overlay {{
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            pointer-events: none;
            z-index: 999999;
            background: #000000;
            animation: svag-overlay-lifecycle 9s ease-in-out forwards;
        }}
        .svag-cinematic-text {{
            font-family: 'Space Grotesk', serif;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: #E8C766;
            font-size: clamp(20px, 4.2vw, 34px);
            line-height: 1.9;
            padding: 0 24px;
            text-shadow:
                0 0 14px rgba(232,199,102,0.85),
                0 0 34px rgba(76,175,110,0.5);
            animation: svag-text-flicker 1.6s ease-in-out infinite;
        }}
        @keyframes svag-text-flicker {{
            0%, 100% {{ opacity: 0.55; }}
            50%      {{ opacity: 1; }}
        }}
        @keyframes svag-overlay-lifecycle {{
            0%   {{ opacity: 1; }}
            75%  {{ opacity: 1; }}
            100% {{ opacity: 0; visibility: hidden; }}
        }}
        /* Hide the native audio player used for the recitation — sound plays, box stays invisible */
        #svag-audio-anchor + div {{
            position: fixed !important;
            width: 1px !important;
            height: 1px !important;
            opacity: 0 !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }}
        </style>
        <div class="svag-cinematic-overlay">
            <div class="svag-cinematic-text">{mantra_display}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mantra_audio = generate_mantra_audio()
    if mantra_audio:
        st.markdown('<div id="svag-audio-anchor"></div>', unsafe_allow_html=True)
        st.audio(mantra_audio, format="audio/mp3", autoplay=True)


def text_to_speech(text, language):
    voice = LANG_TO_MALE_VOICE.get(language, "en-US-GuyNeural")
    clean_text = clean_text_for_speech(text)
    try:
        async def _generate():
            audio_bytes = b""
            communicate = edge_tts.Communicate(clean_text, voice, rate="+30%", pitch="-1Hz")
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]
            return audio_bytes

        audio_bytes = asyncio.run(_generate())
        if not audio_bytes:
            return None
        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.seek(0)
        return audio_buffer
    except Exception:
        return None


def speech_to_text(audio_bytes):
    transcription = client.audio.transcriptions.create(
        file=("question.wav", audio_bytes),
        model="whisper-large-v3",
    )
    return transcription.text

st.set_page_config(page_title="SVAG - Ayurvedic AI", page_icon="logo.png")

SVAG_AVATAR = "logo.png"




LANGUAGES = {
    "English": "English",
    "हिन्दी (Hindi)": "Hindi",
    "मराठी (Marathi)": "Marathi",
    "தமிழ் (Tamil)": "Tamil",
    "తెలుగు (Telugu)": "Telugu",
    "ಕನ್ನಡ (Kannada)": "Kannada",
    "മലയാളം (Malayalam)": "Malayalam",
    "ਪੰਜਾਬੀ (Punjabi)": "Punjabi",
    "বাংলা (Bengali)": "Bengali",
    "ગુજરાતી (Gujarati)": "Gujarati",
    "ଓଡ଼ିଆ (Odia)": "Odia",
    "اردو (Urdu)": "Urdu",
    "नेपाली (Nepali)": "Nepali",
    "සිංහල (Sinhala)": "Sinhala",
    "Español (Spanish)": "Spanish",
    "Français (French)": "French",
    "Deutsch (German)": "German",
    "Português (Portuguese)": "Portuguese",
    "Italiano (Italian)": "Italian",
    "Русский (Russian)": "Russian",
    "中文 (Chinese)": "Chinese",
    "日本語 (Japanese)": "Japanese",
    "한국어 (Korean)": "Korean",
    "العربية (Arabic)": "Arabic",
    "Bahasa Indonesia": "Indonesian",
    "Türkçe (Turkish)": "Turkish",
    "Tiếng Việt (Vietnamese)": "Vietnamese",
    "ไทย (Thai)": "Thai",
    "Kiswahili (Swahili)": "Swahili",
    "Nederlands (Dutch)": "Dutch",
}

if "show_settings" not in st.session_state:
    st.session_state.show_settings = False

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}

    /* Soft pastel gradient background, like the reference screenshot */
    .stApp {
        background: linear-gradient(135deg, #C9B6F0 0%, #F3C6D9 45%, #C9EAD0 100%) !important;
    }

    :root {
        --svag-forest: #0F1B14;
        --svag-leaf: #2F6F4E;
        --svag-leaf-bright: #4CAF6E;
        --svag-gold: #C9A227;
        --svag-glass: rgba(47, 111, 78, 0.08);
        --svag-glass-border: rgba(201, 162, 39, 0.35);
    }

    h1, h2, h3, .svag-topbar-pill, .svag-heading {
        font-family: 'Space Grotesk', sans-serif !important;
    }

    .svag-topbar-pill {
        background: linear-gradient(135deg, var(--svag-glass), rgba(201,162,39,0.06));
        border: 1px solid var(--svag-glass-border);
        border-radius: 999px;
        padding: 8px 20px;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: var(--svag-forest);
        font-size: 14px;
        text-align: center;
        backdrop-filter: blur(6px);
    }

    .svag-empty-logo {
        display: flex;
        justify-content: center;
        margin-top: 6vh;
        margin-bottom: 4vh;
    }
    .svag-empty-logo img {
        width: 55%;
        max-width: 280px;
        filter: drop-shadow(0 8px 24px rgba(47,111,78,0.25));
    }

    /* Signature: the "prana line" — a soft living gradient divider above the action icons */
    .svag-prana-line {
        height: 2px;
        margin: 6px 0 10px 0;
        border-radius: 999px;
        background: linear-gradient(90deg,
            transparent 0%,
            var(--svag-leaf-bright) 20%,
            var(--svag-gold) 50%,
            var(--svag-leaf-bright) 80%,
            transparent 100%);
        background-size: 200% 100%;
        animation: svag-breathe 4s ease-in-out infinite;
        opacity: 0.6;
    }
    @keyframes svag-breathe {
        0%   { background-position: 0% 50%; opacity: 0.35; }
        50%  { background-position: 100% 50%; opacity: 0.75; }
        100% { background-position: 0% 50%; opacity: 0.35; }
    }

    /* Icon buttons — glass circles with a gold-green gradient ring */
    div[data-testid="stButton"] button {
        border-radius: 50%;
        width: 44px;
        height: 44px;
        padding: 0;
        border: 1.5px solid var(--svag-glass-border);
        background: linear-gradient(135deg, rgba(47,111,78,0.06), rgba(201,162,39,0.05));
        backdrop-filter: blur(6px);
        transition: all 0.25s ease;
        box-shadow: 0 1px 3px rgba(15,27,20,0.06);
    }
    div[data-testid="stButton"] button:hover {
        border-color: var(--svag-leaf-bright);
        box-shadow: 0 0 0 4px rgba(76,175,110,0.12), 0 2px 8px rgba(15,27,20,0.12);
        transform: translateY(-1px);
    }
    div[data-testid="stButton"] button [data-testid="stIconMaterial"] {
        color: var(--svag-leaf);
        font-size: 20px;
    }
    div[data-testid="stButton"] button:hover [data-testid="stIconMaterial"] {
        color: var(--svag-gold);
    }

    /* Custom uploaded icon images for specific action buttons (transparent bg) */
    #svag-anchor-plus_btn + div[data-testid="stButton"] button [data-testid="stIconMaterial"] {
        display: none;
    }
    #svag-anchor-plus_btn + div[data-testid="stButton"] button {
        background-image: url("data:image/png;base64,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");
        background-size: 55%;
        background-repeat: no-repeat;
        background-position: center;
    }
    #svag-anchor-mic_btn + div[data-testid="stButton"] button [data-testid="stIconMaterial"] {
        display: none;
    }
    #svag-anchor-mic_btn + div[data-testid="stButton"] button {
        background-image: url("data:image/png;base64,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");
        background-size: 55%;
        background-repeat: no-repeat;
        background-position: center;
    }
    #svag-anchor-voice_assistant_btn + div[data-testid="stButton"] button [data-testid="stIconMaterial"] {
        display: none;
    }
    #svag-anchor-voice_assistant_btn + div[data-testid="stButton"] button {
        background-image: url("data:image/png;base64,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");
        background-size: 55%;
        background-repeat: no-repeat;
        background-position: center;
    }
    #svag-anchor-imagegen_btn + div[data-testid="stButton"] button [data-testid="stIconMaterial"] {
        display: none;
    }
    #svag-anchor-imagegen_btn + div[data-testid="stButton"] button {
        background-image: url("data:image/png;base64,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");
        background-size: 55%;
        background-repeat: no-repeat;
        background-position: center;
    }
    div[data-testid="stButton"] button:active {
        transform: translateY(0px) scale(0.96);
    }

    /* Chat input — pill shape with a soft botanical glow border */
    div[data-testid="stChatInput"] {
        border-radius: 999px !important;
        border: 1.5px solid var(--svag-glass-border) !important;
        background: linear-gradient(135deg, rgba(47,111,78,0.03), rgba(201,162,39,0.03)) !important;
        box-shadow: 0 2px 12px rgba(15,27,20,0.06) !important;
        transition: box-shadow 0.25s ease, border-color 0.25s ease;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: var(--svag-leaf-bright) !important;
        box-shadow: 0 0 0 4px rgba(76,175,110,0.14), 0 2px 12px rgba(15,27,20,0.08) !important;
    }

    /* ChatGPT-style chat alignment: user messages on right, SVAG on left */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        text-align: right;
        margin-left: auto;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, rgba(76,175,110,0.14), rgba(201,162,39,0.08));
        border: 1px solid rgba(76,175,110,0.18);
        border-radius: 18px;
        padding: 10px 14px;
        display: inline-block;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] {
        background: rgba(15,27,20,0.04);
        border: 1px solid rgba(15,27,20,0.06);
        border-radius: 18px;
        padding: 10px 14px;
        display: inline-block;
        max-width: 80%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "selected_language_persist" not in st.session_state:
    st.session_state.selected_language_persist = "English"

LABEL_BY_LANGUAGE = {v: k for k, v in LANGUAGES.items()}

top_c1, top_c2, top_c3 = st.columns([1, 3, 1])
with top_c1:
    if st.button("", key="hamburger_btn", icon=":material/menu:"):
        st.session_state.show_settings = not st.session_state.show_settings
with top_c2:
    st.markdown('<div class="svag-topbar-pill">🌿 SVAG</div>', unsafe_allow_html=True)

if st.session_state.show_settings:
    with st.container(border=True):
        st.markdown("**⚙️ Settings**")
        current_label = LABEL_BY_LANGUAGE.get(st.session_state.selected_language_persist, "English")
        current_index = list(LANGUAGES.keys()).index(current_label)
        selected_label = st.selectbox(
            "Jawab ki bhasha / Answer language",
            list(LANGUAGES.keys()),
            index=current_index,
            key="language_selectbox",
        )
        st.session_state.selected_language_persist = LANGUAGES[selected_label]

selected_language = st.session_state.selected_language_persist

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    import base64
    with open("logo.png", "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<div class="svag-empty-logo"><img src="data:image/png;base64,{logo_b64}"></div>',
        unsafe_allow_html=True,
    )

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


def ensure_data():
    if not os.path.exists("Datasets/Ayurveda"):
        subprocess.run(["git", "clone", "https://github.com/gita/Datasets.git"], check=False)
        subprocess.run(
            ["rm", "-rf", "Datasets/chanakya", "Datasets/srimad-bhagavatam",
             "Datasets/Vectorise_Script", "Datasets/README.md"],
            check=False,
        )
    if not os.path.exists("herb-database"):
        subprocess.run(
            ["git", "clone", "https://github.com/sciencewithsaucee-sudo/herb-database.git"],
            check=False,
        )


@st.cache_resource(show_spinner="SVAG ka brain taiyar ho raha hai... (pehli baar 3-5 min lagenge)")
def load_svag():
    ensure_data()

    def read_json_safe(path):
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            return None

    all_texts = []

    json_files = [
        os.path.join(r, f)
        for r, d, files in os.walk("Datasets/Ayurveda")
        for f in files if f.endswith(".json")
    ]
    for path in json_files:
        data = read_json_safe(path)
        if isinstance(data, list):
            all_texts += [item["text"] for item in data if isinstance(item, dict) and "text" in item]

    herb_data = read_json_safe("herb-database/herb.json")
    if isinstance(herb_data, list):
        all_texts += [json.dumps(item) for item in herb_data if isinstance(item, dict)]

    if not all_texts:
        st.error("Data nahi mila — GitHub se Ayurvedic data fetch nahi ho paya. Repo/network check karein.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=t) for t in all_texts]
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory="svag_db")
    return vectordb


vectordb = load_svag()
client = Groq(api_key=GROQ_API_KEY)

APPOINTMENT_LINK = "https://swamivivekanandayurvedclinic.netlify.app/"
APPOINTMENT_KEYWORDS = [
    "appointment", "book", "booking", "consult", "consultation",
    "मुलाकात", "अपॉइंटमेंट", "बुक", "मिलना", "डॉक्टर से मिलना",
    "भेट", "अपॉईंटमेंट",
]


def is_appointment_request(text):
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in APPOINTMENT_KEYWORDS)


def show_appointment_button():
    st.link_button("📅 Appointment Book Karein", APPOINTMENT_LINK)


WHATSAPP_NUMBER = "919226473457"


def show_whatsapp_button():
    import urllib.parse
    message_text = (
        "Namaste, mujhe Ayurvedic consultation ke baare mein jaankari chahiye.\n\n"
        "Clinic Locations:\n"
        "https://maps.app.goo.gl/aiDnAkxAfK9psQXc6\n"
        "https://maps.app.goo.gl/etuEGgQzwMxsKc8s9"
    )
    message = urllib.parse.quote(message_text)
    whatsapp_url = f"https://wa.me/{WHATSAPP_NUMBER}?text={message}"
    st.link_button("💬 WhatsApp par Sampark Karein", whatsapp_url)


DINCHARYA_TIP = (
    "🌅 **Ayurvedic Dincharya (Daily Routine):** Brahma muhurta mein (sunrise se pehle) uthein, "
    "khali pet paani piyein, oil pulling/daant saaf karein, yoga-pranayam-dhyan karein, snan karein, "
    "samay par saatvik bhojan karein (dhire-dhire, bina jaldi ke), din mein halki dhoop lein, raat ka "
    "khana halka aur jaldi (sunset ke 2-3 ghante andar) karein, aur samay par so jayein (10-11 baje "
    "tak). Yeh santulit Dincharya teeno doshon (Vata, Pitta, Kapha) ko sadhne mein madad karti hai."
)

DOCTOR_INFO_TEXT = (
    "👨‍⚕️ **SVAG Group Doctor:** Dr. Ajit Kadam — sabhi Ayurvedic bimariyon ka ilaj karte hain, "
    "khaaskar Arthritis (jodo ka dard) mein unhe 29 saal ka experience hai."
)

CREATOR_INFO_TEXT = "🌿 **SVAG** — Made by Veenu, SVAG group."


def show_info_bundle():
    with st.container(border=True):
        st.markdown(DINCHARYA_TIP)
        st.markdown(DOCTOR_INFO_TEXT)
        show_appointment_button()
        show_whatsapp_button()
        st.markdown(CREATOR_INFO_TEXT)


VIDEO_KEYWORDS = [
    "video", "वीडियो", "video bhejo", "video dikhao", "vidio",
]


def is_video_request(text):
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in VIDEO_KEYWORDS)


def show_video_button(topic_text):
    import urllib.parse
    query = urllib.parse.quote(f"{topic_text} ayurvedic")
    youtube_url = f"https://www.youtube.com/results?search_query={query}"
    st.link_button("🎥 Video Dekhein (YouTube)", youtube_url)


def is_ayurvedic_herb_request(topic_text):
    """Uses the LLM to strictly check if the requested topic is a genuine Ayurvedic herb/plant."""
    check_prompt = (
        f"Is the following text the name of a genuine Ayurvedic medicinal herb, plant, root, or "
        f"botanical substance used in Ayurveda (e.g. Ashwagandha, Tulsi, Neem, Brahmi, Turmeric, "
        f"Triphala, etc.)? Answer with ONLY one word: YES or NO. Text: \"{topic_text}\""
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": check_prompt}],
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer.startswith("YES")
    except Exception:
        return False


def fetch_real_herb_image(herb_name):
    """Fetches a real photo of the herb from Wikipedia/Wikimedia Commons (free, no API key) — not AI-generated."""
    import requests

    headers = {"User-Agent": "SVAG-Ayurvedic-App/1.0 (contact: svag-app@example.com)"}

    def try_wikipedia_pageimage(query):
        try:
            search_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1},
                headers=headers, timeout=20,
            )
            search_results = search_resp.json().get("query", {}).get("search", [])
            if not search_results:
                return None
            page_title = search_results[0]["title"]

            img_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "titles": page_title, "prop": "pageimages",
                    "format": "json", "pithumbsize": 800,
                },
                headers=headers, timeout=20,
            )
            pages = img_resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                thumb = page.get("thumbnail", {})
                if thumb.get("source"):
                    return thumb["source"], page_title
            return None
        except Exception:
            return None

    def try_commons_search(query):
        try:
            resp = requests.get(
                "https://commons.wikimedia.org/w/api.php",
                params={
                    "action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
                    "gsrlimit": 1, "prop": "imageinfo", "iiprop": "url", "format": "json",
                },
                headers=headers, timeout=20,
            )
            pages = resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                imageinfo = page.get("imageinfo", [])
                if imageinfo and imageinfo[0].get("url"):
                    return imageinfo[0]["url"], page.get("title", query)
            return None
        except Exception:
            return None

    def try_openverse(query):
        try:
            resp = requests.get(
                "https://api.openverse.org/v1/images/",
                params={"q": f"{query} plant", "license_type": "all", "page_size": 1},
                headers=headers, timeout=20,
            )
            results = resp.json().get("results", [])
            if results and results[0].get("url"):
                return results[0]["url"], results[0].get("title", query)
            return None
        except Exception:
            return None

    result = try_wikipedia_pageimage(f"{herb_name} plant") or try_wikipedia_pageimage(herb_name)
    if not result:
        result = try_commons_search(f"{herb_name} ayurvedic herb") or try_commons_search(herb_name)
    if not result:
        result = try_openverse(herb_name)
    if not result:
        return None

    image_url, source_title = result
    try:
        img_resp = requests.get(image_url, headers=headers, timeout=30)
        if img_resp.status_code == 200 and len(img_resp.content) > 1000:
            return img_resp.content, source_title
        return None
    except Exception:
        return None


def fetch_real_image(topic):
    """Fetches a REAL photo (not AI-generated) of ANY topic the user names, from Wikipedia/
    Wikimedia Commons/Openverse (free, no API key). Works for herbs, objects, places, animals,
    people, monuments — jo bhi bola jaye."""
    import requests

    headers = {"User-Agent": "SVAG-Ayurvedic-App/1.0 (contact: svag-app@example.com)"}

    def try_wikipedia_pageimage(query):
        try:
            search_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1},
                headers=headers, timeout=20,
            )
            search_results = search_resp.json().get("query", {}).get("search", [])
            if not search_results:
                return None
            page_title = search_results[0]["title"]

            img_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "titles": page_title, "prop": "pageimages",
                    "format": "json", "pithumbsize": 800,
                },
                headers=headers, timeout=20,
            )
            pages = img_resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                thumb = page.get("thumbnail", {})
                if thumb.get("source"):
                    return thumb["source"], page_title
            return None
        except Exception:
            return None

    def try_commons_search(query):
        try:
            resp = requests.get(
                "https://commons.wikimedia.org/w/api.php",
                params={
                    "action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
                    "gsrlimit": 1, "prop": "imageinfo", "iiprop": "url", "format": "json",
                },
                headers=headers, timeout=20,
            )
            pages = resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                imageinfo = page.get("imageinfo", [])
                if imageinfo and imageinfo[0].get("url"):
                    return imageinfo[0]["url"], page.get("title", query)
            return None
        except Exception:
            return None

    def try_openverse(query):
        try:
            resp = requests.get(
                "https://api.openverse.org/v1/images/",
                params={"q": query, "license_type": "all", "page_size": 1},
                headers=headers, timeout=20,
            )
            results = resp.json().get("results", [])
            if results and results[0].get("url"):
                return results[0]["url"], results[0].get("title", query)
            return None
        except Exception:
            return None

    result = (
        try_wikipedia_pageimage(topic)
        or try_commons_search(topic)
        or try_openverse(topic)
    )
    if not result:
        return None

    image_url, source_title = result
    try:
        img_resp = requests.get(image_url, headers=headers, timeout=30)
        if img_resp.status_code == 200 and len(img_resp.content) > 1000:
            return img_resp.content, source_title
        return None
    except Exception:
        return None


IMAGE_REQUEST_KEYWORDS = [
    "image", "photo", "picture", "chitra", "तस्वीर", "फोटो", "इमेज", "चित्र", "pic",
]
IMAGE_ACTION_KEYWORDS = [
    "banao", "banake", "banaiye", "generate", "do", "dikhao", "chahiye", "bhejo",
    "laao", "lao", "dhundo", "dhundho", "batao", "बनाओ", "दो", "दिखाओ", "लाओ",
]


def is_image_generation_request(text):
    text_lower = text.lower()
    has_image_word = any(k.lower() in text_lower for k in IMAGE_REQUEST_KEYWORDS)
    has_action_word = any(k.lower() in text_lower for k in IMAGE_ACTION_KEYWORDS)
    return has_image_word and has_action_word


def extract_herb_name(text):
    """Uses the LLM to extract a genuine Ayurvedic herb name from a request, or return NONE."""
    extract_prompt = (
        f"The user wrote this message asking for an image: \"{text}\". "
        f"If this message mentions the name of a genuine Ayurvedic medicinal herb, plant, or root "
        f"(e.g. Ashwagandha, Manjistha, Tulsi, Neem, Brahmi, Triphala, etc.), reply with ONLY that "
        f"herb's name (in English/Roman letters). If no genuine Ayurvedic herb is mentioned, reply "
        f"with ONLY the word NONE."
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": extract_prompt}],
            max_tokens=15,
        )
        result = response.choices[0].message.content.strip()
        if result.upper() == "NONE" or len(result) > 40:
            return None
        return result
    except Exception:
        return None


def extract_image_topic(text):
    """Uses the LLM to extract WHATEVER subject/thing the user wants a real photo of — herb,
    object, animal, place, monument, anything — or return NONE if nothing identifiable."""
    extract_prompt = (
        f"The user wrote this message asking for a photo/image: \"{text}\". "
        f"Extract the name of the specific thing they want a real photo of — it could be a herb/"
        f"plant, an animal, an object, a place, a monument, food, or anything else. Reply with "
        f"ONLY that subject's name (in English/Roman letters), nothing else. If you genuinely "
        f"cannot identify what subject they want a photo of, reply with ONLY the word NONE."
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": extract_prompt}],
            max_tokens=20,
        )
        result = response.choices[0].message.content.strip()
        if result.upper() == "NONE" or len(result) > 60:
            return None
        return result
    except Exception:
        return None


def generate_ai_image(prompt):
    """Khud se (AI se) image generate karta hai — kahin se dhoondh kar nahi laata."""
    import requests, urllib.parse, random
    try:
        encoded_prompt = urllib.parse.quote(prompt)
        seed = random.randint(1, 999999)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&seed={seed}"
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200 and len(resp.content) > 1000:
            return resp.content
        return None
    except Exception:
        return None


def handle_herb_image_flow(user_text, language):
    """Full flow: pehle Ayurvedic herb ke liye try karo (real photo + growing/origin details).
    Agar herb nahi mila, to jo bhi cheez bola gaya uski real photo laao (Wikipedia/Commons/
    Openverse se — AI-generated nahi, asli photo)."""
    herb_name = extract_herb_name(user_text)
    if herb_name:
        with st.spinner(f"'{herb_name}' ki real photo dhoondi ja rahi hai..."):
            img_bytes = generate_ai_image(f"{herb_name}, ayurvedic medicinal herb/plant, real photograph style")
        if not img_bytes:
            st.error("Is jadi-buti ki photo abhi generate nahi ho payi, dobara try karein ya naam sahi se likhein.")
            return True
        with st.chat_message("assistant", avatar=SVAG_AVATAR):
            st.image(img_bytes, caption=f"{herb_name} (AI se generate ki gayi image)", width=400)
            with st.spinner("Jaankari taiyar ki ja rahi hai..."):
                growth_question = (
                    f"{herb_name} kahan paya jata hai, kaise ugta hai (mitti, jalvayu, region), "
                    f"aur iski kheti kaise hoti hai — poori detail dein."
                )
                growth_info = svag_ask(growth_question, language)
            st.markdown(growth_info)
            growth_audio = text_to_speech(growth_info, language)
            if growth_audio:
                play_audio_hidden(growth_audio.read())
        show_info_bundle()
        st.session_state.messages.append({"role": "assistant", "content": f"[{herb_name} ki real photo] {growth_info}"})
        return True

    # Not an Ayurvedic herb — try to fetch a real photo of whatever was named instead.
    topic = extract_image_topic(user_text)
    if not topic:
        st.warning("Kripya us cheez ka naam batayein jiski real photo chahiye (jaise Ashwagandha, Taj Mahal, Sher).")
        return True
    with st.spinner(f"'{topic}' ki real photo dhoondi ja rahi hai..."):
        img_bytes = generate_ai_image(topic)
    if not img_bytes:
        st.error(f"'{topic}' ki photo abhi generate nahi ho payi, dobara try karein ya naam sahi se likhein.")
        return True
    with st.chat_message("assistant", avatar=SVAG_AVATAR):
        st.image(img_bytes, caption=f"{topic} (AI se generate ki gayi image)", width=400)
        with st.spinner("Jaankari taiyar ki ja rahi hai..."):
            info_question = f"{topic} ke baare mein sankshep mein batayein."
            topic_info = svag_ask(info_question, language)
        st.markdown(topic_info)
        topic_audio = text_to_speech(topic_info, language)
        if topic_audio:
            play_audio_hidden(topic_audio.read())
    show_info_bundle()
    st.session_state.messages.append({"role": "assistant", "content": f"[{topic} ki real photo] {topic_info}"})
    return True


def get_friendly_error_message(language):
    return (
        "SVAG abhi bahut zyada demand mein hai aur iski usage limit filhal ke liye poori ho gayi hai. "
        "Kripya ek minute ruk kar dobara try karein — thodi der mein ye phir se kaam karega.\n\n"
        "(SVAG is currently experiencing high demand and has reached its usage limit for the moment. "
        "Please wait about a minute and try again — it will be back shortly.)"
    )


def svag_ask(question, language):
    results = vectordb.similarity_search(question, k=10)
    context = "\n\n".join([r.page_content for r in results])
    prompt = (
        f"You are SVAG, an Ayurvedic AI assistant with deep, complete expertise in ALL areas of "
        f"Ayurveda — doshas (Vata, Pitta, Kapha), herbs and their properties (rasa, guna, virya, "
        f"vipaka, prabhav), diseases and their Ayurvedic treatment, methods of preparation (vidhi) "
        f"of Ayurvedic medicines/formulations, Panchakarma, Rasayana, diet and lifestyle (aahar-"
        f"vihar), and every other branch of Ayurveda. "
        f"You also have deep, complete knowledge of ALL classical Ayurvedic granths (texts), "
        f"including but not limited to: Charak Samhita, Sushruta Samhita (including its pioneering "
        f"surgical techniques such as rhinoplasty/plastic surgery), Ashtanga Hridaya, Ashtanga "
        f"Sangraha, Bhaishajya Ratnavali, Sharangdhar Samhita, Madhava Nidana, and any other Rishi-"
        f"authored Ayurvedic text the user asks about. When asked about any granth, always give "
        f"complete details: who wrote/compiled it, approximately when it was written (era/century), "
        f"what topics/branches of medicine it covers, its structure (sthanas/chapters if known), and "
        f"its major/notable contributions or achievements to Ayurveda and medicine. "
        f"You also have deep, complete knowledge of ALL of Sanatan Dharma — every major scripture, "
        f"text, and body of knowledge, including but not limited to: the four Vedas (Rigveda, "
        f"Yajurveda, Samaveda, Atharvaveda) and their Samhitas/Brahmanas/Aranyakas/Upanishads, the "
        f"principal Upanishads (Isha, Kena, Katha, Mundaka, Mandukya, Chandogya, Brihadaranyaka, "
        f"etc.), the Itihasas (Ramayana, Mahabharata including the Bhagavad Gita), the 18 Puranas "
        f"(Vishnu, Shiva, Bhagavata, Markandeya, etc.) and Upapuranas, the Dharmashastras (Manusmriti "
        f"and others), the six Darshanas/schools of philosophy (Nyaya, Vaisheshika, Samkhya, Yoga, "
        f"Mimamsa, Vedanta), Vedanta sub-schools (Advaita, Vishishtadvaita, Dvaita), the Agamas and "
        f"Tantras, Smriti literature, and any other Sanatan Dharma text, concept, deity, ritual, "
        f"festival, or philosophical idea the user asks about. When asked about any scripture/text, "
        f"give complete details: who authored/compiled it (or its traditional attribution), "
        f"approximately when it was composed (era), its structure (chapters/sections/verses), its "
        f"core teachings and content, and its philosophical or spiritual significance. For the "
        f"Bhagavad Gita specifically, cover its context within the Mahabharata, Krishna speaking to "
        f"Arjuna, its 18 chapters, and its main teachings (dharma, karma yoga, bhakti yoga, gyan "
        f"yoga, the nature of the soul/atman, detachment from results of action). "
        f"Always give a COMPLETE, THOROUGH, and DETAILED answer — never give a short, partial, or "
        f"incomplete answer. Cover every relevant aspect of the topic: definitions, types/"
        f"classifications, properties, method/process (vidhi) if applicable, benefits, uses, and "
        f"precautions where relevant. Do not leave out any important detail related to "
        f"the question. Use bullet points or numbered lists where it helps organize a detailed "
        f"answer. "
        f"Always answer in {language} language only, regardless of what language the question is in. "
        f"Do not mix in words or phrases from any other language mid-answer — the entire answer "
        f"must stay consistently in {language} from start to end, including any technical or "
        f"Ayurvedic terms (transliterate them into {language} script/pronunciation rather than "
        f"switching scripts). "
        f"If the user asks who made you, who created you, who your developer is, or any similar "
        f"question about your origin/creator, always answer that you were made by Veenu from SVAG "
        f"group (say this in {language}) — do not mention any AI company, model provider, or "
        f"technology behind you. "
        f"If the user asks about a doctor, or asks for a doctor recommendation/consultation, always "
        f"answer (in {language}) with this information: SVAG group ke doctor ka naam Dr. Ajit Kadam "
        f"hai. Woh sabhi Ayurvedic bimariyon ka treatment karte hain, lekin unki sabse best "
        f"specialization Arthritis (jodo ka dard) ka treatment hai, jisme unko 29 saal ka experience "
        f"hai. Woh arthritis se mukti dilane mein madad karte hain, lekin isme patient ko bhi apni "
        f"taraf se prayas (consistency/discipline) karna padta hai, tabhi poori tarah theek hota hai. "
        f"Adhik jaankari ke liye is link par jayein: https://swamivivekanandayurvedclinic.netlify.app/ "
        f"(mention this link clearly and tell them to type/visit this link for more information). "
        f"If the user says they want to book an appointment, mil na hai, consultation chahiye, or "
        f"anything similar, tell them (in {language}) that you are showing them an appointment "
        f"booking button/link below — mention that a button has appeared for them to tap. "
        f"Use the Ayurvedic context below, plus your own broad Ayurvedic knowledge, to give the "
        f"fullest possible answer. If the specific answer is not in the context, still answer from "
        f"your general Ayurvedic knowledge rather than saying you don't know — only say you don't "
        f"know if the topic is truly unrelated to Ayurveda.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nDetailed Answer (in {language}):"
    )
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return get_friendly_error_message(language)


def svag_ask_image(image_bytes, language, user_note=""):
    import base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    text_instruction = (
        f"You are SVAG, an Ayurvedic AI assistant. Look at this image. It could be TWO kinds of "
        f"things — figure out which one it is and respond accordingly, in {language} language:\n\n"
        f"CASE 1 — If it is an Ayurvedic herb, plant, root, powder, or medicine: identify it if "
        f"possible, and give a COMPLETE, DETAILED explanation covering its name (also give the "
        f"Sanskrit/Ayurvedic name if known), its full Ayurvedic properties (rasa, guna, virya, "
        f"vipaka, prabhav), how it is traditionally prepared/processed (vidhi) if relevant, all "
        f"common uses/benefits, which doshas it balances, dosage guidance if commonly known, and "
        f"any precautions or contraindications.\n\n"
        f"CASE 2 — If it is a photo of a body part, skin condition, swelling, joint, wound, rash, "
        f"or any visible physical ailment/symptom: describe what you can observe in the image, "
        f"explain what Ayurvedic understanding/perspective applies to this kind of condition "
        f"(possible dosha imbalance involved), give a COMPLETE and DETAILED general Ayurvedic "
        f"treatment approach (relevant herbs, Panchakarma therapies, diet/lifestyle changes, home "
        f"remedies commonly used in Ayurveda for this kind of condition). This applies to ANY "
        f"condition, not just one specific disease — analyze whatever is shown. Be clear that this "
        f"is general Ayurvedic guidance based on a photo, not a confirmed medical diagnosis, and "
        f"recommend an in-person consultation for a confirmed diagnosis and personalized treatment. "
        f"If the condition looks like it could be arthritis/joint pain, specifically mention that "
        f"SVAG group ke Dr. Ajit Kadam ko arthritis treatment mein 29 saal ka experience hai aur "
        f"unse consult karne ki salah dein, adhik jaankari ke liye is link ka zikr karein: "
        f"https://swamivivekanandayurvedclinic.netlify.app/\n\n"
        f"Do not give a short or partial answer — cover every relevant detail for whichever case "
        f"applies. If you cannot confidently identify what's in the image, say so clearly in "
        f"{language} but still describe what you can observe. Answer only in {language}."
    )
    if user_note:
        text_instruction += f" Additional note from user: {user_note}"

    try:
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    ],
                }
            ],
            max_tokens=1500,
        )
        return response.choices[0].message.content
    except Exception as e:
        return get_friendly_error_message(language)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = SVAG_AVATAR if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if "show_upload" not in st.session_state:
    st.session_state.show_upload = False
if "show_voice_msg" not in st.session_state:
    st.session_state.show_voice_msg = False
if "show_voice_assistant" not in st.session_state:
    st.session_state.show_voice_assistant = False
if "show_image_gen" not in st.session_state:
    st.session_state.show_image_gen = False

st.markdown('<div class="svag-prana-line"></div>', unsafe_allow_html=True)
icon_c1, icon_c2, icon_c3, icon_c4, icon_c5 = st.columns([1, 1, 1, 1, 2])
with icon_c1:
    st.markdown('<div id="svag-anchor-plus_btn"></div>', unsafe_allow_html=True)
    if st.button("", key="plus_btn", icon=":material/attach_file:", help="Photo/File bhejo"):
        st.session_state.show_upload = not st.session_state.show_upload
        st.session_state.show_voice_msg = False
        st.session_state.show_voice_assistant = False
        st.session_state.show_image_gen = False
with icon_c2:
    st.markdown('<div id="svag-anchor-mic_btn"></div>', unsafe_allow_html=True)
    if st.button("", key="mic_btn", icon=":material/mic:", help="Voice message (bol ke likho)"):
        st.session_state.show_voice_msg = not st.session_state.show_voice_msg
        st.session_state.show_upload = False
        st.session_state.show_voice_assistant = False
        st.session_state.show_image_gen = False
with icon_c3:
    st.markdown('<div id="svag-anchor-voice_assistant_btn"></div>', unsafe_allow_html=True)
    if st.button("", key="voice_assistant_btn", icon=":material/graphic_eq:", help="Voice assistant (bol ke seedha jawab)"):
        st.session_state.show_voice_assistant = not st.session_state.show_voice_assistant
        st.session_state.show_upload = False
        st.session_state.show_voice_msg = False
        st.session_state.show_image_gen = False
with icon_c4:
    st.markdown('<div id="svag-anchor-imagegen_btn"></div>', unsafe_allow_html=True)
    if st.button("", key="imagegen_btn", icon=":material/image:", help="Real photo laao (kisi bhi cheez ki)"):
        st.session_state.show_image_gen = not st.session_state.show_image_gen
        st.session_state.show_upload = False
        st.session_state.show_voice_msg = False
        st.session_state.show_voice_assistant = False

# --- Panel: Real photo laao (kisi bhi cheez ki — herb ho ya kuch aur) ---
if st.session_state.show_image_gen:
    with st.container(border=True):
        st.markdown("**🖼️ Image generate karo**")
        st.caption("Jo bhi naam likhoge uski image AI khud se generate karega — kahin se dhoondh kar nahi laayega.")
        herb_topic = st.text_input("Kis cheez ki photo chahiye? (jaise: Ashwagandha, Taj Mahal, Sher)", key="herb_topic_input")
        if st.button("Photo Laao", key="fetch_herb_image_btn"):
            if herb_topic.strip():
                with st.spinner(f"'{herb_topic}' ki image generate ki ja rahi hai..."):
                    img_bytes = generate_ai_image(herb_topic)
                if img_bytes:
                    st.session_state.messages.append({"role": "user", "content": f"[Photo laao: {herb_topic}]"})
                    with st.chat_message("user"):
                        st.markdown(f"Photo laao: {herb_topic}")
                    with st.chat_message("assistant", avatar=SVAG_AVATAR):
                        st.image(img_bytes, caption=f"{herb_topic} (AI se generate ki gayi image)", width=400)
                    st.session_state.messages.append({"role": "assistant", "content": f"[{herb_topic} ki AI se generate ki gayi image]"})
                    st.session_state.show_image_gen = False
                else:
                    st.error(f"'{herb_topic}' ki image abhi generate nahi ho payi, dobara try karein ya naam sahi se likhein.")

# --- Panel: Photo/File upload ---
if st.session_state.show_upload:
    with st.container(border=True):
        st.markdown("**📷 Photo bhejo — jadi-buti/dawai PHOTO ya apni takleef/body part ki photo**")
        uploaded_image = st.file_uploader("Photo upload karo (jpg/png)", type=["jpg", "jpeg", "png"])
        image_note = st.text_input("Photo ke baare me kuch batana ho to likho (optional)")

        if uploaded_image is not None:
            st.image(uploaded_image, width=250)
            if st.button("Is image ke baare me batao"):
                image_bytes = uploaded_image.getvalue()
                st.session_state.messages.append({"role": "user", "content": "[Image bheji gayi]" + (f" — {image_note}" if image_note else "")})
                with st.chat_message("user"):
                    st.image(uploaded_image, width=200)
                    if image_note:
                        st.markdown(image_note)
                with st.chat_message("assistant", avatar=SVAG_AVATAR):
                    with st.spinner("SVAG image dekh raha hai..."):
                        image_answer = svag_ask_image(image_bytes, selected_language, image_note)
                        st.markdown(image_answer)
                        image_audio = text_to_speech(image_answer, selected_language)
                        if image_audio:
                            play_audio_hidden(image_audio.read())
                show_info_bundle()
                st.session_state.messages.append({"role": "assistant", "content": image_answer})
                st.session_state.show_upload = False

# --- Panel: Voice message (bol ke text banao, edit karke bhejo) ---
if st.session_state.show_voice_msg:
    with st.container(border=True):
        st.markdown("**🎤 Voice message**")
        vm_input = st.audio_input("Yahan tap karke bolo", key="vm_audio")
        if vm_input is not None:
            if st.button("Text me convert karo"):
                with st.spinner("Awaaz samjhi ja rahi hai..."):
                    st.session_state.vm_transcribed = speech_to_text(vm_input.getvalue())
            if "vm_transcribed" in st.session_state:
                edited_text = st.text_area("Check/edit karo, phir bhejo:", value=st.session_state.vm_transcribed)
                if st.button("Bhejo"):
                    st.session_state.messages.append({"role": "user", "content": edited_text})
                    with st.chat_message("user"):
                        st.markdown(edited_text)

                    if is_image_generation_request(edited_text):
                        handle_herb_image_flow(edited_text, selected_language)
                    else:
                        with st.chat_message("assistant", avatar=SVAG_AVATAR):
                            with st.spinner("SVAG soch raha hai..."):
                                vm_answer = svag_ask(edited_text, selected_language)
                                st.markdown(vm_answer)
                                vm_audio = text_to_speech(vm_answer, selected_language)
                                if vm_audio:
                                    play_audio_hidden(vm_audio.read())
                                if is_video_request(edited_text):
                                    show_video_button(edited_text)
                        show_info_bundle()
                        st.session_state.messages.append({"role": "assistant", "content": vm_answer})
                    st.session_state.show_voice_msg = False
                    del st.session_state.vm_transcribed

# --- Panel: Voice assistant (bol ke seedha jawab, audio ke saath) ---
if st.session_state.show_voice_assistant:
    with st.container(border=True):
        st.markdown("**🔵 Voice assistant — bol ke seedha sawaal poocho**")
        voice_input = st.audio_input("Yahan tap karke bolo", key="va_audio")
        if voice_input is not None:
            if st.button("Ye sawaal SVAG ko bhejo"):
                with st.spinner("Awaaz samjhi ja rahi hai..."):
                    spoken_text = speech_to_text(voice_input.getvalue())

                st.session_state.messages.append({"role": "user", "content": spoken_text})
                with st.chat_message("user"):
                    st.markdown(spoken_text)

                if is_image_generation_request(spoken_text):
                    handle_herb_image_flow(spoken_text, selected_language)
                else:
                    with st.chat_message("assistant", avatar=SVAG_AVATAR):
                        with st.spinner("SVAG soch raha hai..."):
                            voice_answer = svag_ask(spoken_text, selected_language)
                            st.markdown(voice_answer)
                            voice_audio = text_to_speech(voice_answer, selected_language)
                            if voice_audio:
                                play_audio_hidden(voice_audio.read())
                            if is_video_request(spoken_text):
                                show_video_button(spoken_text)
                    show_info_bundle()

                    st.session_state.messages.append({"role": "assistant", "content": voice_answer})

if "trailer_shown" not in st.session_state:
    st.session_state.trailer_shown = True
    show_searchbar_trailer_intro()

user_question = st.chat_input("Apna Ayurvedic sawaal likho...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    if is_image_generation_request(user_question):
        handle_herb_image_flow(user_question, selected_language)
    else:
        with st.chat_message("assistant", avatar=SVAG_AVATAR):
            with st.spinner("SVAG soch raha hai..."):
                answer = svag_ask(user_question, selected_language)
                st.markdown(answer)
                audio = text_to_speech(answer, selected_language)
                if audio:
                    play_audio_hidden(audio.read())
                if is_video_request(user_question):
                    show_video_button(user_question)
        show_info_bundle()
        st.session_state.messages.append({"role": "assistant", "content": answer})
