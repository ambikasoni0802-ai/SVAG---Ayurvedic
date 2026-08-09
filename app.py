import streamlit as st
import streamlit.components.v1 as components
import json, os, subprocess, io, asyncio, base64, math, struct, wave
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}

    .svag-topbar-pill {
        background: #f4f4f5;
        border-radius: 999px;
        padding: 8px 18px;
        font-weight: 600;
        color: #444;
        font-size: 14px;
        text-align: center;
    }
    .svag-empty-logo {
        display: flex;
        justify-content: center;
        margin-top: 6vh;
        margin-bottom: 6vh;
    }
    .svag-empty-logo img {
        width: 55%;
        max-width: 280px;
    }
    div[data-testid="stButton"] button {
        border-radius: 50%;
        width: 42px;
        height: 42px;
        padding: 0;
    }

    /* ChatGPT-style chat alignment: user messages on right, SVAG on left */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        text-align: right;
        margin-left: auto;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
        background-color: #e9f5e9;
        border-radius: 16px;
        padding: 10px 14px;
        display: inline-block;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] {
        background-color: #f4f4f5;
        border-radius: 16px;
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
    if st.button("☰", key="hamburger_btn"):
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

if "welcomed" not in st.session_state:
    st.session_state.welcomed = True
    welcome_audio = text_to_speech("Welcome to SVAG. How can I help you?", "English")
    if welcome_audio:
        play_chime_then_speech(welcome_audio.read())

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
        f"You also have deep knowledge of other classical Indian texts such as the Bhagavad Gita "
        f"and similar scriptures. When asked about the Bhagavad Gita, give complete details: its "
        f"context within the Mahabharata, who is speaking to whom (Krishna to Arjuna), when it is "
        f"believed to have been composed, how many chapters (18) and verses it has, its main "
        f"teachings (dharma, karma yoga, bhakti yoga, gyan yoga, the nature of the soul/atman, "
        f"detachment from results of action, etc.), and its overall philosophical significance. "
        f"Always give a COMPLETE, THOROUGH, and DETAILED answer — never give a short, partial, or "
        f"incomplete answer. Cover every relevant aspect of the topic: definitions, types/"
        f"classifications, properties, method/process (vidhi) if applicable, benefits, uses, and "
        f"precautions where relevant. Do not leave out any important Ayurvedic detail related to "
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
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
    )
    return response.choices[0].message.content


def svag_ask_image(image_bytes, language, user_note=""):
    import base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    text_instruction = (
        f"You are SVAG, an Ayurvedic AI assistant. Look at this image — it may be an Ayurvedic "
        f"herb, plant, root, powder, or medicine. Identify it if possible, and give a COMPLETE, "
        f"DETAILED explanation in {language} language covering: its name (also give the Sanskrit/"
        f"Ayurvedic name if known), its full Ayurvedic properties (rasa, guna, virya, vipaka, "
        f"prabhav), how it is traditionally prepared/processed (vidhi) if relevant, all common "
        f"uses/benefits, which doshas it balances, dosage guidance if commonly known, and any "
        f"precautions or contraindications. Do not give a short or partial answer — cover every "
        f"relevant Ayurvedic detail. If you cannot confidently identify it, say so clearly in "
        f"{language} but still describe what you can observe. Answer only in {language}."
    )
    if user_note:
        text_instruction += f" Additional note from user: {user_note}"

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

icon_c1, icon_c2, icon_c3, icon_c4 = st.columns([1, 1, 1, 3])
with icon_c1:
    if st.button("➕", key="plus_btn", help="Photo/File bhejo"):
        st.session_state.show_upload = not st.session_state.show_upload
        st.session_state.show_voice_msg = False
        st.session_state.show_voice_assistant = False
with icon_c2:
    if st.button("🎤", key="mic_btn", help="Voice message (bol ke likho)"):
        st.session_state.show_voice_msg = not st.session_state.show_voice_msg
        st.session_state.show_upload = False
        st.session_state.show_voice_assistant = False
with icon_c3:
    if st.button("🔵", key="voice_assistant_btn", help="Voice assistant (bol ke seedha jawab)"):
        st.session_state.show_voice_assistant = not st.session_state.show_voice_assistant
        st.session_state.show_upload = False
        st.session_state.show_voice_msg = False

# --- Panel: Photo/File upload ---
if st.session_state.show_upload:
    with st.container(border=True):
        st.markdown("**📷 Jadi-buti/medicine ki photo bhejo**")
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
                            st.audio(image_audio, format="audio/mp3")
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
                    with st.chat_message("assistant", avatar=SVAG_AVATAR):
                        with st.spinner("SVAG soch raha hai..."):
                            vm_answer = svag_ask(edited_text, selected_language)
                            st.markdown(vm_answer)
                            vm_audio = text_to_speech(vm_answer, selected_language)
                            if vm_audio:
                                st.audio(vm_audio, format="audio/mp3")
                            if is_appointment_request(edited_text):
                                show_appointment_button()
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

                with st.chat_message("assistant", avatar=SVAG_AVATAR):
                    with st.spinner("SVAG soch raha hai..."):
                        voice_answer = svag_ask(spoken_text, selected_language)
                        st.markdown(voice_answer)
                        voice_audio = text_to_speech(voice_answer, selected_language)
                        if voice_audio:
                            st.audio(voice_audio, format="audio/mp3", autoplay=True)
                        if is_appointment_request(spoken_text):
                            show_appointment_button()

                st.session_state.messages.append({"role": "assistant", "content": voice_answer})

user_question = st.chat_input("Apna Ayurvedic sawaal likho...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant", avatar=SVAG_AVATAR):
        with st.spinner("SVAG soch raha hai..."):
            answer = svag_ask(user_question, selected_language)
            st.markdown(answer)
            audio = text_to_speech(answer, selected_language)
            if audio:
                st.audio(audio, format="audio/mp3")
            if is_appointment_request(user_question):
                show_appointment_button()
    st.session_state.messages.append({"role": "assistant", "content": answer})
