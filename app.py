import streamlit as st
import json, os, subprocess, io, asyncio
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


def text_to_speech(text, language):
    voice = LANG_TO_MALE_VOICE.get(language, "en-US-GuyNeural")
    try:
        async def _generate():
            audio_bytes = b""
            communicate = edge_tts.Communicate(text, voice, rate="+10%", pitch="-12Hz")
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
DOCTOR_VIDEO = "doctor_avatar.mp4"


def show_speaking_screen(answer_text, audio_buffer):
    import base64
    st.markdown(
        """
        <style>
        .svag-call-screen {
            background-color: #10151c;
            border-radius: 20px;
            padding: 30px 20px;
            text-align: center;
        }
        .svag-call-screen video {
            border-radius: 20px;
            width: 260px;
            max-width: 90%;
        }
        .svag-speaking-text {
            color: #4CAF50;
            font-size: 18px;
            margin-top: 15px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with open(DOCTOR_VIDEO, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <div class="svag-call-screen">
            <video autoplay loop muted playsinline>
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
            <p class="svag-speaking-text">🔊 SVAG bol raha hai...</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if audio_buffer:
        st.audio(audio_buffer, format="audio/mp3", autoplay=True)
    with st.expander("Jawab padhein"):
        st.markdown(answer_text)




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

with st.sidebar:
    st.header("⚙️ Settings")
    selected_label = st.selectbox("Jawab ki bhasha / Answer language", list(LANGUAGES.keys()), index=0)
    selected_language = LANGUAGES[selected_label]

col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.png", width=80)
with col2:
    st.title("SVAG — Ayurvedic AI Assistant")
st.caption("Ayurveda se juda koi bhi sawaal poochein — kisi bhi bhasha mein")

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


def svag_ask(question, language):
    results = vectordb.similarity_search(question, k=4)
    context = "\n\n".join([r.page_content for r in results])
    prompt = (
        f"You are SVAG, an Ayurvedic AI assistant. "
        f"Always answer in {language} language only, regardless of what language the question is in. "
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
        f"Use the Ayurvedic context below to answer the question. "
        f"If the answer is not in the context, say you don't know (in {language}).\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer (in {language}):"
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def svag_ask_image(image_bytes, language, user_note=""):
    import base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    text_instruction = (
        f"You are SVAG, an Ayurvedic AI assistant. Look at this image — it may be an Ayurvedic "
        f"herb, plant, root, powder, or medicine. Identify it if possible, and explain in "
        f"{language} language: its name (also give the Sanskrit/Ayurvedic name if known), its "
        f"Ayurvedic properties (rasa, guna, virya if relevant), common uses/benefits, and any "
        f"precautions. If you cannot confidently identify it, say so clearly in {language}. "
        f"Answer only in {language}."
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
    )
    return response.choices[0].message.content


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = SVAG_AVATAR if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

st.divider()
st.subheader("📷 Jadi-buti/medicine ki photo bhejo")
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

st.divider()
st.subheader("🎤 Bol ke sawaal poocho")
voice_input = st.audio_input("Yahan tap karke bolo")

if voice_input is not None:
    if st.button("Ye sawaal SVAG ko bhejo"):
        with st.spinner("Awaaz samjhi ja rahi hai..."):
            spoken_text = speech_to_text(voice_input.getvalue())

        st.session_state.messages.append({"role": "user", "content": spoken_text})

        with st.spinner("SVAG soch raha hai..."):
            voice_answer = svag_ask(spoken_text, selected_language)
            voice_audio = text_to_speech(voice_answer, selected_language)

        show_speaking_screen(voice_answer, voice_audio)
        st.markdown("**Sawaal:** " + spoken_text)

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
    st.session_state.messages.append({"role": "assistant", "content": answer})
