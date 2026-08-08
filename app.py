import streamlit as st
import json, os, subprocess, io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from groq import Groq
from gtts import gTTS

LANG_TO_TTS_CODE = {
    "English": "en", "Hindi": "hi", "Marathi": "mr", "Tamil": "ta", "Telugu": "te",
    "Kannada": "kn", "Malayalam": "ml", "Punjabi": "pa", "Bengali": "bn",
    "Gujarati": "gu", "Odia": "or", "Urdu": "ur", "Nepali": "ne", "Sinhala": "si",
    "Spanish": "es", "French": "fr", "German": "de", "Portuguese": "pt",
    "Italian": "it", "Russian": "ru", "Chinese": "zh-CN", "Japanese": "ja",
    "Korean": "ko", "Arabic": "ar", "Indonesian": "id", "Turkish": "tr",
    "Vietnamese": "vi", "Thai": "th", "Swahili": "sw", "Dutch": "nl",
}


def text_to_speech(text, language):
    tts_code = LANG_TO_TTS_CODE.get(language, "en")
    try:
        tts = gTTS(text=text, lang=tts_code)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
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

st.set_page_config(page_title="SVAG - Ayurvedic AI", page_icon="🌿")

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

st.title("🌿 SVAG — Ayurvedic AI Assistant")
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
    with st.chat_message(msg["role"]):
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
        with st.chat_message("assistant"):
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
        with st.chat_message("user"):
            st.markdown(spoken_text)
        with st.chat_message("assistant"):
            with st.spinner("SVAG soch raha hai..."):
                voice_answer = svag_ask(spoken_text, selected_language)
                st.markdown(voice_answer)
                voice_audio = text_to_speech(voice_answer, selected_language)
                if voice_audio:
                    st.audio(voice_audio, format="audio/mp3")
        st.session_state.messages.append({"role": "assistant", "content": voice_answer})

user_question = st.chat_input("Apna Ayurvedic sawaal likho...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("SVAG soch raha hai..."):
            answer = svag_ask(user_question, selected_language)
            st.markdown(answer)
            audio = text_to_speech(answer, selected_language)
            if audio:
                st.audio(audio, format="audio/mp3")
    st.session_state.messages.append({"role": "assistant", "content": answer})
