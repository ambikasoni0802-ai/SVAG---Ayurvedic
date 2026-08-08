import streamlit as st
import json, os, subprocess
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from groq import Groq

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
        f"Use the Ayurvedic context below to answer the question. "
        f"If the answer is not in the context, say you don't know (in {language}).\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer (in {language}):"
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Apna Ayurvedic sawaal likho...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("SVAG soch raha hai..."):
            answer = svag_ask(user_question, selected_language)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
