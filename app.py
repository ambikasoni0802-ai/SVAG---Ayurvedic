import streamlit as st
import json, os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from groq import Groq

st.set_page_config(page_title="SVAG - Ayurvedic AI", page_icon="🌿")
st.title("🌿 SVAG — Ayurvedic AI Assistant")
st.caption("Ayurveda se juda koi bhi sawaal poochein")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


@st.cache_resource(show_spinner="SVAG ka brain taiyar ho raha hai... (pehli baar 3-5 min lagenge)")
def load_svag():
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=t) for t in all_texts]
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory="svag_db")
    return vectordb


vectordb = load_svag()
client = Groq(api_key=GROQ_API_KEY)


def svag_ask(question):
    results = vectordb.similarity_search(question, k=4)
    context = "\n\n".join([r.page_content for r in results])
    prompt = (
        f"Tum SVAG ho, ek Ayurvedic AI assistant. Neeche diye gaye Ayurvedic context ke "
        f"aadhar par sawaal ka jawab do. Agar context me jawab na mile to bolo ki tumhe pata nahi.\n\n"
        f"Context:\n{context}\n\nSawaal: {question}\n\nJawab:"
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
            answer = svag_ask(user_question)
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
