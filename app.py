import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import shutil
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# -------------------- Load API Key --------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment.")
genai.configure(api_key=api_key)

# -------------------- Constants --------------------
PDF_DIR = "assets"
VECTOR_STORE_DIR = "faiss_index"

# -------------------- Helper Functions --------------------

def get_all_pdf_texts(folder_path):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            with open(os.path.join(folder_path, filename), "rb") as f:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    content = page.extract_text()
                    if content:
                        text += content + "\n"
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(VECTOR_STORE_DIR)

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)

def get_conversational_chain():
    prompt_template = """
You are a helpful and witty assistant who speaks on behalf of a software developer named Rajat Chauhan.
Your job is to promote his skills, experience, and projects with confidence, charm, and solid context.
Use the given context to answer every question persuasively — whether it's about his skills, job fit, background, or accomplishments.

When asked questions like "Is Rajat a good fit for .NET?", use the context to explain how his past experience aligns well with requirement — even if not directly stated.

If the answer is not present in the context, reply gracefully with:
"😅 I'd love to brag more, but this detail isn't in the portfolio!"

Context:\n{context}\n
Question:\n{question}\n

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemma-3-1b-it", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def get_emotion_emoji(answer_text):
    mapping = {
        ".Net": "🚀💪",
        "ASP.NET WEB API": "🛠️📦",
        "azure": "🐍💻",
        "ai|cnn|machine learning|ml": "🤖🧠",
        "achievement|project": "🏆🔥",
        "education|bachelor|school|college|institute": "🎓📚",
        "experience|worked at|company|job": "💼🛠️",
        "skills|technologies|tools": "🧰⚙️",
        "cloud|azure|server": "☁️🖥️",
        "not in the context": "😅🤷‍♂️",
    }
    for keyword, emoji in mapping.items():
        if re.search(keyword, answer_text, re.IGNORECASE):
            return emoji
    return "💬"

def answer_query(question):
    if not os.path.exists(f"{VECTOR_STORE_DIR}/index.faiss"):
        st.error("Vector store not found. Try restarting the app.")
        return

    db = load_vector_store()
    docs = db.similarity_search(question, k=5)

    chain = get_conversational_chain()
    response = chain({
        "input_documents": docs,
        "question": question
    }, return_only_outputs=True)

    output = response['output_text']
    emoji = get_emotion_emoji(output)

    st.markdown(f"""
> {emoji} **{output}**
""")

# -------------------- Main App --------------------

def main():
    st.set_page_config(page_title="Ask My Portfolio", page_icon="💼")
    st.title("💼 Ask My Resume/Portfolio")

    st.markdown("""
### 👋 Welcome!

I'm **Rajat Chauhan** — Enthusiastic software developer with over 3 years of experience in fintech and web services.
Expertise in creating robust payment solutions and streamlining transaction processes. Focused
on delivering high-performance, secure systems that drive innovation.

💡 Ask me anything — about my experience, tech stack, or whether I’m a great fit for your team.  
This chatbot knows my portfolio better than I know my fridge inventory. 😄

✨ _Try questions like:_
- *Is Rajat a good fit for .NET or Azure Cloud?*
- *What are his skills in .NET, AI or Cloud Computing?*
- *Which project did he worked on?*
""")

    # One-time PDF processing
    if not os.path.exists(f"{VECTOR_STORE_DIR}/index.faiss"):
        with st.spinner("🔍 First-time setup: Processing portfolio documents..."):
            full_text = get_all_pdf_texts(PDF_DIR)
            chunks = get_text_chunks(full_text)
            create_vector_store(chunks)
            st.success("✅ Portfolio processed and vector store saved!")

    question = st.text_input("🧠 Ask me anything:")
    if question:
        answer_query(question)

if __name__ == "__main__":
    main()
