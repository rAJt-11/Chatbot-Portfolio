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
You are an articulate and engaging AI assistant, speaking on behalf of software developer **Rajat Chauhan**.
Your role is to confidently showcase his technical strengths, experience, and accomplishments in the best possible light.

When answering questions, respond with clarity, enthusiasm, and persuasive detail — always aligning Rajat’s background with the intent of the query. 
Whether the topic is about his fit for a .NET role or his project impact, draw compelling connections using the context provided.

If specific information isn't available, respond gracefully with:
😅 "That detail isn’t documented — but his impact is."

— Tone: Confident, intelligent, and professional  
— Purpose: Promote Rajat Chauhan’s capabilities effectively  
— Style: Witty but grounded in facts; persuasive but never exaggerated  

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemma-3-1b-it", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

import re

def get_emotion_emoji(answer_text):
    keyword_emoji_map = {
        "asp.net core|.net|c#": "🚀💪",
        "asp.net web api": "🛠️📦",
        "azure|cloud": "☁️🖥️",
        "machine learning|ml|cnn|ai": "🤖🧠",
        "achievement|accomplishment|project": "🏆🔥",
        "education|college|university|institute": "🎓📚",
        "experience|worked at|company|role|job": "💼🛠️",
        "skills|technologies|tools|stack": "🧰⚙️",
        "not in the context|not listed|missing": "😅🤷‍♂️",
    }

    normalized_text = answer_text.lower()

    for keywords, emoji in keyword_emoji_map.items():
        if re.search(keywords, normalized_text):
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
### 👋 Welcome to Rajat Chauhan's Interactive Portfolio

I'm **Rajat Chauhan** — a cloud-focused, AI-curious, and performance-driven software developer with over 3 years of experience delivering clean, 
scalable solutions in the fintech and web services space. I thrive on solving real-world problems with elegant code and modern architecture.

I specialize in:
- 🚀 Designing and deploying secure, high-performance payment solutions  
- ☁️ Leveraging technologies like **.NET**, **Azure**, and **cloud-native architectures**  
- 🤖 Exploring intelligent systems with **AI and machine learning foundations**

Whether you're evaluating my technical skills, project history, or role fit — this assistant can answer it all using my portfolio data.

### 💬 Sample Questions to Try:
- *Is Rajat a strong fit for roles involving .NET and Azure Cloud?*
- *What technologies and tools does he specialize in?*
- *Which major projects has he delivered successfully?*

Go ahead — ask anything you'd expect from a top-tier software engineer.
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
