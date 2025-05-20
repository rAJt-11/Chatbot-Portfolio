# 💼 Rajat Chauhan — Interactive Portfolio Assistant

## ✨ About the Project

This is an AI-powered, interactive resume and portfolio assistant built using **Streamlit**, **LangChain**, **Google Generative AI**, and **FAISS vector search**.

It allows users, recruiters, or collaborators to ask questions about Rajat Chauhan's skills, experience, and accomplishments — with intelligent, contextual responses powered by AI.

Instead of browsing a static resume, you get a conversational experience tailored to real-world tech evaluations.

---

## 🔑 Key Features

- 🤖 **Conversational AI**: Built with LangChain and Google's Gemma model to answer portfolio-related questions contextually.
- 📄 **Smart Document Parsing**: Automatically extracts content from PDFs and converts it into searchable knowledge.
- 🔍 **Semantic Search**: Uses FAISS vector store to perform fast, meaningful similarity searches over resume content.
- 💬 **Witty, Professional Assistant**: Designed to respond with persuasive, helpful answers — with emoji-based sentiment cues.
- 📦 **Modular & Scalable**: Easy to extend with more documents, models, or question types.

---

## 🏗️ Built With

![Python](https://img.shields.io/badge/-Python-000?logo=python)
![Streamlit](https://img.shields.io/badge/-Streamlit-000?logo=streamlit)
![LangChain](https://img.shields.io/badge/-LangChain-000)
![Google Generative AI](https://img.shields.io/badge/-Google%20GenAI-000?logo=google)
![FAISS](https://img.shields.io/badge/-FAISS-000?logo=meta)
![PyPDF2](https://img.shields.io/badge/-PyPDF2-000?logo=adobe-acrobat-reader)
![Visual Studio Code](https://img.shields.io/badge/-Visual%20Studio%20Code-000?logo=visual-studio-code)

---

## 🚀 How It Works

1. Upload Rajat's resume/portfolio PDFs to the `/assets` directory.
2. On first run, the app:
   - Extracts content from all PDFs
   - Splits text into chunks for efficient indexing
   - Creates a vector store with Google Embeddings
3. Users can ask questions in the Streamlit interface.
4. Relevant documents are retrieved and answered using a custom LangChain QA chain and prompt template.

---

## 💬 Sample Questions to Ask

- *Is Rajat a good fit for a .NET or Azure role?*
- *What projects has he worked on involving AI?*
- *What technologies does he specialize in?*
- *Tell me about his experience in cloud or payment solutions.*

---

## 🤝 Contributing

This project is tailored for showcasing **Rajat Chauhan’s** portfolio, but it's open for educational inspiration and extension.

If you'd like to adapt it to your own resume, feel free to fork and modify.

---

## 📧 Contact

For project inquiries or collaborations:

[![Gmail Badge](https://img.shields.io/badge/-dreamerrajat11@gmail.com-FF0000?style=flat-square&logo=Gmail&logoColor=white&link=mailto:dreamerrajat11@gmail.com)](mailto:dreamerrajat11@gmail.com)

---

## 🔐 License

Distributed under the **MIT License**.  
See `LICENSE` for more information.
