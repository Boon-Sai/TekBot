# 🧠 BERT-powered Q&A on Text Files with Streamlit

This Streamlit app lets you **upload a `.txt` file** and ask questions about its content.  
It uses **Google BERT (via `bert-base-nli-mean-tokens`)** to find the most relevant answer from the text using semantic similarity.

---

## 🚀 Features

- Upload and preview a `.txt` file
- Text is split into lines/paragraphs
- Uses `sentence-transformers` with BERT to embed each chunk
- Enter a question and get the **most relevant matching part** of the text
- Runs locally, no external API calls after setup

---

## 📦 Requirements

Install dependencies using:
pip install -r requirements.txt



# 🧪 How to Run
## Login with your token using:
commands : 
huggingface-cli login
"enter your token here"


## Run:

streamlit run app.py
