# Local RAG AI Assistant (Llama 3 + Flask)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Stack](https://img.shields.io/badge/Tech-LangChain%20%7C%20Flask%20%7C%20FAISS-green)
![GPU](https://img.shields.io/badge/Hardware-NVIDIA%20GPU%20(CUDA)-76b900)

A high-performance, privacy-focused AI assistant capable of answering questions based on PDF documents. It runs **100% locally** using **Llama 3** (via `llama.cpp`) and **Retrieval-Augmented Generation (RAG)** technology.

> **Key Feature:** No external APIs (like OpenAI) are required. Your data never leaves your machine.

---

## ⚡ Features

- **Offline Inference:** Runs Llama 3 locally using GPU acceleration (`n_gpu_layers=-1`).
- **RAG Architecture:** Uses **FAISS** and **HuggingFace Embeddings** to retrieve relevant context from PDFs.
- **Web Interface:** Simple and clean UI built with **Flask** and **HTML/CSS**.
- **Memory Efficient:** Optimized for consumer GPUs (e.g., RTX 3060/4060).

---

## Tech Stack

- **LLM Engine:** [LlamaCPP](https://github.com/abetlen/llama-cpp-python) (GGUF Format)
- **Orchestration:** [LangChain](https://www.langchain.com/)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Backend:** Flask (Python)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (running on CUDA)

---

## Prerequisites (Important)

Since the model and document files are large or sensitive, they are **not included** in this repository. You must add them manually:

### 1. Download the Model
Create a folder named `modelo` in the root directory.
Download a **Llama 3 GGUF** model (e.g., `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`) from HuggingFace and rename it to `llama-3.gguf`.
> Place it here: `./modelo/llama-3.gguf`

### 2. Add your Document
Create a folder named `documentos` in the root directory.
Add the PDF you want to chat with and rename it to `contrato.pdf`.
> Place it here: `./documentos/contrato.pdf`

---

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
