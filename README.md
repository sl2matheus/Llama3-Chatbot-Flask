# Local RAG AI Assistant (Llama 3 + Flask)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Stack](https://img.shields.io/badge/Tech-LangChain%20%7C%20Flask%20%7C%20FAISS-green)
![GPU](https://img.shields.io/badge/Hardware-NVIDIA%20GPU%20(CUDA)-76b900)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20WSL2-orange)

A high-performance, privacy-focused AI assistant capable of answering questions based on PDF documents. It runs **100% locally** using **Llama 3** (via `llama.cpp`) and **Retrieval-Augmented Generation (RAG)** technology.

> **Key Feature:** No external APIs (like OpenAI) are required. Your data never leaves your machine.

---

##  Features

- **Offline Inference:** Runs Llama 3 locally using GPU acceleration (`n_gpu_layers=-1`).
- **RAG Architecture:** Uses **FAISS** and **HuggingFace Embeddings** to retrieve relevant context from PDFs.
- **Web Interface:** Simple and clean UI built with **Flask** and **HTML/CSS**.
- **Memory Efficient:** Optimized for consumer GPUs (e.g., RTX 3060/4060) using GGUF quantization.

---

##  Prerequisites & Setup

Since the model and document files are large or sensitive, they are **not included** in this repository. You must add them manually to the **root folder**.

**1. Prepare the Files:**
* **Model:** Download `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` from HuggingFace, rename it to `llama-3.gguf` and place it in the project root.
* **Document:** Place your PDF file in the project root and rename it to `contrato.pdf`.

**2. Installation (Linux / WSL):**

```bash
# 1. Clone and Enter Project
git clone [https://github.com/your_user/your_repository.git](https://github.com/your_user/your_repository.git)
cd your_repository

# 2. Create and Activate Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Install Llama with GPU Support 
pip install llama-cpp-python \
  --extra-index-url [https://abetlen.github.io/llama-cpp-python/whl/cu124](https://abetlen.github.io/llama-cpp-python/whl/cu124)
