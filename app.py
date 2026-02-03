from flask import Flask, render_template, request
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import torch

app = Flask(__name__)


print(" STARTING SYSTEM ")


if torch.cuda.is_available():
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ WARNING: GPU not detected. System might be slow.")

print("1. Loading Llama 3 on GPU...")
llm = LlamaCpp(
    model_path="./modelo/llama-3.gguf",
    n_gpu_layers=-1,       
    n_ctx=2048,            
    n_batch=512,           
    verbose=True
)

#  RAG CONFIGURATION 
print("2. Configuring PDF Loader on GPU...")

# EXPLICIT configuration to use CUDA for Embeddings
model_kwargs = {'device': 'cuda'}

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs=model_kwargs  
)

print("3. Reading PDF and indexing...")
# Load PDF
try:
    
    loader = PyPDFLoader("./documentos/contrato.pdf") 
    docs = loader.load()
    
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("✅ Database created successfully!")

    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever()
    )

except Exception as e:
    print(f"❌ ERROR READING PDF: {e}")
    # Fallback to avoid crashing the app if PDF fails
    qa_chain = None 

#  FLASK ROUTES 

@app.route("/", methods=["GET", "POST"])
def index():
    response_text = ""
    user_question = ""
    
    if request.method == "POST":
        # NOTE: "user_question" must match the 'name' attribute in your HTML input
        user_question = request.form.get("user_question") 
        print(f"Processing question: {user_question}")
        
        if qa_chain:
            try:
                result = qa_chain.invoke(user_question)
                response_text = result['result']
            except Exception as e:
                response_text = f"Error processing response: {str(e)}"
        else:
            response_text = "Error: The database (PDF) was not loaded correctly."

    return render_template("index.html", response=response_text, previous_question=user_question)

if __name__ == "__main__":
    
    app.run(debug=True, use_reloader=False)