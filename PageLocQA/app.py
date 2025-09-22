# -------------------------------
# Streamlit-based scientific QA chatbot
# -------------------------------

# ===== Imports =====
import streamlit as st
from io import BytesIO
import pdfplumber
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.agents import Tool
from langchain.text_splitter import CharacterTextSplitter
from keybert import KeyBERT
from sentence_transformers import CrossEncoder, SentenceTransformer
from typing import List
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
import requests

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# ===== Custom Local LLM Wrapper =====
class OllamaLLM:
    def __init__(self, model="llama3"):
        self.model = model

    def _call(self, prompt: str, stop=None):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model":self.model,"prompt":prompt, "stream":False},
            )
        if response.status_code==200:
            return response.json().get("response", "")
        raise RuntimeError(f"Ollama LLM call failed: {response.text}")

    def invoke(self, input, config=None):
        return self._call(input).strip()

# ===== Custom Local Embeddings =====
class LocalEmbeddings(Embeddings):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

# Cross-encoder reranker
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Global FAISS vectorstore
vectorstore_global = None
if "feedback_log" not in st.session_state:
    st.session_state["feedback_log"] = []

# ===== File Parsing =====
def extract_text_from_pdf(pdf_file, document_name):
    pdf_file.seek(0)
    with pdfplumber.open(BytesIO(pdf_file.read())) as pdf:
        text_chunks = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_chunks.append({"text": text.strip(), "page": i + 1, "document": document_name})
            else:
                print(f"No text on page {i + 1} of {document_name}")
    return text_chunks

# ===== Extract text from uploaded file (PDF, HTML, TXT). =====
def get_uploaded_text(uploaded_files):
    raw_text = []
    for uploaded_file in uploaded_files:
        document_name = uploaded_file.name
        if document_name.endswith(".pdf"):
            text_chunks = extract_text_from_pdf(uploaded_file, document_name)
            raw_text.extend(text_chunks)
        elif uploaded_file.name.endswith((".html", ".htm")):
            soup = BeautifulSoup(uploaded_file.getvalue(), 'lxml')
            raw_text.append({"text": soup.get_text(), "page": None, "document": document_name})
        elif uploaded_file.name.endswith((".txt")):
            content = uploaded_file.getvalue().decode("utf-8")
            raw_text.append({"text": content, "page": None, "document": document_name})
    return raw_text

# ===== Text Chunking =====
def get_text_chunks(raw_text):
    splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
    final_chunks = []
    for chunk in raw_text:
        for split_text in splitter.split_text(chunk["text"]):
            final_chunks.append({"text": split_text, "page": chunk["page"], "document": chunk["document"]})
    return final_chunks

# ===== Vectorstore Creation =====
def get_vectorstore(text_chunks):
    if not text_chunks:
        raise ValueError("text_chunks is empty. Cannot initialize FAISS vectorstore.")

    embeddings = LocalEmbeddings()
    texts = [chunk["text"] for chunk in text_chunks]
    metadatas = [{"page": chunk["page"], "document": chunk["document"]} for chunk in text_chunks]

    return FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)

def set_global_vectorstore(vectorstore):
    global vectorstore_global
    vectorstore_global = vectorstore

# ===== Keyword Model =====
def get_kw_model():
    global kw_model
    if "kw_model" not in st.session_state:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.kw_model = KeyBERT(model=model)
    return st.session_state.kw_model

# ===== Reasoning Prompt =====
def self_reasoning(query, context):
    llm = OllamaLLM(model="llama3")
    reasoning_prompt = f"""
    You are an AI assistant that analyzes the context provided to answer the user's query comprehensively and clearly.
    Answer in a concise, factual way using the terminology from the context. Avoid extra explanation unless explicitly asked.
    YOU MUST mention the page number.
    If the user asked for only the page number, then you MUST answe ONLY THE PAGE NUMBER
    ### Example 1:
    **Question:** What is the purpose of the MODTRAN GUI?
    **Context:**
    [Page 10 of the docuemnt] The MODTRAN GUI helps users set parameters and visualize the model's output.
    **Answer:** The MODTRAN GUI assists users in parameter setup and output visualization. You can find the answer at Page 10 of the document provided.

    ### Example 2:
    **Question:** How do you run MODTRAN on Linux? Answer with page number.
    **Context:**
    [Page 15 of the docuemnt] On Linux systems, MODTRAN can be run using the `mod6c` binary via terminal.
    **Answer:** Use the `mod6c` binary via terminal. (Page 15 of the document)

    ### Now answer:
    **Question:** {query}
    **Context:**
    {context}

    **Answer:**
    """
    return llm.invoke(reasoning_prompt)

# ===== Search Functions =====
def faiss_search_with_keywords(query):
    global vectorstore_global
    if vectorstore_global is None:
        raise ValueError("FAISS vectorstore is not initialized.")
    kw_model = get_kw_model()
    keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    refined_query = " ".join([keyword[0] for keyword in keywords])
    retriever = vectorstore_global.as_retriever(search_kwargs={"k": 13})
    docs = retriever.get_relevant_documents(refined_query)
    context= '\n\n'.join([f"[Page {doc.metadata.get('page', 'Unknown')}] {doc.page_content}" for doc in docs])
    return self_reasoning(query, context)

def faiss_search_with_reasoning(query):
    global vectorstore_global
    if vectorstore_global is None:
        raise ValueError("FAISS vectorstore is not initialized.")
    retriever = vectorstore_global.as_retriever(search_kwargs={"k": 13})
    docs = retriever.get_relevant_documents(query)
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    reranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in reranked_docs[:5]]
    context = '\n\n'.join([f"[Page {doc.metadata.get('page', 'Unknown')}] {doc.page_content.strip()}" for doc in top_docs])
    return self_reasoning(query, context)

# ===== Tools =====
faiss_keyword_tool = Tool(
    name="FAISS Keyword Search",
    func=faiss_search_with_keywords,
    description="Searches FAISS with a keyword-based approach to retrieve context."
)

faiss_reasoning_tool = Tool(
    name="FAISS Reasoning Search",
    func=faiss_search_with_reasoning,
    description="Searches FAISS with detailed reasoning to retrieve context."
)

# ===== User Query Handling =====
def handle_user_query(query):
    if st.session_state.get("vectorstore"):
        # Use FAISS-based tools
        if "how" in query.lower():
            context = faiss_search_with_reasoning(query)
        else:
            context = faiss_search_with_keywords(query)
    else:
        # Use Wikipedia fallback
        print("No documents uploaded — using Wikipedia instead.")
        wiki_docs = wikipedia.run(query)
        context = f"[Wikipedia] {wiki_docs}"

    return self_reasoning(query, context)

# ===== Streamlit App =====
def main():
    global vectorstore_global

    if "chat_ready" not in st.session_state:
        st.session_state.chat_ready = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with Uploaded Documents")
    user_question = st.chat_input("Ask a question about your uploaded files:")

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload files:", accept_multiple_files=True)
        if st.button("Process") and uploaded_files:
            with st.spinner("Processing..."):
                raw_text = get_uploaded_text(uploaded_files)
                st.write(f"Extracted {len(raw_text)} raw text entries")

                if not raw_text:
                    st.error("No text extracted from uploaded files.")
                    return
                text_chunks = get_text_chunks(raw_text)
                st.write(f"Created {len(text_chunks)} text chunks")

                if not text_chunks:
                    st.error("No text chunks generated. Please check the uploaded file content.")
                    return
                st.session_state.vectorstore = get_vectorstore(text_chunks)
                set_global_vectorstore(st.session_state.vectorstore)
                st.session_state.chat_ready = True
                st.success("Files processed successfully!")

    if st.session_state.chat_ready and user_question:
        set_global_vectorstore(st.session_state.vectorstore)
        response = handle_user_query(user_question)
        st.session_state.chat_history.append({"user": user_question, "bot": response})

    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")

if __name__ == "__main__":
    main()
