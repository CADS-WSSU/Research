# PageLocQA: A Scientific QA Chatbot with Page-level Grounding 

This repository contains a **Streamlit-based chatbot** that enables users to query scientific documents (PDF, HTML, TXT) with page-level grounding.  

## Features
- Upload and process **PDF, HTML, or TXT** files.
- Build a **FAISS vectorstore** for semantic retrieval.
- Integrate **Ollama (LLaMA3)** as the local reasoning LLM.
- Two retrieval modes:
  - **Keyword-based FAISS Search** (KeyBERT-enhanced).
  - **Reasoning-based FAISS Search** (CrossEncoder reranker).
- **Wikipedia fallback** when no local documents are uploaded.
- Provides **page number grounding** for transparency and trust.


## Installation

1. Clone the repository:
   - git clone https://github.com/CADS-WSSU/Research/PagelocQA.git
   - cd PagelocQA
2. Create and activate a virtual environment:
   - python -m venv venv
   - source venv/bin/activate   # Linux/Mac
   - venv\Scripts\activate      # Windows
3. Install dependencies:
   pip install -r requirements.txt

## How to use 
1. Start the Ollama server locally (requires Ollama):
   ollama run llama3
2. Launch the Streamlit app:
   streamlit run app.py
4. Upload one or more documents via the "Browse Files" button in the sidebar. Wait until the "Running" is done. Then press "Process". When you see "Files processed successfully", then use the chatbox and write your question.

## Acknowledgement 

This research was supported by the NASA MUREP/DEAP IMPACT (Institute for Multi-agent Perception through Advanced Cyber-Physical Technologies) Award, No. 80NSSC23M0054. Portions of this research were carried out at the Jet Propulsion Laboratory, California Institute of Technology, under a contract with the National Aeronautics and Space Administration (80NM0018D0004). Reference
herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise, does not constitute or imply its endorsement by the United States Government or the Jet Propulsion Laboratory, California Institute of Technology

## Citation

If you use this repository in your research, please cite the following paper:

**Zarin T. Shejuti, Debzani Deb, Emily R. Dunkel, “From Answer to Origin: Page Number Grounding in Document-Level Question Answering”, 24th International Conference on Machine Learning and Applications (ICMLA), Boca Center, FL, USA, 2025. (Accepted)**
