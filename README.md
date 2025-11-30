

# 🔍 **RAG for Diagnostic Reasoning in Clinical Notes (DiReCT)**

*A Retrieval-Augmented Generation system for answering clinical queries using the MIMIC-IV-Ext Direct dataset.*

---

## 🧠 **Introduction**

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline designed to support diagnostic reasoning from clinical notes.
Given a natural-language clinical query, the system:

1. **Retrieves** the most relevant clinical notes from MIMIC-IV-Ext Direct
2. **Generates** a medically coherent, context-aware answer using a generative LLM
3. Displays results in a **Streamlit web app**

This repository includes **data loading, preprocessing, retrieval, generation, evaluation, Streamlit frontend**, and full documentation.

---

# 📁 **Project Features**

### ✅ **Complete RAG Pipeline**

* Dense retrieval using **BioBERT embeddings**
* Context-aware generation using **Flan-T5**
* Fully modular API structure

### ✅ **Streamlit Frontend**

* Query input
* Retrieved documents viewer
* Final AI-generated reasoning/diagnosis

### ✅ **End-to-End Codebase**

* Data loading
* Preprocessing
* Retriever
* Generator
* Evaluation metrics

### ✅ **Documentation + Social Posts**

* Medium blog post (800–1500 words) — **template included**
* LinkedIn post — **template included**

---

# 🏥 **Dataset: MIMIC-IV-Ext Direct**

⚠️ **Dataset is NOT included in this repo** (medical data cannot be redistributed).
Users must download it separately.

Dataset consists of:

* **Diagnostic Flowcharts** (`Diagnosis_flowchart/`)
* **Annotated Clinical Notes** (`Finished/`)
* **Structured reasoning steps**
* **Physician decision pathways**

Your code loads both flowcharts & annotated samples automatically.

---

# 🧱 **RAG System Architecture**

```
              ┌────────────────────────┐
              │   User Clinical Query   │
              └─────────────┬──────────┘
                            ▼
     ┌──────────────────────────────────────┐
     │      Dense Retriever (BioBERT)       │
     │ - Embeds all documents               │
     │ - Embeds query                       │
     │ - Computes cosine similarity         │
     └─────────────┬────────────────────────┘
                   ▼
       ┌────────────────────────────┐
       │     Top-K Retrieved Docs    │
       └─────────────┬──────────────┘
                     ▼
     ┌──────────────────────────────────────┐
     │  Generative Model (Flan-T5 / LLM)    │
     │ - Combines context + query           │
     │ - Generates physician-style answer   │
     └─────────────┬────────────────────────┘
                   ▼
          ┌──────────────────────┐
          │   Final AI Response  │
          └──────────────────────┘
```

---

# ⚙️ **Installation**

## 1️⃣ Clone the Repo

```bash
git clone https://github.com/yourusername/RAG-DiReCT.git
cd RAG-DiReCT
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Set Path to Dataset (MIMIC-IV-Ext Direct)

Place data in:

```
/data/Diagnosis_flowchart/
/data/Finished/
```

Or modify paths inside `rag_pipeline.py`.

---

# 🚀 **Usage**

## ▶️ **Run the RAG Pipeline**

```bash
python main.py
```

This will:

* Load dataset
* Preprocess documents
* Run retrieval
* Generate final answer

---

## 🌐 **Run the Streamlit App**

```bash
streamlit run app.py
```

The UI provides:

* **Query Input Box**
* **Retrieved Documents Viewer**
* **Final RAG-Generated Clinical Answer**

---

# 🧩 **Code Structure**

```
RAG-DiReCT/
│
├── app.py                    # Streamlit frontend
├── NLP_RAG.py             # Complete RAG Project
├── utils/                    # Misc functions
├── requirements.txt
└── README.md
```

---

# 🛠 **Technical Components**

## 🔹 **Retrieval Module**

* Model: **pritamdeka/BioBERT-NLI-STSB**
* Uses cosine similarity
* Top-K retrieval implemented using PyTorch

## 🔹 **Generation Module**

* Model: **google/flan-t5-large**
* Deterministic decoding (`do_sample=False`)
* Max length: 512 tokens

## 🔹 **Pipeline Integration**

Retrieval → Prompt Construction → Generation

## 🔹 **Frontend**

* Built in **Streamlit**
* Lightweight & responsive
* Transparent document preview

---

# 📊 **Evaluation**

## **Retrieval Metrics**

* **Precision@K**
* **Recall@K**

## **Generation Metrics**

* **Relevance**
* **Coherence**
* **Medical reasoning quality**
* **Human evaluation (optional)**

---

# 🔒 **Ethical & Privacy Considerations**

This project follows:

✔ HIPAA guidelines
✔ No raw patient data included
✔ Only processed embeddings used
✔ No private PHI is displayed
✔ Model outputs should **not** replace clinical judgment

---

It covers:

* Motivation
* Dataset
* RAG architecture
* Retrieval
* Generation
* Lessons learned

---

# 🤝 **Acknowledgments**

* MIMIC-IV-Ext dataset creators
* Hugging Face for open-source tools
* BioBERT authors
* FLAN-T5 team
* Streamlit for UI

---


# 📄 **License**

Apache 2.0 License
