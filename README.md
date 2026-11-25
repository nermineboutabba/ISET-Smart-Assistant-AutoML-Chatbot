# ISET Smart Assistant — AutoML Workbench & Hybrid RAG Chatbot

> **Collaboration Notice**: Co-developed in collaboration with **Moetaz** ([@Moetaz0](https://github.com/Moetaz0)).

An interactive Python **Flask** machine learning and NLP platform uniting an **AutoML Data Science Workbench** with a **multilingual hybrid FAQ/RAG Chatbot** built for **ISET Sfax**.

## Key Features
1. **Multilingual Hybrid RAG Chatbot (`Chatbot/`)**
   - Combines TF-IDF / FAISS semantic similarity retrieval, `rapidfuzz` fuzzy matching, greeting detection, multilingual translation fallback, and optional HuggingFace LLM answer synthesis (`llm_model.py`).
   - Built-in ISET Sfax knowledge base (`Chatbot/data/iset_faq.csv` & `iset_faq_enhanced.json`) plus interactive user feedback logging (`/feedback`).
   - Web scraping & QA generation utilities (`scrape_page.py`, `web_crawler.py`) and Falcon fine-tuning script (`train_falcon.py`).
2. **No-Code AutoML & Data Preprocessing Studio (`/upload`, `/preprocess`, `/train_models`)**
   - Upload CSV or JSON datasets, apply missing-value strategies (`mean`, `median`, `mode`, `drop`), inspect dataset diffs, and download cleaned CSV + ReportLab PDF audit reports.
   - Train and benchmark classification (`LazyClassifier`), regression (`LazyRegressor`), and clustering (`KMeans`, `DBSCAN`, `AgglomerativeClustering`) models with automated Seaborn correlation heatmaps and downloadable PDF results reports.

## Architecture & Documentation
- Full design reports (`Rapport/RapportV2.pdf`), MLOps diagrams, and UI screenshots are available under `Rapport/`.
- Both the unified application (`Chatbot/app.py`) and original standalone AutoML prototype (`Treatment App/flask_data_science_app/app.py`) are preserved.

## Quick Start
```bash
cd Chatbot
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5050` for the unified landing page (`/chatbot` for the assistant and `/upload` for AutoML).
