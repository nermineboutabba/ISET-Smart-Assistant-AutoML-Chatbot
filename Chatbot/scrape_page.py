import requests
from bs4 import BeautifulSoup
import pandas as pd
import urllib.parse
import time
import re
import json

# NLP imports
import spacy
from transformers import pipeline

# Load French spaCy model
nlp_spacy = spacy.load("fr_core_news_sm")

# Load CamemBERT for QA
qa_pipeline = pipeline("question-answering", model="illuin/camembert-base-fquad")
# Configuration
USER_AGENT = "Mozilla/5.0"
HEADERS = {"User-Agent": USER_AGENT}
DELAY = 2
MAX_RETRIES = 1
base_url = "https://isetsf.rnu.tn/fr "

MENU_CATEGORIES = {
    "admission": "Admissions",
    "programme": "Programs",
    "contact": "Contact",
    "actualite": "News",
    "bureau": "Administration",
    "departement": "Departments",
    "recherche": "Research",
}

def categorize_page(url):
    for keyword, category in MENU_CATEGORIES.items():
        if keyword in url.lower():
            return category
    return "General"

def safe_request(url):
    for attempt in range(MAX_RETRIES):
        try:
            print(f"[Request] {url}")
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"[Error] {e} - Retrying ({attempt+1}/{MAX_RETRIES})")
            time.sleep(DELAY)
    return None

# Rule-based question generator from paragraphs
def generate_questions_from_text(text):
    doc = nlp_spacy(text)
    questions = []

    for sent in doc.sents:
        if len(sent.text.split()) < 6:
            continue
        if "?" in sent.text:
            continue

        subj = None
        verb = None
        for token in sent:
            if token.dep_ == "nsubj":
                subj = token.text
            if token.pos_ == "VERB":
                verb = token.text

        if subj and verb:
            questions.append(f"Qu'est-ce que {subj} fait ?")
            questions.append(f"Pourquoi {subj} {verb} ?")
            questions.append(f"Comment {subj} {verb} ?")

    return list(set(questions))

# Extract paragraphs and run QA pipeline
def extract_qa_with_nlp(context, source_url, category):
    doc = nlp_spacy(context)
    paragraphs = [sent.text for sent in doc.sents if len(sent.text.split()) > 10]

    results = []
    seen = set()

    for para in paragraphs:
        questions = generate_questions_from_text(para)
        for q in questions:
            try:
                answer = qa_pipeline(question=q, context=para)
                if len(answer['answer'].strip()) < 3 or answer['answer'].lower() in seen:
                    continue
                seen.add(answer['answer'].lower())
                results.append({
                    "category": category,
                    "question": q,
                    "answer": answer["answer"].strip(),
                    "source": source_url
                })
            except Exception as e:
                print(f"[QA Error] {e}")
                continue
    return results

# Main scraping function
def extract_qa_from_page(url):
    html = safe_request(url)
    if not html:
        return []
    soup = BeautifulSoup(html, 'html.parser')

    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    text_content = ' '.join(soup.stripped_strings)
    category = categorize_page(url)

    # Clean and split into chunks for QA processing
    clean_text = re.sub(r'\s+', ' ', text_content).strip()
    return extract_qa_with_nlp(clean_text, url, category)

def is_valid_link(href):
    return (
        href and
        href.startswith("http") and
        "isetsf.rnu.tn/fr" in href and
        not any(ext in href for ext in [".pdf", ".jpg", ".png", ".doc", "mailto:", "tel:"])
    )

def collect_all_internal_links(start_url):
    visited = set()
    to_visit = [start_url]

    while to_visit:
        current_url = to_visit.pop()
        if current_url in visited:
            continue

        visited.add(current_url)
        html = safe_request(current_url)
        if not html:
            continue

        soup = BeautifulSoup(html, 'html.parser')
        for link_tag in soup.find_all('a', href=True):
            href = urllib.parse.urljoin(current_url, link_tag['href'])
            if is_valid_link(href) and href not in visited:
                to_visit.append(href)

    return list(visited)

def create_database_json(output_file="iset_faq_enhanced.json"):
    print("Crawling the site for internal links...")
    links = collect_all_internal_links("https://isetsf.rnu.tn/fr")

    all_data = []

    for idx, link in enumerate(links):
        print(f"\n({idx + 1}/{len(links)}) Processing: {link}")
        qa_list = extract_qa_from_page(link)
        all_data.extend(qa_list)
        time.sleep(DELAY)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\nEnhanced JSON database saved to '{output_file}' with {len(all_data)} entries.")

if __name__ == "__main__":
    create_database_json()