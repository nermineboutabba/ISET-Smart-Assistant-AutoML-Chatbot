from difflib import SequenceMatcher
import matplotlib
matplotlib.use("Agg")
import os
import pickle
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, session
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from rapidfuzz import fuzz as _rapidfuzz
except ImportError:
    _rapidfuzz = None
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as PdfImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from werkzeug.utils import secure_filename

from llm_model import generate_answer
from web_crawler import is_greeting, search_local_knowledge

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_CSV = DATA_DIR / "iset_faq.csv"
UPLOAD_FOLDER = BASE_DIR / "uploads"
STATIC_FOLDER = BASE_DIR / "static"
ALLOWED_EXTENSIONS = {"csv", "json"}

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(STATIC_FOLDER),
)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-iset-smart-assistant-key")

df = None
question_corpus = []
tfidf_vectorizer = None
tfidf_matrix = None


def preprocess(text):
    tokens = [w for w in str(text).lower().split() if w.isalpha()]
    return " ".join(tokens) if tokens else str(text).lower().strip()


def load_knowledge_base():
    global df, question_corpus, tfidf_vectorizer, tfidf_matrix
    if not DATA_CSV.exists():
        df = pd.DataFrame(columns=["Question", "Answer", "Category", "Source"])
        return

    loaded = pd.read_csv(DATA_CSV).dropna(subset=["Question", "Answer", "Source"]).copy()
    loaded["CleanedQuestion"] = loaded["Question"].apply(preprocess)
    df = loaded
    question_corpus = df["CleanedQuestion"].tolist()

    if question_corpus:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(question_corpus)


def fuzzy_score(q1, q2):
    if _rapidfuzz is not None:
        return _rapidfuzz.token_sort_ratio(q1, q2)
    return int(round(SequenceMatcher(None, str(q1), str(q2)).ratio() * 100))


def search_csv(query_text, threshold=0.35, top_k=5, fuzzy_threshold=60):
    if df is None or df.empty or tfidf_vectorizer is None:
        return None

    clean_q = preprocess(query_text)
    if not clean_q.strip():
        return None

    q_vec = tfidf_vectorizer.transform([clean_q])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    ranked = np.argsort(sims)[::-1][:top_k]

    top_matches = []
    for idx in ranked:
        score = float(sims[idx])
        matched_q = str(df.iloc[idx]["Question"])
        fscore = fuzzy_score(clean_q, str(df.iloc[idx]["CleanedQuestion"]))
        if score >= threshold or fscore >= fuzzy_threshold:
            top_matches.append({
                "matched_question": matched_q,
                "answer": str(df.iloc[idx]["Answer"]),
                "source": str(df.iloc[idx]["Source"]),
            })
    return top_matches if top_matches else None


def _translate_text(text, dest="en"):
    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source="auto", target=dest).translate(text)
        return translated or text
    except Exception:
        return text


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    translated_en = _translate_text(question, dest="en")
    if is_greeting(translated_en):
        greeting = (
            "Hello! I am the ISET Smart Assistant — ask me about ISET Sfax admissions, "
            "departments, timetables, internships, or AutoML datasets."
        )
        return jsonify({"question": question, "answer": greeting})

    top_matches = search_csv(translated_en)
    if top_matches:
        best = top_matches[0]
        return jsonify({
            "question": best["matched_question"],
            "answer": best["answer"],
            "source": best["source"],
        })

    content = search_local_knowledge(translated_en)
    answer = generate_answer(translated_en, question, content)
    return jsonify({
        "question": question,
        "answer": answer,
        "source_type": "local_knowledge",
    })


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    user_query = data.get("question", "")
    fb = data.get("feedback", "")
    scraped_answer = data.get("scraped_answer", "")
    source = data.get("source", "community")

    if fb == "Yes" and user_query and scraped_answer:
        new_row = pd.DataFrame({
            "Question": [user_query],
            "Answer": [scraped_answer],
            "Category": ["general"],
            "Source": [source],
        })
        existing = pd.read_csv(DATA_CSV) if DATA_CSV.exists() else pd.DataFrame(columns=["Question", "Answer", "Category", "Source"])
        updated = pd.concat([existing, new_row], ignore_index=True)
        updated.to_csv(DATA_CSV, index=False)
        load_knowledge_base()
        return jsonify({"message": "Thanks for your feedback! The entry has been saved."})

    return jsonify({"message": "Feedback recorded. Re-evaluating answer options."})


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/chatbot")
def chatbot():
    return render_template("index.html")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.context_processor
def inject_preprocessing_done():
    return dict(preprocessing_done=session.get("preprocessing_done", False))


def _upload_path(name):
    return str(UPLOAD_FOLDER / name)


def generate_pdf_from_differences(differences, output_path=None):
    out_file = output_path or _upload_path("differences.pdf")
    doc = SimpleDocTemplate(out_file, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Data Preprocessing Changes", styles["Title"]))
    elements.append(Spacer(1, 12))

    if not differences:
        elements.append(Paragraph("No differences were detected.", styles["Normal"]))
    else:
        for col, diff in differences.items():
            elements.append(Paragraph(f"<b>Column:</b> {col}", styles["Heading4"]))
            elements.append(Spacer(1, 6))
            rows = [["Original", "New"]]
            max_len = max(len(diff["original"]), len(diff["new"]))
            for i in range(max_len):
                orig = diff["original"][i] if i < len(diff["original"]) else ""
                new = diff["new"][i] if i < len(diff["new"]) else ""
                rows.append([str(orig), str(new)])
            table = Table(rows, repeatRows=1, colWidths=[250, 250])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))

    doc.build(elements)


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files or not request.files["file"].filename:
            flash("No file selected.")
            return redirect(request.url)

        file = request.files["file"]
        if not allowed_file(file.filename):
            flash("Unsupported file format. Upload a CSV or JSON file.")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = _upload_path(filename)
        file.save(filepath)
        session["filename"] = filename

        loaded_df = pd.read_csv(filepath) if filename.endswith(".csv") else pd.read_json(filepath)
        with open(_upload_path("original_df.pkl"), "wb") as f:
            pickle.dump(loaded_df, f)
        with open(_upload_path("current_df.pkl"), "wb") as f:
            pickle.dump(loaded_df, f)

        session["preprocessing_done"] = False
        return render_template(
            "upload.html",
            table_html=loaded_df.head(10).to_html(classes="table table-bordered", index=False),
            column_names=loaded_df.columns.tolist(),
        )

    session["preprocessing_done"] = False
    return render_template("upload.html", table_html=None)


@app.route("/preprocess", methods=["POST"])
def preprocess_data():
    if not os.path.exists(_upload_path("current_df.pkl")):
        flash("Please upload a dataset first.")
        return redirect("/upload")

    with open(_upload_path("current_df.pkl"), "rb") as f:
        cur_df = pickle.load(f)

    missing_strategy = request.form.get("missing_values", "")
    col_drop = request.form.get("column_to_drop", "")

    if col_drop and col_drop in cur_df.columns:
        cur_df.drop(columns=[col_drop], inplace=True)

    for col in cur_df.columns:
        if missing_strategy == "mean" and pd.api.types.is_numeric_dtype(cur_df[col]):
            cur_df[col] = cur_df[col].fillna(cur_df[col].mean())
        elif missing_strategy == "median" and pd.api.types.is_numeric_dtype(cur_df[col]):
            cur_df[col] = cur_df[col].fillna(cur_df[col].median())
        elif missing_strategy == "mode":
            mode_val = cur_df[col].mode()
            if not mode_val.empty:
                cur_df[col] = cur_df[col].fillna(mode_val.iloc[0])
        elif missing_strategy == "drop":
            cur_df.dropna(subset=[col], inplace=True)

    cur_df.to_csv(_upload_path("cleaned_dataset.csv"), index=False)
    with open(_upload_path("current_df.pkl"), "wb") as f:
        pickle.dump(cur_df, f)

    with open(_upload_path("original_df.pkl"), "rb") as f:
        original_df = pickle.load(f)

    differences = {}
    for col in cur_df.columns:
        if col in original_df.columns:
            orig_vals = original_df[col].astype(str).tolist()
            new_vals = cur_df[col].astype(str).tolist()
            if orig_vals != new_vals:
                differences[col] = {"original": orig_vals[:10], "new": new_vals[:10]}

    generate_pdf_from_differences(differences)
    session["preprocessing_done"] = True
    flash("Preprocessing completed successfully.")
    return render_template(
        "upload.html",
        table_html=cur_df.head(10).to_html(classes="table table-bordered", index=False),
        column_names=cur_df.columns.tolist(),
        differences=differences,
        preprocessing_done=True,
    )


@app.route("/download")
def download_file():
    return send_file(_upload_path("cleaned_dataset.csv"), as_attachment=True)


@app.route("/download_changes_pdf")
def download_changes_pdf():
    return send_file(_upload_path("differences.pdf"), as_attachment=True)


@app.route("/download_results_pdf")
def download_results_pdf():
    output_path = _upload_path("results_report.pdf")
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [Paragraph("Model Results Report", styles["Title"]), Spacer(1, 12)]

    best_model = request.args.get("best_model", "N/A")
    elements.append(Paragraph(f"<b>Best Model:</b> {best_model}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    for label, fname in [("Model Score Comparison:", "model_scores.png"), ("Correlation Heatmap:", "heatmap.png")]:
        fpath = STATIC_FOLDER / fname
        if fpath.exists():
            elements.append(Paragraph(f"<b>{label}</b>", styles["Heading2"]))
            elements.append(Spacer(1, 6))
            elements.append(PdfImage(str(fpath), width=400, height=300))
            elements.append(Spacer(1, 16))

    doc.build(elements)
    return send_file(output_path, as_attachment=True)


@app.route("/train_models", methods=["POST"])
def train_models():
    with open(_upload_path("current_df.pkl"), "rb") as f:
        cur_df = pickle.load(f)

    model_type = request.form["model_type"]
    target_col = request.form["target"]
    X = cur_df.drop(columns=[target_col]).select_dtypes(include=["number"])
    y = cur_df[target_col]

    if X.empty:
        return "Error: No numeric features available for model training."

    if model_type in ["classification", "regression"]:
        from lazypredict.Supervised import LazyClassifier, LazyRegressor

        if model_type == "classification":
            task = "classification"
            if y.dtype == "object":
                y = LabelEncoder().fit_transform(y.astype(str))
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            models, _ = LazyClassifier(verbose=0, ignore_warnings=True).fit(X_tr, X_te, y_tr, y_te)
            score_col = "Accuracy"
        else:
            task = "regression"
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            models, _ = LazyRegressor(verbose=0, ignore_warnings=True).fit(X_tr, X_te, y_tr, y_te)
            score_col = "R-Squared"
    elif model_type == "clustering":
        task = "clustering"
        scaled = StandardScaler().fit_transform(X)
        scores = {}
        for name, model in {
            "KMeans": KMeans(n_clusters=3, random_state=42, n_init="auto"),
            "DBSCAN": DBSCAN(),
            "Agglomerative": AgglomerativeClustering(n_clusters=3),
        }.items():
            labels = model.fit_predict(scaled)
            scores[name] = silhouette_score(scaled, labels) if len(set(labels)) > 1 else -1.0
        models = pd.DataFrame(list(scores.items()), columns=["Model", "Score"]).set_index("Model")
        score_col = "Score"
    else:
        return f"Unsupported model type: {model_type}"

    chart_data = models[[score_col]].dropna()
    plt.figure(figsize=(9, 5))
    sns.barplot(x=chart_data[score_col], y=chart_data.index, palette="viridis")
    plt.title(f"{task.title()} Model Performance")
    plt.tight_layout()
    plt.savefig(STATIC_FOLDER / "model_scores.png")
    plt.close()

    numeric_df = cur_df.select_dtypes(include=["number"])
    heatmap_path = None
    if not numeric_df.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="BuPu", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(STATIC_FOLDER / "heatmap.png")
        plt.close()
        heatmap_path = "static/heatmap.png"

    best_model = chart_data[score_col].idxmax() if not chart_data.empty else "N/A"
    models_out = models.reset_index()
    return render_template(
        "results.html",
        scores=models_out.round(3).to_html(classes="table table-striped", index=False),
        best_model=best_model,
        task=task,
        chart_path="static/model_scores.png",
        heatmap_path=heatmap_path,
        model_type=model_type,
    )


load_knowledge_base()

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=int(os.environ.get("PORT", 5050)))
