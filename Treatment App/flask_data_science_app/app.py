import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import pickle
from flask import Flask, request, redirect, render_template, flash, send_file, session
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import Image

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'json'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'secret'


@app.route('/')
def index():
    session['preprocessing_done'] = False
    return render_template('index.html')

def drop_duplicates(df, subset_name):
    df.drop_duplicates(subset=[subset_name], inplace=True)
    return df

def outlier_handling(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    return df[(df[column] >= q1 - 1.5 * iqr) & (df[column] <= q3 + 1.5 * iqr)]

def date_formatting(df, column):
    try:
        df[column] = pd.to_datetime(df[column], format='%m/%d/%Y')
    except Exception:
        try:
            df[column] = pd.to_datetime(df[column])  # fallback generic parse
        except Exception as e:
            print(f"Could not convert column {column} to datetime: {e}")
    return df


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.context_processor
def inject_preprocessing_done():
    return dict(preprocessing_done=session.get('preprocessing_done', False))


def generate_pdf_from_differences(differences, output_path='uploads/differences.pdf'):
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title = Paragraph("Data Preprocessing Changes", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    if not differences:
        elements.append(Paragraph("No differences were detected.", styles["Normal"]))
    else:
        for col, diff in differences.items():
            elements.append(Paragraph(f"<b>Column:</b> {col}", styles["Heading4"]))
            elements.append(Spacer(1, 6))

            # Prepare table data
            data = [["Original", "New"]]
            max_len = max(len(diff['original']), len(diff['new']))
            for i in range(max_len):
                orig = diff['original'][i] if i < len(diff['original']) else ""
                new = diff['new'][i] if i < len(diff['new']) else ""
                data.append([str(orig), str(new)])

            table = Table(data, repeatRows=1, colWidths=[250, 250])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ]))

            elements.append(table)
            elements.append(Spacer(1, 20))

    doc.build(elements)


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    table_html = None

    if request.method == 'POST':

        for file_name in ['cleaned_dataset.csv', 'differences.csv', 'differences.pdf', 'current_df.pkl', 'original_df.pkl']:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load the file into a DataFrame
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                elif filename.endswith('.json'):
                    df = pd.read_json(filepath)
                else:
                    df = pd.DataFrame()

                with open('uploads/current_df.pkl', 'wb') as f:
                    pickle.dump(df, f)

                with open('uploads/original_df.pkl', 'wb') as f:
                    pickle.dump(df, f)

                table_html = df.head(10).to_html(classes='', header="true", index=False)
                table_html = table_html.replace('<table border="1" class="dataframe">', '').replace('</table>', '')
                table_html = table_html.replace('<th>', '<th style="padding:12px; justify-content: left;">')
                table_html = table_html.replace('<td>', '<td style="padding:12px;">')

                flash('File uploaded and read successfully!')
                session['preprocessing_done'] = False
            except Exception as e:
                flash(f'Error reading file: {e}')

    try:
        column_names = df.columns.tolist()
    except NameError:
        column_names = []
    return render_template('upload.html', table_html=table_html, column_names=column_names)


@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        # Load dataframe
        with open('uploads/current_df.pkl', 'rb') as f:
            df = pickle.load(f)

        # Step 1: Detect ID columns (likely unique values)
        id_columns = [col for col in df.columns if df[col].is_unique and df[col].nunique() == len(df)]

        # Step 2: Detect binary columns
        binary_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns
                       if set(df[col].dropna().unique()).issubset({0, 1})]

        # Step 3: Drop duplicates if ID column exists
        if id_columns:
            df = drop_duplicates(df, id_columns[0])

        # Step 4: Handle missing values
        for col in df.columns:
            if col in id_columns:
                continue

            missing_ratio = df[col].isnull().mean()

            if missing_ratio > 0.5:
                df.drop(columns=[col], inplace=True)
                continue

            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    if col not in binary_cols:
                        # Use median if outliers exist
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        has_outliers = ((df[col] < lower) | (df[col] > upper)).any()
                        value = df[col].median() if has_outliers else df[col].mean()
                        df[col] = df[col].fillna(value)
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])

        # Step 5: Handle outliers (remove them from numeric columns except binary)
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if col not in binary_cols and col not in id_columns:
                df = outlier_handling(df, col)

        # Step 6: Format date columns
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    sample = df[col].dropna().iloc[0]
                    if isinstance(pd.to_datetime(sample, errors='coerce'), pd.Timestamp):
                        df = date_formatting(df, col)
                except Exception:
                    continue

        # Step 7: Scaling (StandardScaler) for numeric non-binary columns
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cols_to_scale = [col for col in numeric_cols if col not in binary_cols + id_columns]
        
        if cols_to_scale:
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

        # Step 8: Encoding categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Drop categorical columns with too many unique values (e.g., names, IDs)
        categorical_cols = [col for col in categorical_cols if df[col].nunique() <= 20]
        exclude_cols = ['name', 'surname', 'id', 'email']
        categorical_cols = [col for col in categorical_cols if col.lower() not in exclude_cols]


        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


        # Save updated DataFrame
        with open('uploads/current_df.pkl', 'wb') as f:
            pickle.dump(df, f)

        df.to_csv('uploads/cleaned_dataset.csv', index=False)

        with open('uploads/changes_df.pkl', 'wb') as f:
            pickle.dump(df, f)

        table_html = df.head(10).to_html(classes='', header="true", index=False)
        table_html = table_html.replace('<table border="1" class="dataframe">', '').replace('</table>', '')
        table_html = table_html.replace('<th>', '<th style="padding:12px; justify-content: left;">')
        table_html = table_html.replace('<td>', '<td style="padding:12px;">')

        # Compare with original to detect changes
        try:
            with open('uploads/original_df.pkl', 'rb') as f:
                original_df = pickle.load(f)

            differences = {}
            for col in df.columns:
                if col in original_df.columns:
                    orig_vals = original_df[col].astype(str).tolist()
                    new_vals = df[col].astype(str).tolist()
                    if orig_vals != new_vals:
                        differences[col] = {
                            "original": orig_vals[:10],
                            "new": new_vals[:10]
                        }

            # Save differences as CSV
            changes_list = [
                {"Column": col, "Original": str(diff["original"]), "New": str(diff["new"])}
                for col, diff in differences.items()
            ]
            pd.DataFrame(changes_list).to_csv('uploads/differences.csv', index=False)

            # Save differences as PDF
            generate_pdf_from_differences(differences)

        except Exception as e:
            differences = {}
            flash(f"Could not generate differences file: {e}")

        session['preprocessing_done'] = True
        flash('Preprocessing completed successfully.')

    except Exception as e:
        flash(f"Error during preprocessing: {e}")
        table_html = None
        df = pd.DataFrame()

    return render_template('upload.html', table_html=table_html, column_names=df.columns.tolist(), differences=differences,  preprocessing_done=session.get('preprocessing_done', False))


@app.route('/download')
def download_file():
    if not session.get('preprocessing_done', False):
        flash('Preprocessing has not been completed yet.')
        return redirect('/upload')
    path = 'uploads/cleaned_dataset.csv'
    return send_file(path, as_attachment=True)

@app.route('/download-differences')
def download_differences():
    if not session.get('preprocessing_done', False):
        flash('Preprocessing has not been completed yet.')
        return redirect('/upload')
    path = 'uploads/differences.csv'
    return send_file(path, as_attachment=True)

@app.route('/download_changes')
def download_changes():
    path = 'uploads/differences.csv'
    return send_file(path, as_attachment=True)

@app.route('/download_changes_pdf')
def download_changes_pdf():
    path = 'uploads/differences.pdf'
    return send_file(path, as_attachment=True)

@app.route('/download_results_pdf')
def download_results_pdf():
    output_path = 'uploads/results_report.pdf'
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title = Paragraph("Model Results Report", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Best Model
    best_model = request.args.get('best_model', 'N/A')
    elements.append(Paragraph(f"<b>Best Model:</b> {best_model}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Model Score Comparison Chart
    chart_path = 'static/model_scores.png'
    if os.path.exists(chart_path):
        elements.append(Paragraph("<b>Model Score Comparison:</b>", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        elements.append(Image(chart_path, width=400, height=300))
        elements.append(Spacer(1, 20))

    # Correlation Heatmap
    heatmap_path = 'static/heatmap.png'
    if os.path.exists(heatmap_path):
        elements.append(Paragraph("<b>Correlation Heatmap:</b>", styles["Heading2"]))
        elements.append(Spacer(1, 6))
        elements.append(Image(heatmap_path, width=400, height=300))
        elements.append(Spacer(1, 20))

    # Build the PDF
    doc.build(elements)

    return send_file(output_path, as_attachment=True)



@app.route('/train_models', methods=['POST'])
def train_models():
    model_type = request.form['model_type']

    # Load cleaned dataset
    with open('uploads/current_df.pkl', 'rb') as f:
        df = pickle.load(f)

    target_col = request.form['target']

    # Prepare features and label
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Drop non-numeric columns from X
    X = X.select_dtypes(include=['number'])

    if X.empty:
        return "Error: No numeric features available for model training. Please preprocess data properly."
    
    # Optional: filter features based on correlation with the target (only for regression/classification)
    if model_type in ['regression', 'classification'] and y.dtype != 'object':
        numeric_df = df.select_dtypes(include=[np.number])  # Keeps only numeric columns
        if target_col in numeric_df.columns:
            correlations = numeric_df.corr()[target_col].drop(target_col)
        else:
            correlations = pd.Series(dtype=float)

        high_corr_features = correlations[correlations.abs() >= 0.1].index.tolist()
        X = X[high_corr_features]

    if model_type == 'classification':
        task = 'classification'
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y.astype(str))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = LazyClassifier(verbose=0, ignore_warnings=True)
        models, _ = clf.fit(X_train, X_test, y_train, y_test)
        score_col = 'Accuracy'

    elif model_type == 'regression':
        task = 'regression'
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = LazyRegressor(verbose=0, ignore_warnings=True)
        models, _ = clf.fit(X_train, X_test, y_train, y_test)
        score_col = 'R-Squared'

    elif model_type == 'clustering':
        task = 'clustering'
        models = {}
        clustering_algorithms = {
            'KMeans': KMeans(n_clusters=3, random_state=42),
            'DBSCAN': DBSCAN(),
            'Agglomerative': AgglomerativeClustering(n_clusters=3)
        }
        for name, model in clustering_algorithms.items():
            try:
                labels = model.fit_predict(X)
                if len(set(labels)) > 1:
                    score = silhouette_score(X, labels)
                    models[name] = score
                else:
                    models[name] = -1  # silhouette_score needs at least 2 clusters
            except Exception as e:
                models[name] = None

        # Convert to DataFrame for consistency
        models = pd.DataFrame(list(models.items()), columns=['Model', 'Score']).set_index('Model')
        score_col = 'Score'

    else:
        return f"Unsupported model type: {model_type}"

    # Filter and chart
    chart_data = models[[score_col]].dropna()
    chart_data = chart_data[chart_data[score_col] >= 0]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=chart_data[score_col], y=chart_data.index, palette='viridis')
    plt.title(f'{task.title()} Model Performance')
    plt.xlabel(score_col)
    plt.ylabel('Model')
    plt.tight_layout()
    chart_path = 'static/model_scores.png'
    plt.savefig(chart_path)
    plt.close()

    # Generate heatmap
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='BuPu', fmt=".2f")
        plt.title("Correlation Heatmap")
        heatmap_path = 'static/heatmap.png'
        plt.savefig(heatmap_path)
        plt.close()
    else:
        heatmap_path = None

    # Get best model name
    if not chart_data.empty:
        best_model = chart_data[score_col].idxmax()
    else:
        best_model = "N/A"

    # Convert to HTML for rendering
    models.reset_index(inplace=True)  # Ensure 'Model' is part of the header row
    scores_html = models.round(3).to_html(classes='table table-striped', index=False)

    return render_template(
        'results.html',
        scores=scores_html,
        best_model=best_model,
        task=task,
        chart_path=chart_path,
        heatmap_path=heatmap_path,
        model_type=model_type
    )


if __name__ == '__main__':
    app.run(debug=True)
