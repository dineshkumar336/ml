"""
app.py — Flask API for Complaint Prioritization System
Run: python app.py
"""

import os, io, json
import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import your model
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_core import Config, ComplaintAnalyzer

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ── Load base model once at startup ──────────────────────────────────────────
config   = Config()
analyzer = ComplaintAnalyzer("best_model", config)
print("✅ Base model loaded successfully")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/single')
def single():
    return render_template('single.html')

@app.route('/batch')
def batch():
    return render_template('batch.html')

@app.route('/finetune')
def finetune():
    return render_template('finetune.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/predict/single', methods=['POST'])
def api_single():
    """
    Mode 1 — Single complaint prediction.
    POST JSON: { "sector": "...", "product": "...", "complaint": "..." }
    """
    try:
        data      = request.get_json()
        sector    = data.get('sector', '').strip()
        product   = data.get('product', '').strip()
        complaint = data.get('complaint', '').strip()

        if not all([sector, product, complaint]):
            return jsonify({"error": "sector, product and complaint are required"}), 400

        result = analyzer.predict_single(sector, product, complaint)

        return jsonify({
            "success":        True,
            "sector":         sector,
            "product":        product,
            "complaint":      complaint,
            "priority_label": result["priority_label"],
            "urgency_score":  result["urgency_score"],
            "confidence":     round(result["confidence"] * 100, 1),
            "urgency_pct":    round((result["urgency_score"] - 1) / 9 * 100, 1),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
def api_batch():
    """
    Mode 2 — Batch prediction from uploaded CSV.
    POST multipart/form-data with file field 'csv_file'
    Returns JSON summary + downloadable CSV.
    """
    try:
        if 'csv_file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Read CSV
        df = pd.read_csv(file)
        required = ['Sector', 'Product', 'Complaint_Narrative']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}. Required: Sector, Product, Complaint_Narrative"}), 400

        total = len(df)
        if total > 10000:
            return jsonify({"error": "Maximum 10,000 complaints per batch"}), 400

        # Run predictions
        results = []
        for _, row in df.iterrows():
            r = analyzer.predict_single(
                row['Sector'], row['Product'], row['Complaint_Narrative'])
            results.append(r)

        df['priority_label'] = [r['priority_label'] for r in results]
        df['urgency_score']  = [r['urgency_score']  for r in results]
        df['confidence_pct'] = [round(r['confidence'] * 100, 1) for r in results]
        df = df.sort_values('urgency_score', ascending=False).reset_index(drop=True)

        # Save output CSV
        out_path = os.path.join('outputs', 'batch_results.csv')
        df.to_csv(out_path, index=False)

        # Summary stats
        priority_counts = df['priority_label'].value_counts().to_dict()
        sector_summary  = df.groupby('Sector')['priority_label'].value_counts().unstack(fill_value=0).to_dict()

        # Top 10 most urgent as JSON
        top10 = df.head(10)[['Sector','Product','Complaint_Narrative',
                              'priority_label','urgency_score','confidence_pct']].to_dict(orient='records')

        return jsonify({
            "success":         True,
            "total":           total,
            "priority_counts": priority_counts,
            "mean_urgency":    round(df['urgency_score'].mean(), 2),
            "top10":           top10,
            "download_ready":  True,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/download/batch', methods=['GET'])
def download_batch():
    """Download the last batch results CSV."""
    path = os.path.join('outputs', 'batch_results.csv')
    if not os.path.exists(path):
        return jsonify({"error": "No batch results available. Run batch prediction first."}), 404
    return send_file(path, as_attachment=True, download_name='prioritized_complaints.csv')


@app.route('/api/finetune', methods=['POST'])
def api_finetune():
    """
    Mode 3 — Fine-tune model on company's labeled data.
    POST multipart/form-data with:
        - labeled_csv : CSV with Sector,Product,Complaint_Narrative,Priority_Level,Urgency_Score
        - company_name: str (used as model folder name)
        - freeze_bert : "true"/"false"
        - epochs      : int
    """
    try:
        if 'labeled_csv' not in request.files:
            return jsonify({"error": "No labeled CSV uploaded"}), 400

        file         = request.files['labeled_csv']
        company_name = request.form.get('company_name', 'company').strip().replace(' ', '_')
        freeze_bert  = request.form.get('freeze_bert', 'true').lower() == 'true'
        epochs       = int(request.form.get('epochs', 3))

        df = pd.read_csv(file)
        required = ['Sector','Product','Complaint_Narrative','Priority_Level','Urgency_Score']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        n = len(df)
        if n < 30:
            return jsonify({"error": f"Need at least 30 labeled samples, got {n}"}), 400

        # Save uploaded file
        csv_path = os.path.join('uploads', f'{company_name}_labeled.csv')
        df.to_csv(csv_path, index=False)

        # Run fine-tuning in background
        import subprocess, sys
        output_dir = f'{company_name}_model'
        freeze_flag = '--freeze_bert' if freeze_bert else ''
        cmd = (f'{sys.executable} step5_finetune.py '
               f'--company_data {csv_path} '
               f'--output {output_dir} '
               f'--epochs {epochs} '
               f'{freeze_flag}')

        subprocess.Popen(cmd, shell=True)

        return jsonify({
            "success":     True,
            "message":     f"Fine-tuning started for {company_name}. This takes 5–15 minutes.",
            "samples":     n,
            "epochs":      epochs,
            "freeze_bert": freeze_bert,
            "model_name":  output_dir,
            "status":      "training"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "loaded"})


if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=5000)