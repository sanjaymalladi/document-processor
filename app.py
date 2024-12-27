# app.py
from flask import Flask, request, jsonify, render_template_string
import PyPDF2
import sqlite3
from datetime import datetime
import re
import os
import hashlib
from typing import List, Dict
import shutil
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import joblib
import base64
from werkzeug.utils import secure_filename
import tempfile
# HTML template with embedded JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Processor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Additional custom styles can go here */
        .processing {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body class="bg-gray-50">
    <div class="container mx-auto p-6 max-w-4xl">
        <div class="mb-8">
            <h1 class="text-3xl font-bold mb-2">Smart Document Processor</h1>
            <p class="text-gray-600">Upload and analyze PDF documents with AI</p>
        </div>

        <!-- Upload Section -->
        <div class="mb-8">
            <div id="dropZone" class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors">
                <input type="file" multiple accept=".pdf" id="fileInput" class="hidden">
                <div class="cursor-pointer">
                    <svg class="w-12 h-12 text-gray-400 mx-auto mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
                    </svg>
                    <span class="text-lg mb-2 block">Drop PDFs here or click to upload</span>
                    <span class="text-sm text-gray-500">Supports multiple files</span>
                </div>
            </div>
        </div>

        <!-- File List -->
        <div id="fileList" class="mb-8 hidden">
            <h2 class="text-xl font-semibold mb-4">Selected Files</h2>
            <div id="fileListContent" class="space-y-2"></div>
            <button id="processButton" class="mt-4 bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50">
                Process Documents
            </button>
        </div>

        <!-- Results Section -->
        <div id="results" class="space-y-4"></div>

        <!-- Error Alert -->
        <div id="error" class="hidden mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded"></div>
    </div>

    <script>
        let files = [];
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const fileList = document.getElementById('fileList');
        const fileListContent = document.getElementById('fileListContent');
        const processButton = document.getElementById('processButton');
        const resultsDiv = document.getElementById('results');
        const errorDiv = document.getElementById('error');

        // Drag and drop handlers
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('border-blue-500');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('border-blue-500');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-blue-500');
            handleFiles(e.dataTransfer.files);
        });

        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });

        function handleFiles(uploadedFiles) {
            files = Array.from(uploadedFiles).filter(file => file.name.toLowerCase().endsWith('.pdf'));
            updateFileList();
        }

        function updateFileList() {
            if (files.length > 0) {
                fileList.classList.remove('hidden');
                fileListContent.innerHTML = files.map((file, index) => `
                    <div class="flex items-center p-3 bg-gray-50 rounded">
                        <svg class="w-5 h-5 text-gray-500 mr-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                        </svg>
                        <span>${file.name}</span>
                    </div>
                `).join('');
            } else {
                fileList.classList.add('hidden');
            }
        }

        processButton.addEventListener('click', async () => {
            if (files.length === 0) return;

            processButton.disabled = true;
            processButton.innerHTML = `
                <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white inline" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
            `;

            const formData = new FormData();
            files.forEach(file => {
                formData.append('files[]', file);
            });

            try {
                const response = await fetch('/batch_process', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                displayResults(data);
                errorDiv.classList.add('hidden');
            } catch (error) {
                errorDiv.textContent = 'Failed to process documents. Please try again.';
                errorDiv.classList.remove('hidden');
            } finally {
                processButton.disabled = false;
                processButton.textContent = 'Process Documents';
            }
        });

        function displayResults(results) {
            resultsDiv.innerHTML = results.map(result => `
                <div class="border rounded-lg p-4 bg-white shadow-sm">
                    <h3 class="font-medium mb-2">${result.result.filename}</h3>
                    <div class="grid grid-cols-2 gap-4">
                        <div>
                            <span class="text-gray-600">Type:</span>
                            <span class="ml-2">${result.result.doc_type}</span>
                        </div>
                        <div>
                            <span class="text-gray-600">Date:</span>
                            <span class="ml-2">${result.result.date || 'N/A'}</span>
                        </div>
                        <div>
                            <span class="text-gray-600">Amount:</span>
                            <span class="ml-2">${result.result.amount ? '$' + result.result.amount.toFixed(2) : 'N/A'}</span>
                        </div>
                        <div>
                            <span class="text-gray-600">Person:</span>
                            <span class="ml-2">${result.result.person_name}</span>
                        </div>
                    </div>
                </div>
            `).join('');
        }
    </script>
</body>
</html>
"""

class MLDocumentClassifier:
    def __init__(self):
        self.labels = ['Invoice', 'Statement', 'Contract', 'Receipt', 'Report', 'Letter', 'Form']
        self.classifier = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=10000)),
            ('clf', MultinomialNB())
        ])
        self.is_trained = False
        
    def predict(self, text):
        return self._rule_based_classify(text)
    
    def _rule_based_classify(self, text):
        text_lower = text.lower()
        rules = [
            ('Invoice', ['invoice', 'bill', 'payment due', 'amount due']),
            ('Statement', ['statement', 'balance', 'transaction history']),
            ('Contract', ['contract', 'agreement', 'terms and conditions']),
            ('Receipt', ['receipt', 'purchased', 'payment received']),
            ('Report', ['report', 'analysis', 'findings']),
            ('Letter', ['dear', 'sincerely', 'regards']),
            ('Form', ['form', 'please fill', 'application'])
        ]
        
        scores = []
        for doc_type, keywords in rules:
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores.append((doc_type, score / len(keywords) if keywords else 0))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

class EnhancedDocProcessor:
    def __init__(self):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.setup_database()
        self.classifier = MLDocumentClassifier()
        
    def setup_database(self):
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                filename TEXT,
                doc_type TEXT,
                person_name TEXT,
                amount REAL,
                date TEXT,
                account_number TEXT,
                raw_text TEXT,
                processed_date TEXT,
                file_hash TEXT,
                version INTEGER,
                user_id TEXT
            );
            
            CREATE TABLE IF NOT EXISTS similar_docs (
                doc_id INTEGER,
                similar_doc_id INTEGER,
                similarity_score REAL,
                FOREIGN KEY (doc_id) REFERENCES documents (id),
                FOREIGN KEY (similar_doc_id) REFERENCES documents (id)
            );
        ''')
        self.conn.commit()

    def extract_text(self, pdf_path: str) -> str:
        try:
            text_parts = []
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return "\n".join(text_parts)
        except Exception as e:
            return f"Error extracting text: {str(e)}"

    def extract_metadata(self, text: str) -> Dict:
        return {
            'amount': next((float(amt.replace('$','').replace(',','')) 
                          for amt in re.findall(r'\$[\d,]+\.?\d*', text)), 0.0),
            'date': next(iter(re.findall(r'\d{1,2}/\d{1,2}/\d{4}', text)), None),
            'account_number': next(iter(re.findall(r'Account\s*#?\s*:?\s*(\d{8,12})', text)), None),
            'person_name': next(iter(re.findall(r'(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', text)), "Unknown")
        }

    def process_document(self, pdf_path: str, filename: str, user_id: str = None) -> Dict:
        text = self.extract_text(pdf_path)
        doc_type = self.classifier.predict(text)
        metadata = self.extract_metadata(text)
        
        cursor = self.conn.execute('''
            INSERT INTO documents 
            (filename, doc_type, person_name, amount, date, 
             account_number, raw_text, processed_date, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename, doc_type, metadata['person_name'],
            metadata['amount'], metadata['date'],
            metadata['account_number'], text,
            datetime.now().isoformat(), user_id
        ))
        
        doc_id = cursor.lastrowid
        self.conn.commit()
        
        return {
            'id': doc_id,
            'filename': filename,
            'doc_type': doc_type,
            **metadata
        }

    def process_batch(self, file_paths: List[str], user_id: str = None) -> List[Dict]:
        results = []
        for file_path in file_paths:
            try:
                result = self.process_document(file_path, os.path.basename(file_path), user_id)
                results.append({"status": "success", "result": result, "file": file_path})
            except Exception as e:
                results.append({"status": "error", "error": str(e), "file": file_path})
        return results

app = Flask(__name__)
processor = EnhancedDocProcessor()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/batch_process', methods=['POST'])
def batch_process():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    user_id = request.form.get('user_id')
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = []
        for file in files:
            if file.filename.endswith('.pdf'):
                # Create a secure filename
                secure_name = secure_filename(file.filename)
                # Create full path in temporary directory
                temp_path = os.path.join(temp_dir, secure_name)
                file.save(temp_path)
                file_paths.append(temp_path)
        
        try:
            results = processor.process_batch(file_paths, user_id)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        # No need to manually clean up - TemporaryDirectory does it automatically
                
        return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)