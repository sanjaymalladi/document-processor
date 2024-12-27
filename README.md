---
title: Document Processor
emoji: 🐠
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
---
# 🏦 Appian Credit Union - Smart Document Processor AI

## 🎯 Problem Statement
Appian Credit Union receives thousands of PDF documents daily that need to be classified, verified, and organized. Our solution automates this process using AI, significantly reducing manual effort and processing time.

## 💡 Innovation Highlights
- 🤖 Hierarchical document classification system
- 👤 Intelligent person-document association
- 📊 Automated metadata extraction
- 🔄 Batch processing capabilities
- 🎨 Modern, intuitive UI

## 🎯 Document Types Supported
- 💳 Bank Account Applications
  - Credit Card Applications
  - Savings Account Applications
- 🪪 Identity Documents
  - Driver's License
  - State/Country ID
  - Passport
- 📊 Financial Documents
  - Income Statements
  - Paystubs
  - Tax Returns
- 🧾 Receipts

## 🛠️ Technical Architecture
- **Backend Framework**: Python + Flask
- **Document Processing**: PyPDF2
- **ML/AI Pipeline**: 
  - TF-IDF Vectorization
  - Naive Bayes Classification
  - Named Entity Recognition
- **Frontend**: HTML + JavaScript + Tailwind CSS
- **Database**: SQLite
- **Deployment**: Hugging Face Spaces

## ✨ Key Features

### 1. Hierarchical Classification
- Person-level document association using:
  - Name matching
  - Government ID recognition
  - Email address extraction
- Document type categorization
- Automatic grouping of similar documents

### 2. Information Extraction
- Automated extraction of:
  - Personal information
  - Financial data
  - Document dates
  - Account numbers
  - Government ID numbers

### 3. Processing Pipeline
- Batch document upload
- Real-time processing
- Error handling and validation
- Progress tracking
- Results summary

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.9+
pip
Virtual Environment (recommended)
```

### Installation
1. Clone the repository
```bash
git clone https://github.com/yourusername/appian-document-processor.git
cd appian-document-processor
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
python app.py
```

4. Access at `http://localhost:7860`

## 👥 Team Members
- Sanjay Malladi

## 📝 License
MIT License

## 🤝 Acknowledgments
- Appian AI Challenge Team
- IIT Madras
- Open Source Community

---
*Developed for the Appian AI Challenge 2024-25 at IIT Madras*