# NavDoc AI - Medical Document Analyzer

A comprehensive AI-powered medical document analyzer that extracts, summarizes, and translates medical reports into multiple languages with intelligent medical insights.

## Features

- **Medical Document Analysis**: Intelligent extraction of key medical information with normal value ranges
- **Multi-language Support**: Translation to 12+ languages including Hindi, Spanish, French, German, Arabic, Chinese
- **OCR Processing**: Handles both text-based and image-based PDFs using Tesseract OCR
- **Document Type Detection**: Automatically identifies medical, legal, financial, or general documents
- **Human-readable Summaries**: Converts complex medical data into understandable insights
- **Comprehensive Analysis**: Detailed test result interpretation with clinical significance

## Live Demo

**GitHub Codespaces Demo**: Click "Code" → "Create codespace on main" for instant cloud deployment

**GitHub Pages Demo**: [https://vikashmehta292511.github.io/NavDoc-AI](https://vikashmehta292511.github.io/NavDoc-AI)

## Quick Start Options

### Method 1: For GitHub Codespaces
1. Click the "Code" button above
2. Select "Codespaces" → "Create codespace on main"
3. Wait for environment setup (3-5 minutes)
4. Run: `uvicorn medical_ai_backend:app --host 0.0.0.0 --port 8000`
5. Open forwarded port and navigate to index.html

### Method 2: Local Installation
```bash
# Clone repository
git clone https://github.com/vikashmehta292511/NavDoc-AI.git
cd NavDoc-AI

# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# macOS: brew install tesseract
# Ubuntu: sudo apt install tesseract-ocr

# Run application
uvicorn medical_ai_backend:app --reload

# Open index.html in your browser
```

### Method 3: Windows One-Click Setup
1. Download or clone this repository
2. Double-click `setup.bat`
3. Follow automated installation prompts
4. Application launches automatically

## System Requirements

- **RAM**: 8GB minimum (16GB recommended for translation features)
- **Storage**: 4GB free space for AI model downloads with all required dependencies
- **Python**: 3.8 or higher
- **Internet**: Required for initial AI model downloads
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+

## Supported Languages

- English (Native)
- Hindi, Marathi, Nepali
- Spanish, French, German, Portuguese
- Arabic, Chinese, Japanese, Korean

## Medical Analysis Capabilities

### Supported Test Types
- Complete Blood Count (CBC)
- Liver Function Tests (LFT)
- Kidney Function Tests (KFT)
- Lipid Profiles
- Thyroid Function Tests
- Blood Glucose Tests
- Electrolyte Panels

### Analysis Features
- Automatic abnormal value detection
- Normal range comparisons
- Clinical significance explanations
- Human-readable medical interpretations
- Structured result presentations

## API Documentation

### Health Check Endpoint
```
GET /health
Response: System status, available models, supported languages
```

### Document Analysis Endpoint
```
POST /analyze
Content-Type: multipart/form-data

Parameters:
- file: PDF document (required)
- language: Target language for summary (required)

Response: Comprehensive analysis with summary and recommendations
```

## Architecture

- **Backend**: FastAPI with transformer-based AI models
- **Summarization**: BART/DistilBART for content summarization
- **Translation**: mBART + Marian MT models for multilingual support  
- **OCR**: Tesseract for image-based PDF processing
- **Frontend**: Vanilla HTML/CSS/JavaScript

## Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with debug logging
uvicorn medical_ai_backend:app --reload --log-level debug

# Development server with auto-reload
python -m uvicorn medical_ai_backend:app --reload --host 0.0.0.0
```

## Troubleshooting

### Common Issues

**Models downloading slowly**
- First run downloads approximately 4GB of AI models
- Ensure stable internet connection
- Consider using faster internet for initial setup

**Tesseract not found error**
- Install Tesseract OCR from official sources
- Verify installation path in medical_ai_backend.py
- Windows: Ensure installation to default directory

**Translation not working**
- Translation models are memory-intensive
- Requires 8GB+ RAM for full functionality
- Falls back to English if translation fails

**Memory issues during analysis**
- Close other applications to free RAM
- Process smaller PDF files if memory constraints exist

## Performance Notes - may vary depend on your system

- **First Run**: 5-15 minutes (model downloads + loading)
- **Subsequent Runs**: 1-2 minutes startup time
- **Analysis Speed**: 10-30 seconds per document depends on the page
- **Translation**: Additional 30-60 seconds for non-English languages

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit Pull Request


## Acknowledgments

- Hugging Face Transformers library
- Google PEGASUS and T5 models  
- Facebook BART and mBART models
- Helsinki-NLP Marian translation models
- Tesseract OCR community
- FastAPI framework

---
