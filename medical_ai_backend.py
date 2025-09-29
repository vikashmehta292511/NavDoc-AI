# medical_ai_backend.py - Enhanced version with better multilingual support and longer summaries
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast
import pdfplumber
import pytesseract
import os
from pdf2image import convert_from_bytes
import io
import logging
import re
import torch

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
summarizer = None
translator = None
tokenizer = None

# Force CPU usage to avoid GPU memory issues
device = "cpu"
torch.set_num_threads(2)  # Limit CPU threads

# Load summarization model with better parameters
try:
    logger.info("Loading summarization model...")
    # Try BART first as it's most reliable for medical text
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn",
        device=device
    )
    logger.info("BART summarization model loaded successfully")
except Exception as e:
    logger.error(f"BART load failed: {e}, trying smaller model")
    try:
        # Fallback to smaller model
        summarizer = pipeline(
            "summarization", 
            model="sshleifer/distilbart-cnn-12-6",
            device=device
        )
        logger.info("DistilBART summarization model loaded successfully")
    except Exception as e2:
        logger.error(f"All summarization models failed: {e2}")
        summarizer = None

# Load translation model with better error handling
try:
    logger.info("Loading MBart translation model (this may take time)...")
    translator = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt",
        low_cpu_mem_usage=True
    )
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    translator.to(device)
    logger.info("MBart translation model loaded successfully")
except Exception as e:
    logger.error(f"Translation model failed to load: {e}")
    logger.info("Attempting to load smaller translation model...")
    try:
        # Try smaller multilingual model
        from transformers import MarianMTModel, MarianTokenizer
        # This will be loaded dynamically per language pair
        translator = "marian"  # Flag for dynamic loading
        tokenizer = "marian"
        logger.info("Marian translation system ready (dynamic loading)")
    except Exception as e2:
        logger.error(f"All translation models failed: {e2}")
        translator = None
        tokenizer = None

# Language codes 
LANG_CODE = {
    "english": "en_XX",
    "hindi": "hi_IN", 
    "marathi": "mr_IN",
    "nepali": "ne_NP",
    "spanish": "es_XX",
    "french": "fr_XX",
    "german": "de_DE",
    "portuguese": "pt_XX",
    "arabic": "ar_AR",
    "chinese": "zh_CN",
    "japanese": "ja_XX",
    "korean": "ko_KR"
}

# Marian language pairs (for fallback translation)
MARIAN_MODELS = {
    "hindi": "Helsinki-NLP/opus-mt-en-hi",
    "spanish": "Helsinki-NLP/opus-mt-en-es", 
    "french": "Helsinki-NLP/opus-mt-en-fr",
    "german": "Helsinki-NLP/opus-mt-en-de",
    "portuguese": "Helsinki-NLP/opus-mt-en-pt",
    "chinese": "Helsinki-NLP/opus-mt-en-zh",
}

# Document type detection
def detect_document_type(text):
    text_lower = text.lower()
    medical_keywords = [
        'patient name', 'blood', 'urine', 'test', 'laboratory', 'medical', 'doctor', 
        'hospital', 'clinic', 'diagnosis', 'treatment', 'medication', 'hemoglobin', 
        'glucose', 'cholesterol', 'creatinine', 'bilirubin', 'liver function', 
        'kidney function', 'cbc', 'complete blood count', 'lipid profile', 'thyroid'
    ]
    
    medical_score = sum(1 for k in medical_keywords if k in text_lower)
    
    if medical_score >= 2:  # Lower threshold for detection
        return "medical"
    else:
        return "general"

# Enhanced medical insights extraction
def extract_comprehensive_medical_insights(text):
    insights = []
    
    # Patient information
    patient_match = re.search(r'Patient Name\s*:?\s*([^\n]+)', text, re.IGNORECASE)
    if patient_match:
        insights.append(f"Patient: {patient_match.group(1).strip()}")
    
    age_match = re.search(r'Age[/\s]*Gender\s*:?\s*([^\n]+)', text, re.IGNORECASE)
    if age_match:
        insights.append(f"Demographics: {age_match.group(1).strip()}")
    
    date_match = re.search(r'(?:Collection Date|Test Date|Date)[/\s]*:?\s*([^\n]+)', text, re.IGNORECASE)
    if date_match:
        insights.append(f"Test Date: {date_match.group(1).strip()}")
    
    # Comprehensive test analysis
    test_explanations = {
        'hemoglobin': ('Hemoglobin', 'carries oxygen in blood', 'g/dl', (13.0, 17.0)),
        'hematocrit': ('Hematocrit', 'percentage of red blood cells', '%', (40, 50)),
        'glucose': ('Blood Glucose', 'blood sugar levels', 'mg/dl', (70, 140)),
        'cholesterol': ('Total Cholesterol', 'fat levels in blood', 'mg/dl', (0, 200)),
        'creatinine': ('Creatinine', 'kidney function indicator', 'mg/dl', (0.9, 1.3)),
        'bilirubin': ('Bilirubin', 'liver function marker', 'mg/dl', (0.3, 1.2)),
        'urea': ('Blood Urea', 'kidney waste filtration', 'mg/dl', (17, 55)),
        'sgot': ('SGOT/AST', 'liver enzyme indicating liver health', 'U/L', (0, 50)),
        'sgpt': ('SGPT/ALT', 'liver enzyme for liver damage detection', 'U/L', (17, 63)),
        'ggtp': ('GGTP', 'liver enzyme for bile duct problems', 'U/L', (7, 50)),
        'albumin': ('Albumin', 'protein made by liver', 'g/dl', (3.5, 5.0)),
        'protein': ('Total Protein', 'overall protein levels', 'g/dl', (6.5, 8.1)),
        'sodium': ('Sodium', 'electrolyte balance', 'mmol/L', (136, 144)),
        'potassium': ('Potassium', 'heart and muscle function', 'mmol/L', (3.6, 5.1)),
        'calcium': ('Calcium', 'bone and muscle health', 'mg/dl', (8.9, 10.3))
    }
    
    findings = []
    normal_findings = []
    
    # Enhanced pattern matching for test results
    for test_key, (test_name, explanation, unit, (low, high)) in test_explanations.items():
        # More flexible pattern to catch various formats
        patterns = [
            rf'{test_key}[^0-9]*(\d+\.?\d*)',  # Basic pattern
            rf'{test_name}[^0-9]*(\d+\.?\d*)',  # Full name pattern
            rf'{test_key}.*?(\d+\.?\d*)\s*{unit}',  # With units
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match.group(1))
                    
                    # Determine if abnormal
                    if value < low:
                        status = f"LOW (normal: {low}-{high} {unit})"
                        findings.append(f"{test_name}: {value} {unit} - {status}")
                        findings.append(f"  This measures {explanation}. Low levels may need medical attention.")
                    elif value > high:
                        status = f"HIGH (normal: {low}-{high} {unit})"
                        findings.append(f"{test_name}: {value} {unit} - {status}")
                        findings.append(f"  This measures {explanation}. High levels may indicate health concerns.")
                    else:
                        normal_findings.append(f"{test_name}: {value} {unit} - Normal")
                    
                    break  # Found match, don't check other patterns
                except ValueError:
                    continue
    
    # Add findings to insights
    if findings:
        insights.append("ABNORMAL TEST RESULTS:")
        insights.extend(findings[:8])  # Show top 8 abnormal findings
        
    if normal_findings and len(findings) < 5:  # Show some normal results if few abnormal
        insights.append("NORMAL TEST RESULTS:")
        insights.extend(normal_findings[:3])
    
    # Extract interpretations and recommendations
    interp_patterns = [
        r'interpretation[^:]*:(.+?)(?=\n\n|\*\*\*|page|laboratory|$)',
        r'conclusion[^:]*:(.+?)(?=\n\n|\*\*\*|page|laboratory|$)',
        r'remarks?[^:]*:(.+?)(?=\n\n|\*\*\*|page|laboratory|$)',
        r'clinical significance[^:]*:(.+?)(?=\n\n|\*\*\*|page|laboratory|$)'
    ]
    
    for pattern in interp_patterns:
        matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            interp = match.group(1).strip()
            # Clean up the interpretation
            interp = re.sub(r'\s+', ' ', interp)
            if 20 < len(interp) < 500:  # Reasonable length
                insights.append(f"MEDICAL INTERPRETATION: {interp}")
                break
    
    return insights

# Enhanced human-readable summary
def create_comprehensive_medical_summary(text, doc_type):
    if doc_type == "medical":
        insights = extract_comprehensive_medical_insights(text)
        if insights:
            # Create detailed medical summary
            summary_parts = [
                "MEDICAL REPORT ANALYSIS:",
                "",  # Empty line for formatting
            ]
            
            # Group insights by type
            patient_info = [i for i in insights if any(keyword in i.lower() for keyword in ['patient:', 'demographics:', 'test date:'])]
            abnormal_results = []
            normal_results = []
            interpretations = []
            
            current_section = None
            for insight in insights:
                if "ABNORMAL TEST RESULTS:" in insight:
                    current_section = "abnormal"
                    continue
                elif "NORMAL TEST RESULTS:" in insight:
                    current_section = "normal"
                    continue
                elif "MEDICAL INTERPRETATION:" in insight:
                    interpretations.append(insight)
                    continue
                
                if current_section == "abnormal":
                    abnormal_results.append(insight)
                elif current_section == "normal":
                    normal_results.append(insight)
                elif insight not in patient_info:
                    patient_info.append(insight)
            
            # Build comprehensive summary
            if patient_info:
                summary_parts.extend(patient_info)
                summary_parts.append("")
            
            if abnormal_results:
                summary_parts.append("KEY FINDINGS REQUIRING ATTENTION:")
                summary_parts.extend(abnormal_results)
                summary_parts.append("")
            
            if normal_results:
                summary_parts.append("NORMAL TEST RESULTS:")
                summary_parts.extend(normal_results[:5])  # Limit normal results
                summary_parts.append("")
            
            if interpretations:
                summary_parts.extend(interpretations)
                summary_parts.append("")
            
            summary_parts.extend([
                "IMPORTANT RECOMMENDATIONS:",
                "Discuss these results with your healthcare provider",
                "Follow up on any abnormal values as recommended", 
                "Maintain regular health monitoring",
            ])
            
            return "\n".join(summary_parts)
        else:
            return "This appears to be a medical report. Please consult with your healthcare provider for proper interpretation of the test results and findings."
    else:
        return None  # Use AI summarization for non-medical documents

# Enhanced translation functions 
def translate_with_mbart(text, target_language):
    """Translate using mBART model"""
    if not translator or translator == "marian":
        return text, False
    
    target_code = LANG_CODE.get(target_language)
    if not target_code:
        return text, False
    
    try:
        # Split long text into chunks for translation
        max_length = 400  # Conservative length
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        translated_chunks = []
        
        for chunk in chunks:
            tokenizer.src_lang = "en_XX"
            inputs = tokenizer(chunk, return_tensors="pt", max_length=512, truncation=True)
            
            generated_tokens = translator.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[target_code],
                max_length=600,
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
            
            translated_chunk = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            translated_chunks.append(translated_chunk)
        
        final_translation = " ".join(translated_chunks)
        return final_translation, True
        
    except Exception as e:
        logger.error(f"mBART translation failed: {e}")
        return text, False

def translate_with_marian(text, target_language):
    """Fallback translation using Marian models"""
    model_name = MARIAN_MODELS.get(target_language)
    if not model_name:
        return text, False
    
    try:
        from transformers import MarianMTModel, MarianTokenizer
        
        logger.info(f"Loading Marian model for {target_language}...")
        marian_tokenizer = MarianTokenizer.from_pretrained(model_name)
        marian_model = MarianMTModel.from_pretrained(model_name)
        
        # Split text into smaller chunks
        max_length = 300
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        translated_chunks = []
        for chunk in chunks:
            inputs = marian_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
            generated = marian_model.generate(**inputs, max_length=400, num_beams=3)
            translated = marian_tokenizer.decode(generated[0], skip_special_tokens=True)
            translated_chunks.append(translated)
        
        return " ".join(translated_chunks), True
        
    except Exception as e:
        logger.error(f"Marian translation failed: {e}")
        return text, False

def translate_text(text, target_language):
    """Enhanced translation with multiple fallbacks"""
    if target_language == "english":
        return text, True
    
    logger.info(f"Attempting translation to {target_language}")
    
    # Try mBART first
    if translator and translator != "marian":
        translated, success = translate_with_mbart(text, target_language)
        if success:
            logger.info("Translation successful with mBART")
            return translated, True
    
    # Fallback to Marian for supported languages
    if target_language in MARIAN_MODELS:
        translated, success = translate_with_marian(text, target_language)
        if success:
            logger.info(f"Translation successful with Marian for {target_language}")
            return translated, True
    
    # If all translation fails
    logger.warning(f"Translation to {target_language} failed, returning English")
    return text, False

# PDF text extraction 
def extract_text_from_pdf(contents: bytes) -> str:
    text = ""
    try:
        logger.info("Extracting text with pdfplumber...")
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    logger.info(f"Extracted {len(page_text)} characters from page {page_num + 1}")
    except Exception as e:
        logger.error(f"pdfplumber failed: {e}")
    
    # OCR fallback for image-based PDFs
    if len(text.strip()) < 100:  # Threshold for OCR
        logger.info("Running OCR as fallback...")
        try:
            if os.path.exists(pytesseract.pytesseract.tesseract_cmd):
                images = convert_from_bytes(contents, dpi=300)  # Higher DPI for better OCR
                for i, img in enumerate(images):
                    ocr_text = pytesseract.image_to_string(img, lang="eng", config='--psm 6')
                    text += ocr_text + "\n"
                    logger.info(f"OCR extracted {len(ocr_text)} characters from page {i + 1}")
        except Exception as e:
            logger.error(f"OCR failed: {e}")
    
    return text.strip()

# Enhanced AI summarization
def ai_summarize(text, doc_type):
    """Enhanced AI summarization with longer, more detailed output"""
    if not summarizer:
        return f"This appears to be a {doc_type} document. AI summarization is currently unavailable."
    
    try:
        # Clean and prepare text
        clean_text = re.sub(r'\s+', ' ', text.strip())
        
        # For longer, more detailed summaries
        max_input_length = 1500  # Allow more input
        words = clean_text.split()
        
        if len(words) > max_input_length:
            # Take more text for better context
            input_text = " ".join(words[:max_input_length])
        else:
            input_text = clean_text
        
        # Generate longer summary
        result = summarizer(
            input_text,
            max_length=600,  # Much longer summary
            min_length=150,  # Ensure minimum detail
            do_sample=False,
            truncation=True,
            length_penalty=1.0,
            num_beams=4
        )
        
        if result and 'summary_text' in result[0]:
            summary = result[0]['summary_text'].strip()
            
            # Enhance the summary with context
            enhanced_summary = f"""DOCUMENT ANALYSIS SUMMARY:

{summary}

This {doc_type} document contains detailed information that has been analyzed and summarized above. The AI system has processed the content to extract the most relevant information while maintaining medical accuracy and clarity.

For complete understanding, please review the original document alongside this summary."""
            
            return enhanced_summary
        else:
            return f"This {doc_type} document was processed but could not be automatically summarized."
            
    except Exception as e:
        logger.error(f"AI summarization failed: {e}")
        # Enhanced fallback
        sentences = clean_text.split('.')[:8]  # More sentences
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        if meaningful_sentences:
            fallback = '. '.join(meaningful_sentences) + '.'
            return f"""DOCUMENT CONTENT SUMMARY:

{fallback}

Note: This is an extracted content summary as automated AI summarization was not available. Please review the original document for complete information."""
        else:
            return f"This {doc_type} document could not be processed for summarization."

# ------------------ Main summary generation ------------------
def generate_comprehensive_summary(text, target_language="english"):
    """Generate comprehensive summary with translation"""
    logger.info("Generating comprehensive document summary...")
    
    doc_type = detect_document_type(text)
    logger.info(f"Detected document type: {doc_type}")
    
    # Try specialized medical analysis first
    rule_based_summary = create_comprehensive_medical_summary(text, doc_type)
    
    if rule_based_summary:
        summary = rule_based_summary
    else:
        # Use AI summarization for non-medical or when medical analysis fails
        summary = ai_summarize(text, doc_type)
    
    # Translate if needed
    if target_language != "english":
        logger.info(f"Translating summary to {target_language}")
        translated_summary, translation_success = translate_text(summary, target_language)
        
        if translation_success:
            return translated_summary, doc_type, True
        else:
            # Add translation failure note
            return f"{summary}\n\n[Note: Translation to {target_language} was not available - summary provided in English]", doc_type, False
    
    return summary, doc_type, True

#  FastAPI routes 
@app.get("/")
async def root():
    return {"status": "NavDoc AI Medical Analyzer - Enhanced Version"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "summarizer_available": summarizer is not None,
        "summarizer_model": str(summarizer.model.name_or_path) if summarizer else "None",
        "translator_available": translator is not None,
        "translator_type": "mBART" if (translator and translator != "marian") else "Marian" if translator else "None",
        "tesseract_path": pytesseract.pytesseract.tesseract_cmd,
        "tesseract_available": os.path.exists(pytesseract.pytesseract.tesseract_cmd),
        "supported_languages": list(LANG_CODE.keys()),
        "features": ["Enhanced medical analysis", "Comprehensive summaries", "Multi-language support", "OCR processing"]
    }

@app.post("/analyze")
async def analyze_report(file: UploadFile, language: str = Form(...)):
    try:
        logger.info(f"Analyzing file: {file.filename}, target language: {language}")
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Extract text from PDF
        text = extract_text_from_pdf(contents)
        if not text or len(text.strip()) < 20:
            raise HTTPException(status_code=400, detail="No readable text found in PDF. The document may be corrupted or contain only images.")
        
        logger.info(f"Extracted {len(text)} characters from PDF")
        
        # Generate comprehensive summary
        summary_text, doc_type, translation_success = generate_comprehensive_summary(text, language.lower())
        
        # Create detailed recommendation based on document type
        if doc_type == "medical":
            recommendation = """MEDICAL REPORT RECOMMENDATIONS:

1. IMMEDIATE ACTION: Schedule a follow-up appointment with your healthcare provider to discuss these results
2. PREPARATION: Bring this report and any previous test results to your appointment  
3. QUESTIONS TO ASK: Inquire about any abnormal values and what they mean for your health
4. LIFESTYLE: Discuss any necessary dietary, exercise, or medication changes
5. MONITORING: Ask about the frequency of future testing and monitoring

Remember: This AI analysis is for informational purposes only and does not replace professional medical advice."""
        else:
            recommendation = "Please review this analysis carefully and consult with relevant professionals as needed for proper interpretation and action."
        
        # Translate recommendation if needed
        if language.lower() != "english" and translation_success:
            translated_recommendation, _ = translate_text(recommendation, language.lower())
            recommendation = translated_recommendation
        
        return {
            "success": True,
            "summary": summary_text,
            "document_type": doc_type,
            "recommendation": recommendation,
            "language": language.lower(),
            "translation_available": translator is not None,
            "translation_success": translation_success,
            "text_length": len(text),
            "features_used": [
                "Advanced document analysis",
                "Comprehensive medical insights" if doc_type == "medical" else "AI summarization",
                "Multi-language translation" if translation_success and language.lower() != "english" else "English analysis",
                "OCR processing" if len(text) > 1000 else "Direct text extraction"
            ]
        }
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)