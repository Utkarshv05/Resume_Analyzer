import streamlit as st
import pickle
import docx  # Requires `python-docx` package
import PyPDF2
import re

# === Model Paths (Make sure files exist at these paths) ===
MODEL_PATH = r"D:\finalYear\Resume-Screening-App-main\clf.pkl"
VECTORIZER_PATH = r"D:\finalYear\Resume-Screening-App-main\tfidf.pkl"
ENCODER_PATH = r"D:\finalYear\Resume-Screening-App-main\encoder.pkl"

# === Load Pre-trained Artifacts ===
try:
    with open(MODEL_PATH, 'rb') as model_file:
        svc_model = pickle.load(model_file)

    with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
        tfidf = pickle.load(vectorizer_file)

    with open(ENCODER_PATH, 'rb') as encoder_file:
        le = pickle.load(encoder_file)

except FileNotFoundError as e:
    st.error(f"Model file not found: {e.filename}")
    st.stop()

# === Clean Resume Text ===
def cleanResume(txt):
    txt = re.sub(r'http\S+\s*', ' ', txt)
    txt = re.sub(r'RT|cc', ' ', txt)
    txt = re.sub(r'#\S+|@\S+', ' ', txt)
    txt = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()

# === Extract Text Functions ===
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''.join([page.extract_text() or '' for page in pdf_reader.pages])
        return text
    except Exception as e:
        st.error(f"Failed to read PDF: {str(e)}")
        return ""

def extract_text_from_docx(file):
    try:
        document = docx.Document(file)
        return '\n'.join([para.text for para in document.paragraphs])
    except Exception as e:
        st.error(f"Failed to read DOCX: {str(e)}")
        return ""

def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        try:
            file.seek(0)
            return file.read().decode('latin-1')
        except Exception as e:
            st.error(f"Failed to read TXT file: {str(e)}")
            return ""

# === File Upload Handling ===
def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")

# === Predict Resume Category ===
def pred(input_resume):
    cleaned = cleanResume(input_resume)
    vectorized = tfidf.transform([cleaned])
    prediction = svc_model.predict(vectorized.toarray())
    return le.inverse_transform(prediction)[0]

# === Streamlit App ===
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="centered")
    st.title("📄 Resume Category Predictor")
    st.markdown("Upload a resume (PDF, DOCX, or TXT) to predict the job category.")

    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    if uploaded_file:
        resume_text = handle_file_upload(uploaded_file)

        if resume_text.strip():
            st.success("✅ Resume text successfully extracted.")

            if st.checkbox("Show Extracted Resume Text"):
                st.text_area("Resume Text", resume_text, height=300)

            st.subheader("🎯 Predicted Job Category")
            category = pred(resume_text)
            st.markdown(f"**Predicted Category:** 🏷️ `{category}`")
        else:
            st.warning("The uploaded file doesn't contain readable text.")

# === Run App ===
if __name__ == "__main__":
    main()
