import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2  # For PDF extraction
from docx import Document
import time
from PyPDF2 import PdfReader

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    import nltk
    nltk.download('stopwords')

st.set_page_config(page_title="Resume Ranking", page_icon="📑", layout="wide")

# Custom CSS for styling and animation
st.markdown(
    """
    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes countUp {
            from { opacity: 0; transform: scale(0.8); }
            to { opacity: 1; transform: scale(1); }
        }
        @keyframes cardClick {
            from { background-color: #f8f9fa; }
            to { background-color: #007bff; color: white; }
        }
        .animated-text {
            animation: fadeIn 2s;
            text-align: center;
            color: #004aad;
            font-size: 42px;
            font-weight: bold;
            margin-top: 40px;
        }
        .score-text {
            animation: countUp 1.5s ease-out;
            font-size: 48px;
            font-weight: bold;
            text-align: center;
        }
        .subtext {
            text-align: center;
            color: #666;
            font-size: 20px;
            margin-bottom: 40px;
        }
        .feature-box {
            background: #f8f9fa;
            padding: 30px;  
            border-radius: 12px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            margin: 20px auto;
            width: 80%;
            transition: transform 0.3s, background-color 0.3s;
        }
        .feature-box:hover {
            transform: scale(1.03);
        }
        .feature-box:active {
            animation: cardClick 0.5s forwards;
        }
        .good { color: #28a745; }
        .average { color: #ffc107; }
        .bad { color: #dc3545; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='animated-text'>📑 Resume Ranking System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Optimize your resume to match job descriptions and enhance your chances of getting hired.</p>", unsafe_allow_html=True)

st.markdown("""
<div class='feature-box'>
    <h3>📌 Rank Resumes</h3>
    <p>Analyze and rank resumes based on job descriptions.</p>
</div>
<div class='feature-box'>
    <h3>📊 Compare Resumes</h3>
    <p>Compare two resumes and get similarity scores.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 class='animated-text'>📄 Resume Ranking</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Resume Ranking", "Compare Resumes"])

def extract_text_from_file(uploaded_file):
    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1]
        if file_type == "pdf":
            pdf_reader = PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        elif file_type == "docx":
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
    return ""

def preprocess(text):
    text = ''.join([char for char in text if char.isalpha() or char == ' '])
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

def extract_features(corpus):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(corpus), vectorizer

def calculate_similarity(tfidf_matrix, job_description_vector):
    return cosine_similarity(tfidf_matrix, job_description_vector)[0][0]
    
with tab1:
    st.write("### Upload your resume and enter a job description to rank it.")
    job_description = st.text_area("Enter Job Description:", height=150)
    if not job_description:
        st.warning("⚠️ Please enter a job description before uploading a resume.")
    else:
        uploaded_file = st.file_uploader("Upload Resume (PDF or Word)", type=["pdf", "docx"], accept_multiple_files=False)
        
        if uploaded_file:
            with st.spinner("Processing..."):
                time.sleep(1.5)
                resume_text = extract_text_from_file(uploaded_file)
                if resume_text:
                    processed_resume = preprocess(resume_text)
                    processed_job_description = preprocess(job_description)
                    corpus = [processed_resume, processed_job_description]
                    tfidf_matrix, vectorizer = extract_features(corpus)
                    job_description_vector = vectorizer.transform([processed_job_description])
                    similarity_score = calculate_similarity(tfidf_matrix, job_description_vector) * 100
                    
                    category = "bad" if similarity_score < 40 else "average" if similarity_score < 70 else "good"
                    st.success("✅ File Uploaded and Processed Successfully!")
                    if st.button("Show Extracted Resume Text"):
                        st.text_area("Extracted Resume Text:", resume_text, height=200)
                    
                    st.subheader("Resume Score")
                    st.markdown(f"<p class='score-text {category}'>{similarity_score:.2f}%</p>", unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Could not extract text from file.")

with tab2:
    st.write("### Compare Two Resumes")
    file1 = st.file_uploader("Upload First Resume", type=["pdf", "docx"], key="file1")
    file2 = st.file_uploader("Upload Second Resume", type=["pdf", "docx"], key="file2")
    
    if file1 and file2:
        if file1.name == file2.name:
            st.warning("⚠️ You have uploaded the same file twice. Please upload different resumes.")
        else:
            with st.spinner("Comparing Resumes..."):
                time.sleep(2)
                text1 = extract_text_from_file(file1)
                text2 = extract_text_from_file(file2)
                similarity_score = len(set(text1.split()) & set(text2.split())) / max(len(set(text1.split())), 1) * 100
                st.success(f"✅ Similarity Score: {similarity_score:.2f}%")

st.markdown("""
    <hr>
    
""", unsafe_allow_html=True)
