from flask import Flask , render_template,request
from sentence_transformers import SentenceTransformer,util
import fitz
import os


app=Flask(__name__)
UPLOAD_FOLDER='uploads'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER


model=SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    doc=fitz.open(pdf_path)
    text=""
    for page in doc:
        text+=page.get_text()
    return text

def score_resume(resume_text,job_description):
    resume_embedding=model.encode(resume_text)
    job_embedding=model.encode(job_description)
    similarity=util.cos_sim(resume_embedding,job_embedding)
    return round(float(similarity)*100,2)

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/screen", methods=["POST"])
def screen():
    job_description = request.form["job_description"]
    files = request.files.getlist("resumes")
    
    results = []
    
    for file in files:
        if file.filename.endswith(".pdf"):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            resume_text = extract_text_from_pdf(filepath)
            score = score_resume(resume_text, job_description)
            
            results.append({
                "name": file.filename,
                "score": score
            })
    
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    return render_template("index.html", results=results)


if __name__=="__main__":
    app.run(debug=True)