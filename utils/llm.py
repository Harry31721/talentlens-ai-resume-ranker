import ollama

MODEL_NAME = "mistral"

def summarize_resume(resume_text):
    prompt = f"""Summarize the following resume professionally.
    Include:
    -key skills
    -experience
    -strengths
    
    Resume:{resume_text}"""
    response = ollama.chat(model = MODEL_NAME,messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def generate_interview_questions(resume_text):
    prompt = f"""Generate 5 interview questions based on this candidate's resume.
    Resume:{resume_text}"""
    response = ollama.chat(model = MODEL_NAME,messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def ask_resume_chatbot(query, context):
    prompt = f"""You are an AI hiring assistant. 
    Use the provided candidate resume information to answer the recruiter's question.
    Candidate Resume Information: {context}
    Recruiter's Question: {query}"""
    response = ollama.chat(model = MODEL_NAME,messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
