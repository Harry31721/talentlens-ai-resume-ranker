from utils.llm import (summarize_resume, generate_interview_questions)

resume = """ Python developer with experience in Machine Learning, NLP, TensorFlow, FastAPI, and SQL"""
summary = summarize_resume(resume)
print("\nSummary:\n")
print(summary)

questions = generate_interview_questions(resume)
print("\nInterview Questions:\n")
print(questions)