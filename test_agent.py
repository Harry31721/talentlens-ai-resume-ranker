from utils.agent import ask_agent

resume = """
Python developer with experience
in NLP, TensorFlow, FastAPI,
and Machine Learning.
"""

query = f"""
Generate interview questions
for this candidate:

{resume}
"""

response = ask_agent(query)
print(response)