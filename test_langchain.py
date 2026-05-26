from utils.langchain_chat import (ask_llm, chat_with_memory)

response1 = chat_with_memory("John is backend engineer skilled in FastAPI.")
print("\nResponse 1:\n")
print(response1)

reponse2 = chat_with_memory("What are John's skills?")
print("\nResponse 2:\n")
print(reponse2)

reponse3 = chat_with_memory("Generate interview questions for John.")
print("\nResponse 3:\n")
print(reponse3)