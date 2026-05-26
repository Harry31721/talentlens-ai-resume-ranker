from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


llm = ChatOllama(model="mistral", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant.") , ("human", "{input}"), ( MessagesPlaceholder(variable_name="history"))])

chain = prompt | llm

conversation = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="input", history_messages_key="history")

def chat_with_memory(prompt_text):
    response = conversation.invoke({ "input": prompt_text }, config={"configurable": {"session_id": "user1"}})
    return response.content
def ask_llm(prompt):
    response = llm.invoke(prompt)
    return response.content

