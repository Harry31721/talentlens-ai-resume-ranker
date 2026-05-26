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

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI recruiter assistant. Use the conversation history to provide context-aware answers."),
    MessagesPlaceholder(variable_name="history"),  # ✅ Fix: history must come BEFORE the current human input
    ("human", "{input}"),
])

chain = prompt | llm

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


def chat_with_memory(prompt_text: str, session_id: str = "user1") -> str:
    """Chat with persistent memory across turns."""
    response = conversation.invoke(
        {"input": prompt_text},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content


def ask_llm(prompt: str) -> str:
    """One-shot LLM call with no memory."""
    response = llm.invoke(prompt)
    return response.content


def clear_session(session_id: str = "user1"):
    """Clear chat history for a given session."""
    if session_id in store:
        del store[session_id]