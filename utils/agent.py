from langchain_ollama import ChatOllama

from utils.llm import (
    summarize_resume,
    generate_interview_questions
)


llm = ChatOllama(

    model="mistral",

    temperature=0.7
)


def ask_agent(query):

    query_lower = query.lower()


    # -----------------------------------------
    # TOOL: INTERVIEW QUESTIONS
    # -----------------------------------------

    if (
        "interview" in query_lower
        or "questions" in query_lower
    ):

        return generate_interview_questions(query)


    # -----------------------------------------
    # TOOL: RESUME SUMMARY
    # -----------------------------------------

    elif (
        "summarize" in query_lower
        or "summary" in query_lower
    ):

        return summarize_resume(query)


    # -----------------------------------------
    # DEFAULT LLM RESPONSE
    # -----------------------------------------

    else:

        response = llm.invoke(query)

        return response.content