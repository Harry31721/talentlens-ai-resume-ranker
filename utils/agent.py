from langchain_ollama import ChatOllama

llm = ChatOllama(model="mistral", temperature=0.7)


def ask_agent(query: str, resume_context: str = "") -> str:
    try:
        # Build the resume section only if context is provided
        resume_section = f"""
        You have access to the following candidate resumes:
        -------------------------------------------------------
        {resume_context}
        -------------------------------------------------------
        Use this information to answer questions about specific candidates.
        Always mention the candidate's name (resume filename) when referring to them.
        """ if resume_context.strip() else ""

        system_prompt = f"""
        You are an AI recruiter assistant.

        {resume_section}

        Help recruiters with:
        - Comparing candidates across resumes
        - Identifying the most experienced or skilled candidate
        - Candidate evaluation and shortlisting
        - Generating interview questions
        - Resume analysis and highlights
        - Technical assessments

        Recruiter Query:
        {query}
        """

        response = llm.invoke(system_prompt)
        return response.content

    except Exception as e:
        return f"Error: {str(e)}"