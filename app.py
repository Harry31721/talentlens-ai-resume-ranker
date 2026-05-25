import streamlit as st
import pandas as pd

from utils.parser import extract_text_from_pdf
from utils.embeddings import generate_embedding
from utils.embeddings import generate_embeddings
from utils.ranking import rank_resumes
from utils.llm import (summarize_resume, generate_interview_questions, ask_resume_chatbot)
from utils.vector_store import retrieve_relevant_resumes
from utils.vectors_store import (search_resumes, create_faiss_index)

#Page configuration
st.set_page_config(
    page_title="AI Resume Ranker",
    page_icon="📄",
    layout="wide"
)

#Header
st.title("📄 AI Resume Ranker")
st.markdown("""Upload resumes and compare candidates against a job description using AI-powered semantic similarity.""")

#Sidebar
st.sidebar.title("Settings")
search_candidate = st.sidebar.text_input("Search Candidate")
minimum_score = st.sidebar.slider("Minimum Match Score", 0, 100, 50)

#Input section
job_description = st.text_area("Enter Job Description")
uploaded_files = st.file_uploader("Upload Resumes (PDF)", accept_multiple_files=True, type=["pdf"])

#Button section
if st.button("Rank Candidates"):
    #Validation
    if not uploaded_files:
        st.warning("Please upload at least one resume in PDF format.")
    elif not job_description.strip():
        st.warning("Please enter a job description.")
    else:
        resume_data = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            if text:
                resume_data.append({"name":file.name, "text": text})
        
        #Generating Embeddings
        with st.spinner("Processing resumes..."):
            job_embedding = generate_embedding(job_description)
            resume_texts = [resume["text"] for resume in resume_data]
            resume_embeddings = generate_embeddings(resume_texts)
            faiss_index = create_faiss_index(resume_embeddings)
            scores = rank_resumes(resume_embeddings, job_embedding)
        
        #Creating results
        results = []
        for i,score in enumerate(scores):
                results.append({
                    "Resume": resume_data[i]["name"],
                    "Match Score": round(score*100,2)
                })
        
        #Data Frame
        results_df = pd.DataFrame(results)

        #Score filter
        results_df = results_df[results_df["Match Score"] >= minimum_score]
        if search_candidate:
            results_df = results_df[results_df["Resume"].str.contains(search_candidate, case=False)]
        
        #Sort results
        results_df = results_df.sort_values(by="Match Score", ascending=False)
        if results_df.empty:
            st.warning("No candidates meet the minimum match score criteria.")
        else:
            #Top candidate
            top_candidate = results_df.iloc[0]
            
            st.markdown("## 🏆 Top Candidate")
            st.info(f"""Resume: {top_candidate['Resume']} Match Score: {top_candidate['Match Score']}%""")

            #Generative AI Insights button
            if st.button("Generate AI Insights for Top Candidate"):
                top_resume_name = top_candidate["Resume"]
                top_resume_text = next(resume["text"] for resume in resume_data if resume["name"] == top_resume_name)
                with st.spinner("Generating AI insights..."):
                    summary = summarize_resume(top_resume_text)
                    questions = generate_interview_questions(top_resume_text)
                
                st.markdown("## 🤖 AI Candidate Insights")
                st.subheader("Candidate Summary:")
                st.write(summary)
                st.subheader("Suggested Interview Questions:")
                st.write(questions)

            #Total matching candidates
            st.write(f"Total Matching Candidates: {len(results_df)}")

            #Dashboard layout
            col1, col2 = st.columns(2)

            #Left column
            with col1:
                st.subheader("Candidate Rankings")
                st.dataframe(results_df)

                #CSV Download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="resume_rankings.csv",
                    mime="text/csv"
                )
            
            #Right column
            with col2:
                st.subheader("Match Score Visualization")
                st.bar_chart(results_df.set_index("Resume")["Match Score"])

                #Statistics
                average_score = results_df["Match Score"].mean()
                highest_score = results_df["Match Score"].max()
                st.subheader("Statistics")
                st.metric("Average Match Score", f"{average_score:.2f}%")
                st.metric("Highest Match Score", f"{highest_score:.2f}%")

            #Recruiter AI Chatbot
            st.markdown("## 💬 Recruiter AI Assistant")
            recruiter_query = st.text_input("Ask questions about candidates:")
            if recruiter_query:
                with st.spinner("Analyzing candidates..."):
                    relevant_resumes = retrieve_relevant_resumes(recruiter_query, generate_embedding, faiss_index, resume_data, top_k=3)
                    context = "\n\n".join([ resume["text"] for resume in relevant_resumes])
                    chatbot_response = ask_resume_chatbot(recruiter_query, context)
                st.subheader("AI Assistant Response:")
                st.write(chatbot_response)

#Footer
st.markdown("---")
st.caption("Built using Streamlit, Sentence Transformers, and NLP embeddings.")