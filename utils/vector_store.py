def retrieve_relevant_resumes(query, embedding_function, index, resume_data, top_k=3):
    query_embedding = embedding_function(query)
    distances, indices = seaerch_resumes(query_embedding, index, top_k)
    retrieved_resumes = []
    for idx in indices:
        retrieved_resumes.append(resume_data[idx])
    return retrieved_resumes
