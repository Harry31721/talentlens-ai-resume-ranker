from utils.vectors_store import search_resumes  # ✅ Fix: was missing entirely


def retrieve_relevant_resumes(query, embedding_function, index, resume_data, top_k=3):
    query_embedding = embedding_function(query)
    distances, indices = search_resumes(query_embedding, index, top_k)  # ✅ Fix: was "seaerch_resumes"
    retrieved_resumes = []
    for idx in indices[0]:  # indices is 2D from FAISS, need [0]
        idx = int(idx)  # ✅ Fix: convert numpy int → Python int so list indexing works
        if 0 <= idx < len(resume_data):  # guard against -1 (FAISS padding) and out-of-bounds
            retrieved_resumes.append(resume_data[idx])
    return retrieved_resumes