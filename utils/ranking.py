from sklearn.metrics.pairwise import cosine_similarity

def rank_resumes(resume_embeddings, job_embedding):
    scores = cosine_similarity([job_embedding],resume_embeddings)[0]
    return scores