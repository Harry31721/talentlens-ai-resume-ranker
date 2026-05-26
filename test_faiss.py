from utils.embeddings import generate_embedding
from utils.embeddings import generate_embeddings
from utils.vectors_store import (create_faiss_index, search_resumes)

resumes = ["Python developer with NLP experience", "Frontend React developer", "Machine learning engineer with TensorFlow", "Backend engineer with FastAPI"]
resume_embeddings = generate_embeddings(resumes)
index = create_faiss_index(resume_embeddings)

query = "Looking for NLP engineer"
query_embedding = generate_embedding(query)
distances, indices = search_resumes(query_embedding, index)

print(indices)