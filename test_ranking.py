from utils.embeddings import generate_embedding
from utils.embeddings import generate_embeddings
from utils.ranking import rank_resumes

job_description = """Looking for a Python Machine Learning Developer with NLP and Deep Learning experience."""
resumes = [
    {"name": "John Doe", "text": "Python developer with machine learning and NLP skills"},
    {"name": "Jane Smith", "text": "Frontend react developer with UI/UX experience"},
    {"name": "Bob Johnson", "text": "Data scientist experienced in Tensorflow and deep learning"}
]

job_embedding = generate_embedding(job_description)
resume_texts = [r["text"] for r in resumes]
resume_embeddings = generate_embeddings(resume_texts)
scores = rank_resumes(resume_embeddings, job_embedding)

results = []

for i, score in enumerate(scores):

    if score > 0.3:
        results.append({
            "Resume": resumes[i]["name"],
            "Score": round(score*100,2)
        })


results = sorted(
    results,
    key=lambda x: x["Score"],
    reverse=True
)


for result in results:

    print(
        f"{result['Resume']} "
        f"-> {result['Score']:.4f}"
    )
