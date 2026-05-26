from utils.embeddings import generate_embedding, generate_embeddings
from sklearn.metrics.pairwise import cosine_similarity

text1 = "Python Machine Learning Engineer"
text2 = "ML Developer skilled in Python"
text3 = "Graphic Designer with Photoshop"

embedding1 = generate_embedding(text1)
embedding2 = generate_embedding(text2)
embedding3 = generate_embedding(text3)

similarity_1_2 = cosine_similarity(
    [embedding1],
    [embedding2]
)

similarity_1_3 = cosine_similarity(
    [embedding1],
    [embedding3]
)

print("Similar Texts:", similarity_1_2)
print("Different Texts:", similarity_1_3)

texts = [
    "Python Developer",
    "Machine Learning Engineer",
    "Frontend React Developer"
]

embeddings = generate_embeddings(texts)

print(len(embeddings))