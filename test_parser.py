from utils.parser import extract_text_from_pdf

with open("Resume_Harikrishan.pdf", "rb") as file:
    text = extract_text_from_pdf(file)
print(text[:2000])