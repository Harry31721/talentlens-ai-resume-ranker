from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_file):

    try:
        text = ""
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
        cleaned_text = text.replace("\n", " ")
        cleaned_text = cleaned_text.strip()
        return cleaned_text

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""