import PyPDF2
from pdfminer.high_level import extract_text
from texttomodel import text_to_model

def ingest_pdf(file_path):
    # Extract the text from the PDF
    text = extract_text(file_path)

    # Analyze the text using the Llama model
    output = text_to_model(text)

    # Process the output
    # This will depend on your specific requirements
    # For now, let's just print the output
    print(output)
