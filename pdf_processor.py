# pdf_processor.py
import fitz  # PyMuPDF
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path: str) -> str | None:
    """Extracts text content from a PDF file."""
    try:
        logging.info(f"Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        text = ""
        logging.info(f"Extracting text from {len(doc)} pages...")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        logging.info(f"Successfully extracted {len(text)} characters.")
        # Basic cleaning (optional, can be more sophisticated)
        text = ' '.join(text.split()) # Remove excessive whitespace
        return text
    except Exception as e:
        logging.error(f"Error processing PDF {pdf_path}: {e}")
        return None

if __name__ == '__main__':
    # Example usage: Replace with an actual PDF path
    pdf_file = "Recommender_Systems.pdf" # <--- PUT YOUR PDF FILENAME HERE
    if pdf_file == "example_paper.pdf":
         print("Please replace 'example_paper.pdf' with the actual path to your PDF file.")
    else:
        extracted_text = extract_text_from_pdf(pdf_file)
        if extracted_text:
            print(f"Extracted text (first 500 chars):\n{extracted_text[:500]}...")
            print(f"\nTotal characters extracted: {len(extracted_text)}")
        else:
            print("Failed to extract text from PDF.")