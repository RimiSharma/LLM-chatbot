# main_digester.py
import os
import logging
from pdf_processor import extract_text_from_pdf
from chatbot import Chatbot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PDF_PATH = "Recommender_Systems.pdf"  # <--- IMPORTANT: SET THIS TO YOUR PDF FILE
LLM_MODEL = "gpt-3.5-turbo" # Or "gpt-4", "gpt-4-turbo", etc.
MAX_TEXT_LENGTH_WARN = 50000 # Warn if extracted text is very long (potential context issues)
# --- End Configuration ---


def run_digestion_pipeline(pdf_path: str):
    """
    Orchestrates the dual-chatbot digestion process.
    """
    logging.info(f"--- Starting Research Paper Digestion for: {pdf_path} ---")

    # 1. Extract Text from PDF
    logging.info("Step 1: Extracting text from PDF...")
    paper_text = extract_text_from_pdf(pdf_path)
    if not paper_text:
        logging.error("Failed to extract text. Aborting.")
        return
    logging.info(f"Text extracted ({len(paper_text)} characters).")
    if len(paper_text) > MAX_TEXT_LENGTH_WARN:
        logging.warning(f"Extracted text is long ({len(paper_text)} chars). "
                        f"This might exceed LLM context limits or be slow/expensive.")
    elif len(paper_text) < 500: # Arbitrary short length check
         logging.warning(f"Extracted text is very short ({len(paper_text)} chars). "
                         f"Was the PDF text-based? Check extraction quality.")


    # 2. Initialize Chatbots
    try:
        extractor_bot = Chatbot(model=LLM_MODEL, temperature=0.3) # More factual
        synthesizer_bot = Chatbot(model=LLM_MODEL, temperature=0.6) # Slightly more creative/critical
    except ValueError as e:
        logging.error(f"Failed to initialize chatbots: {e}. Ensure API key is set in .env")
        return
    except Exception as e:
        logging.error(f"Unexpected error initializing chatbots: {e}")
        return


    # 3. Extractor Bot Task
    logging.info("\n--- Step 2: Extractor Bot Processing ---")
    extractor_role = "You are an AI assistant specialized in extracting key structured information from academic research papers. Focus on accuracy and conciseness."
    extraction_prompt = f"""
    Carefully read the following research paper text. Extract the following sections clearly and concisely:
    1.  **Core Problem:** What specific problem or question does the paper address?
    2.  **Proposed Method/Solution:** Briefly describe the key technique, model, or approach proposed.
    3.  **Key Results:** What were the main quantitative or qualitative findings? Mention key metrics if possible.
    4.  **Main Conclusion:** What is the primary takeaway or claim of the paper?
    5.  **Mentioned Limitations:** List any limitations, weaknesses, or future work mentioned by the authors.

    Research Paper Text:
    ---
    {paper_text[:8000]}
    ---
    [Note: Only the first 8000 characters of the paper text are provided above due to potential length constraints. Base your extraction on this initial part if the full text is too long for a single prompt in a real scenario, or adjust slicing as needed.]

    Provide the output clearly structured under the headings above.
    """
    # WARNING: Sending full 'paper_text' might fail for long papers. Slicing [:8000] is a temporary measure.
    # A robust solution needs chunking or RAG.
    initial_extraction = extractor_bot.execute_task(extractor_role, extraction_prompt.replace("{paper_text[:8000]}", paper_text)) # Replace placeholder

    if not initial_extraction:
        logging.error("Extractor Bot failed to produce an output. Aborting.")
        return
    print("\n--- Extractor Bot Output ---")
    print(initial_extraction)


    # 4. Synthesizer/Critic Bot Task
    logging.info("\n--- Step 3: Synthesizer/Critic Bot Processing ---")
    synthesizer_role = "You are an AI assistant skilled at critically evaluating and synthesizing research paper summaries. Your goal is to produce a final, balanced digest (around 150-250 words) that incorporates the key findings and offers a brief critical perspective."
    synthesis_prompt = f"""
    You have been provided with an initial extraction from a research paper. Review this extraction in the context of the full paper text (provided again below for reference).

    Your Tasks:
    1.  **Synthesize:** Combine the extracted points into a coherent narrative digest summarizing the paper's core contribution.
    2.  **Critique:** Briefly assess the significance of the findings. Are the limitations acknowledged appropriately? Does the method seem sound based on the description? (Be objective).
    3.  **Format:** Produce a single block of text representing the final digest.

    Initial Extraction:
    ---
    {initial_extraction}
    ---

    Full Paper Text (for context - may be truncated):
    ---
    {paper_text[:8000]}
    ---
    [Note: Full paper text context provided again, potentially truncated.]

    Generate the final synthesized and critiqued digest.
    """
    # WARNING: Again, sending full 'paper_text' is risky for long papers.
    final_digest = synthesizer_bot.execute_task(synthesizer_role, synthesis_prompt.replace("{paper_text[:8000]}", paper_text)) # Replace placeholder

    if not final_digest:
        logging.error("Synthesizer Bot failed to produce an output.")
        print("\n--- Final Digest: FAILED ---")
        return

    print("\n--- Final Synthesized Digest ---")
    print(final_digest)
    logging.info("--- Digestion Pipeline Completed ---")


if __name__ == "__main__":
    if not os.path.exists(PDF_PATH) and PDF_PATH == "example_paper.pdf":
         print(f"Error: Default PDF '{PDF_PATH}' not found.")
         print("Please update the 'PDF_PATH' variable in 'main_digester.py' to your research paper PDF file.")
    elif not os.path.exists(PDF_PATH):
         print(f"Error: PDF file not found at '{PDF_PATH}'.")
         print("Please check the 'PDF_PATH' variable in 'main_digester.py'.")
    else:
        run_digestion_pipeline(PDF_PATH)