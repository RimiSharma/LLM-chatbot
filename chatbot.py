# chatbot.py
import os
import logging
import openai
from openai import OpenAI, RateLimitError, APIError
import time
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv() # Load environment variables from .env file

class Chatbot:
    """A chatbot class to interact with OpenAI's API."""

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.5):
        """
        Initializes the Chatbot.

        Args:
            model: The OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4").
            temperature: Controls randomness (0.0 to 1.0). Lower is more deterministic.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        logging.info(f"Chatbot initialized with model: {self.model}")

    def _call_llm(self, system_prompt: str, user_prompt: str, max_retries: int = 3, delay: int = 5) -> str | None:
        """
        Makes a call to the OpenAI API with retry logic.

        Args:
            system_prompt: The role or context for the AI.
            user_prompt: The specific instruction or question for the AI.
            max_retries: Maximum number of retry attempts for rate limits/server errors.
            delay: Delay in seconds between retries.

        Returns:
            The AI's response content or None if an error persists.
        """
        attempt = 0
        while attempt < max_retries:
            try:
                logging.info(f"Calling LLM (Attempt {attempt + 1}/{max_retries})...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                )
                # Ensure response.choices is not empty and has a message
                if response.choices and response.choices[0].message:
                     content = response.choices[0].message.content
                     if content:
                        logging.info("LLM call successful.")
                        return content.strip()
                     else:
                        logging.warning("LLM returned an empty message.")
                        return None # Or handle as appropriate
                else:
                     logging.warning("LLM response structure unexpected or empty.")
                     return None # Or handle appropriately

            except RateLimitError as e:
                logging.warning(f"Rate limit exceeded. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                attempt += 1
            except APIError as e:
                logging.warning(f"API error occurred: {e}. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                attempt += 1
            except Exception as e:
                logging.error(f"An unexpected error occurred during LLM call: {e}")
                return None # Non-retryable error or final attempt failed

        logging.error("LLM call failed after multiple retries.")
        return None

    def execute_task(self, role_prompt: str, task_prompt: str) -> str | None:
        """
        Executes a specific task by calling the LLM.

        Args:
            role_prompt: The system message defining the bot's role.
            task_prompt: The user message defining the specific task.

        Returns:
            The result from the LLM or None if an error occurred.
        """
        logging.info(f"Executing task with role: {role_prompt[:100]}...") # Log snippet
        return self._call_llm(system_prompt=role_prompt, user_prompt=task_prompt)

if __name__ == '__main__':
    # Example usage (requires .env file with API key)
    try:
        bot = Chatbot(model="gpt-3.5-turbo") # Use "gpt-4" for potentially better results if available

        # --- Test Extraction Task ---
        extractor_role = "You are an AI assistant specialized in extracting key information from research papers."
        sample_text = """
        Abstract: We propose a novel method for improving widget sorting efficiency.
        Introduction: Sorting widgets is a common problem. Existing methods suffer from O(n^2) complexity.
        Methods: Our technique, 'FastSort', uses a divide-and-conquer approach with a heuristic pivot selection.
        Results: FastSort achieves O(n log n) average-case performance, outperforming BubbleSort by 100x on datasets up to 1 million widgets.
        Conclusion: FastSort offers a significant improvement for large-scale widget sorting. Limitations include sensitivity to pivot choice in worst-case scenarios.
        """
        extraction_task = f"""
        From the following research paper text, extract:
        1. Core Problem:
        2. Proposed Method/Solution:
        3. Key Results:
        4. Main Conclusion:
        5. Mentioned Limitations (if any):

        Paper Text:
        ---
        {sample_text}
        ---
        Provide the extracted information clearly labeled.
        """
        extraction_result = bot.execute_task(extractor_role, extraction_task)
        print("\n--- Extraction Bot Test ---")
        if extraction_result:
            print(extraction_result)
        else:
            print("Extraction task failed.")

        # --- Test Synthesis Task ---
        synthesizer_role = "You are an AI assistant skilled at synthesizing and critiquing research summaries. You aim to produce a concise, informative digest highlighting strengths and weaknesses."
        synthesis_task = f"""
        Below is an initial extraction from a research paper. Review it along with the original text (provided again for context).
        Your goal is to synthesize this into a final digest (~150 words).
        Critically evaluate: Are the results clearly impactful? Are limitations properly acknowledged? Is the core idea novel?
        Integrate your critique into the final digest.

        Original Text:
        ---
        {sample_text}
        ---

        Initial Extraction:
        ---
        {extraction_result or 'Extraction Failed'}
        ---

        Produce the final synthesized digest.
        """
        synthesis_result = bot.execute_task(synthesizer_role, synthesis_task)
        print("\n--- Synthesizer Bot Test ---")
        if synthesis_result:
            print(synthesis_result)
        else:
            print("Synthesis task failed.")

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")