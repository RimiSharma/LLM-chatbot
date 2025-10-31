# LLM-chatbot

📄 LLM Document Digester Chatbot

This project implements a Retrieval-Augmented Generation (RAG) system using an LLM to answer user questions based on the content of provided PDF documents. It features a modular structure for processing documents, creating vector stores, and running an interactive command-line chat session.

✨ Features

PDF Processing: Extracts text efficiently from single or multiple PDF files.

Data Digestion: Processes extracted text into a format suitable for the LLM (e.g., chunking and embedding).

Conversational Chat Interface: Allows interactive querying against the document content.

Modular Design: Clear separation of concerns between data preparation, core processing, and the user interface.

🛠 Prerequisites

Before running this project, ensure you have the following installed:

Python 3.8+

An API Key for the LLM service OpenAI.

🚀 Installation and Setup

Clone the repository (if applicable) and navigate to the project directory:

git clone <your-repo-url>
cd llm-document-digester


Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Linux/macOS
.\venv\Scripts\activate   # On Windows


Install the required libraries:

pip install -r requirements.txt \
OR libraries \
pip install pypdf langchain google-openai numpy


Set up your API Key:

The application requires an API key to communicate with the LLM. It is best practice to set this as an environment variable.

# On Linux/macOS
export OPENAI_API_KEY="YOUR_API_KEY_HERE"

# On Windows (Command Prompt)
set OPENAI_API_KEY="YOUR_API_KEY_HERE"


📖 Usage

Place your PDF documents: Put all the PDF files you want the chatbot to read into a designated directory (a folder named docs/).

Run the main digester script: This script handles the initial processing, embedding, and storage of the document content.

python main_digester.py


Start the Chatbot: Once the data is processed, you can launch the interactive chat interface.

python chatbot.py


You can now ask questions related to the content of your digested PDF files!

📂 Project Structure

The project is divided into three core modules, each with a distinct responsibility:

pdf_processor.py

This module is dedicated to the extraction and initial preparation of raw text from documents.

Function

Responsibility

Document Reading

Handles loading and iterating through PDF files.

Text Extraction

Uses a library (e.g., pypdf) to extract all text content.

Preprocessing

May include basic cleanup like removing excessive whitespace or header/footer noise.

Output

Returns a list of text strings or a single concatenated string of the document content.

main_digester.py

This script serves as the core pipeline manager, taking raw text and transforming it into a structured, queryable data store (the "digested" knowledge base).

Function

Responsibility

Orchestration

Imports and calls functions from pdf_processor.py.

Text Chunking

Breaks the long document text into smaller, manageable chunks suitable for embedding.

Embedding Generation

Uses an embedding model (e.g., Openai's embedding models) to convert text chunks into dense vector representations.

Vector Store Management

Saves the embedded vectors into a searchable database (e.g., FAISS, ChromaDB, or similar) for efficient retrieval.

chatbot.py

This module contains the interactive chat loop and the Retrieval-Augmented Generation (RAG) logic.

Function

Responsibility

User Interface

Provides the command-line interface for the user to input questions.

Retrieval

When a user asks a question, this script queries the vector store (created by main_digester.py) to find the most relevant document chunks.

Prompt Engineering

Constructs the final prompt for the LLM, combining the user's query and the retrieved context chunks.

LLM Interaction

Sends the final prompt to the LLM OPENAI and displays the grounded response to the user.
