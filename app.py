import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer
import PyPDF2
# import traceback # Optional: for detailed error printing in process_file

# --- Function Definitions ---

def read_file(file_path: Path) -> str:
    """
    Read file content from .txt or .pdf.
    Returns empty string if reading fails or no text extracted.
    """
    try:
        if file_path.suffix.lower() == ".txt":
            return file_path.read_text(encoding="utf-8", errors='ignore') # Added errors='ignore' for robustness
        elif file_path.suffix.lower() == ".pdf":
            text = ""
            try:
                with file_path.open("rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    if reader.is_encrypted:
                         print(f"Warning: Skipping encrypted PDF: {file_path.name}")
                         return ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as pdf_err:
                 print(f"Error reading PDF {file_path.name}: {pdf_err}")
                 return "" # Return empty string on PDF read error
            return text
        else:
            print(f"Warning: Unsupported file type skipped: {file_path.name}")
            return "" # Return empty for unsupported types
    except Exception as e:
        print(f"Error opening or reading file {file_path.name}: {e}")
        return "" # Return empty string on general file read error


def clean_text(text: str) -> str:
    """
    Remove sections like 'Bibliography' or 'References' if present.
    """
    if not text: # Handle empty input text
        return ""
    # Use a case-insensitive search and look for the start of the line
    match = re.search(r"^(Bibliography|References)\s*$", text, re.IGNORECASE | re.MULTILINE)
    return text[:match.start()] if match else text


def chunk_text(text: str, max_chunk_length: int = 2500) -> list:
    """
    Split text into smaller chunks; avoids creating empty chunks.
    """
    if not text: # Handle empty input text
        return []
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        stripped_para = para.strip()
        if not stripped_para: # Skip empty paragraphs
            continue

        if len(current_chunk) + len(stripped_para) + 1 > max_chunk_length:
            # Add the current chunk if it's not empty
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # Start new chunk with the current paragraph
            current_chunk = stripped_para + "\n"
        else:
            # Add paragraph to the current chunk
            current_chunk += stripped_para + "\n"

    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def embed_chunks(chunks: list, embedder) -> np.ndarray:
    """
    Compute embedding for each chunk. Handles case of no chunks.
    """
    if not chunks:
        return np.array([]) # Return empty numpy array if no chunks
    try:
        embeddings = embedder.encode(chunks, show_progress_bar=False) # Hide progress bar for cleaner logs
        return np.array(embeddings)
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        return np.array([])


def retrieve_relevant_chunks(query: str, chunks: list, chunk_embeddings: np.ndarray,
                              embedder, top_k: int = 3) -> list:
    """
    Retrieve top_k chunks that are most similar to the query.
    Handles cases with fewer chunks than top_k or no embeddings.
    """
    if chunk_embeddings.size == 0 or not chunks:
        print("Warning: No chunks or embeddings available for retrieval.")
        return [] # Return empty list if no embeddings/chunks

    try:
        query_embedding = embedder.encode(query)
    except Exception as e:
        print(f"Error embedding query: {e}")
        return []

    # Ensure embeddings are 2D
    if chunk_embeddings.ndim == 1:
        # This might happen if only one chunk was embedded incorrectly?
        # Or if embed_chunks returned an empty array incorrectly structured.
        # Let's reshape, assuming it should have been (1, N)
        if chunk_embeddings.size > 0:
             print("Warning: Reshaping potentially 1D chunk embeddings.")
             chunk_embeddings = chunk_embeddings.reshape(1, -1)
        else:
             print("Warning: Chunk embeddings array is empty after checking size.")
             return [] # Cannot proceed if truly empty

    # Ensure query_embedding is 1D
    query_embedding = np.array(query_embedding).flatten()

    try:
        # Calculate norms safely
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)

        # Avoid division by zero
        denominator = chunk_norms * query_norm
        # Use a small epsilon where denominator is close to zero
        denominator = np.where(denominator < 1e-10, 1e-10, denominator)

        similarities = np.dot(chunk_embeddings, query_embedding) / denominator

        # Determine how many indices to retrieve (cannot exceed number of chunks)
        actual_top_k = min(top_k, len(chunks))
        if actual_top_k <= 0:
            return []

        # Get indices of the top similarities
        # argsort sorts ascending, so take the last 'actual_top_k' indices
        top_indices = np.argsort(similarities)[-actual_top_k:][::-1] # Get top k and reverse to descending

        return [chunks[i] for i in top_indices]

    except ValueError as ve:
         print(f"Error calculating similarities (possible dimension mismatch): {ve}")
         print(f"Chunk embeddings shape: {chunk_embeddings.shape}")
         print(f"Query embedding shape: {query_embedding.shape}")
         return []
    except Exception as e:
        print(f"Error during chunk retrieval: {e}")
        return []


def rag_summarize(document_text: str, query: str) -> str:
    """
    Given a document and a query, retrieve top relevant chunks and use them to prompt the LLM.
    """
    cleaned_text = clean_text(document_text)
    if not cleaned_text:
        print("Warning: Text is empty after cleaning.")
        return "" # Return empty if nothing to process

    chunks = chunk_text(cleaned_text)
    if not chunks:
        print("Warning: No chunks generated from the text.")
        return "" # Return empty if no chunks

    print(f"Document split into {len(chunks)} chunks.")

    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2") # Consider loading model once outside if performance matters
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        return ""

    embeddings = embed_chunks(chunks, embedder)
    if embeddings.size == 0:
        print("Warning: Embedding generation failed.")
        return "" # Return empty if embedding failed

    relevant_chunks = retrieve_relevant_chunks(query, chunks, embeddings, embedder, top_k=3)
    if not relevant_chunks:
        print("Warning: Could not retrieve relevant chunks.")
        # Optional: Fallback - use first chunk? Or just return empty?
        # context = chunks[0] if chunks else "" # Example fallback
        return "" # Return empty if no relevant chunks found

    context = "\n\n---\n\n".join(relevant_chunks) # Add separator for clarity

    prompt = (f"Based *only* on the following context, please answer the question.\n\n"
              f"Context:\n{context}\n\n"
              f"---\n\n"
              f"Question: {query}\n\n"
              f"Answer:")

    try:
        print("Generating response with Ollama...")
        response = ollama.generate(model="gemma:2b", prompt=prompt) # Using gemma:2b as a common alternative
        # response = ollama.generate(model="llama3:8b", prompt=prompt) # Example using llama3 8b
        generated_text = response.get("response", "").strip()
        if not generated_text:
             print("Warning: Ollama returned an empty response.")
             return ""
        print("Ollama generation complete.")
        return generated_text
    except Exception as e:
        print(f"Error during Ollama generation: {e}")
        # Check if Ollama service is running
        # print("Ensure the Ollama service is running and the model (e.g., 'gemma:2b') is pulled ('ollama pull gemma:2b').")
        return ""


def process_file(file_path: Path, output_folder: Path, query: str) -> tuple[str, str] or None:
    """
    Process a file using RAG: read the file, summarize it,
    save the summary as a .txt file, and return (filename, summary).
    """
    try:
        print(f"Reading file: {file_path.name}")
        text = read_file(file_path)
    except Exception as e:
        # read_file now handles its internal errors and returns ""
        print(f"Error reported while trying to read {file_path.name}: {e}") # Log error even if read_file handles it
        return None

    if not text or text.isspace(): # Check if text is empty or just whitespace after reading
        print(f"Warning: No text could be extracted or read from {file_path.name}.")
        return None

    try:
        print(f"Attempting RAG summarization for {file_path.name}...")
        answer = rag_summarize(text, query)
        if answer: # Ensure the answer is not empty or just whitespace
            output_file = output_folder / f"{file_path.stem}_rag_answer.txt"
            try:
                output_file.write_text(answer, encoding="utf-8")
                print(f"RAG answer for {file_path.name} saved to {output_file}")
                return file_path.name, answer
            except Exception as write_err:
                 print(f"Error writing summary file {output_file}: {write_err}")
                 return None # Failed to save the result
        else:
            print(f"Warning: RAG summarization returned no answer for {file_path.name}.")
            return None
    except Exception as e:
        print(f"Error during RAG processing for {file_path.name}: {e}")
        # Optional: print detailed traceback for debugging
        # traceback.print_exc()
        return None


# --- Main Execution Logic ---

def main():
    # Define input and output paths
    input_folder = Path("input")
    output_folder = Path("output_rag")

    # --- Refined Directory Handling ---
    # Check and create input directory if it doesn't exist
    if not input_folder.is_dir():
        print(f"Input directory '{input_folder}' not found. Creating it...")
        input_folder.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{input_folder}' created.")
        print(f"Please place your .txt and .pdf files into the '{input_folder}' directory and run again.")
        # No point proceeding if input was just created and is empty
        return

    # Check and create output directory if it doesn't exist
    if not output_folder.is_dir():
        print(f"Output directory '{output_folder}' not found. Creating it...")
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{output_folder}' created.")
    # --- End Refined Directory Handling ---


    query = "Summarize the key points of this document or the main argument."
    # query = "What is the main topic of this document?" # Alternative query example

    # Now, safely scan the input folder (it's guaranteed to exist)
    print(f"\nScanning for files in '{input_folder}'...")
    # Using case-insensitive matching for PDF extension
    files = list(input_folder.glob("*.txt")) + list(input_folder.glob("*.pdf")) + list(input_folder.glob("*.PDF"))

    if not files:
        print(f"No supported files (.txt, .pdf, .PDF) found in the '{input_folder}' folder.")
        print("Please add files to process and run the script again.")
        return # Exit gracefully if no files

    print(f"Found {len(files)} files to process.")
    results = []
    for file in files:
        print("-" * 40) # Separator for each file
        print(f"Processing file: {file.name}")
        result = process_file(file, output_folder, query) # Pass output_folder which is guaranteed to exist
        if result:
            results.append(result)
        else:
             print(f"Skipping {file.name} due to processing issues.")

    print("-" * 40) # Final separator

    if results:
        df = pd.DataFrame(results, columns=["Filename", "Summary"])
        # Save the Excel file (output_folder is guaranteed to exist)
        excel_path = output_folder / "summaries.xlsx"
        try:
            df.to_excel(excel_path, index=False, engine='openpyxl')
            print(f"\nProcessing complete. {len(results)} summaries compiled into '{excel_path}'")
        except Exception as e:
            print(f"\nError saving Excel file to '{excel_path}': {e}")
            print("Please ensure the 'openpyxl' library is installed (`pip install openpyxl`)")
            print("Summaries were generated but not saved to Excel.")
    else:
        print("\nProcessing complete. No summaries were successfully generated.")


if __name__ == "__main__":
    # Check if Ollama is installed and running (basic check)
    try:
        # Try a simple command like listing local models
        ollama.list()
        print("Ollama service detected.")
    except Exception as ollama_err:
        print("\n--- Ollama Connection Error ---")
        print(f"Error: Could not connect to Ollama: {ollama_err}")
        print("Please ensure the Ollama application or service is running.")
        print("You might need to start it manually.")
        print("Exiting script.")
        exit() # Exit if Ollama is not reachable

    # Check for required Ollama model (e.g., gemma:2b)
    required_model = "gemma:2b" # Or "llama3:8b", etc. Change in rag_summarize too!
    try:
        models = ollama.list()['models']
        model_names = [m['name'] for m in models]
        if required_model not in model_names:
             print(f"\n--- Ollama Model Missing ---")
             print(f"Error: Required Ollama model '{required_model}' not found locally.")
             print(f"Please pull the model using the command: ollama pull {required_model}")
             print("Exiting script.")
             exit()
        else:
             print(f"Required Ollama model '{required_model}' found.")

    except Exception as ollama_list_err:
         print(f"\nWarning: Could not verify Ollama models list: {ollama_list_err}")
         print("Proceeding, but generation might fail if the model is missing.")

    # Proceed to main processing
    main()
