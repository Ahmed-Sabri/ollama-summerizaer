import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer
import PyPDF2

# ... (Keep the functions read_file, clean_text, chunk_text, embed_chunks, retrieve_relevant_chunks, rag_summarize, process_file exactly as they were) ...

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
        # Optional: Add a message prompting the user to add files
        print(f"Please place your .txt and .pdf files into the '{input_folder}' directory.")

    # Check and create output directory if it doesn't exist
    if not output_folder.is_dir():
        print(f"Output directory '{output_folder}' not found. Creating it...")
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{output_folder}' created.")
    # --- End Refined Directory Handling ---


    query = "Summarize the key points of this document or the main argument."

    # Now, safely scan the input folder (it's guaranteed to exist)
    print(f"\nScanning for files in '{input_folder}'...")
    files = list(input_folder.glob("*.txt")) + list(input_folder.glob("*.pdf")) + list(input_folder.glob("*.PDF"))

    if not files:
        # This message is now clearer if the input folder was just created
        print(f"No supported files (.txt, .pdf, .PDF) found in the '{input_folder}' folder.")
        print("Please add files to process and run the script again.")
        return # Exit gracefully if no files

    print(f"Found {len(files)} files to process.")
    results = []
    for file in files:
        print(f"\nProcessing file: {file.name} with RAG.")
        result = process_file(file, output_folder, query) # Pass output_folder which is guaranteed to exist
        if result:
            results.append(result)

    if results:
        df = pd.DataFrame(results, columns=["Filename", "Summary"])
        # Save the Excel file (output_folder is guaranteed to exist)
        excel_path = output_folder / "summaries.xlsx"
        try:
            df.to_excel(excel_path, index=False, engine='openpyxl') # Ensure openpyxl is installed
            print(f"\nAll summaries compiled into '{excel_path}'")
        except Exception as e:
            print(f"\nError saving Excel file to '{excel_path}': {e}")
            print("Please ensure the 'openpyxl' library is installed (`pip install openpyxl`)")


if __name__ == "__main__":
    main()
