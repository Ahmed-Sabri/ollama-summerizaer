# ollama-summerizaer
# Local RAG Document Q&A Script

This Python script uses a Retrieval-Augmented Generation (RAG) approach to process local documents (`.txt` and `.pdf`). It answers a specific query about each document by leveraging a local Large Language Model (LLM) via Ollama and sentence embeddings. The results are saved as individual text files and compiled into an Excel spreadsheet.

## Features

*   **Reads Multiple Formats:** Processes both `.txt` and `.pdf` files.
*   **Automatic Directory Creation:** Automatically creates the necessary `input` and `output_rag` directories if they don't exist.
*   **Text Cleaning:** Optionally removes common "References" or "Bibliography" sections before processing.
*   **Intelligent Chunking:** Splits documents into manageable chunks suitable for embedding and retrieval.
*   **Local Embeddings:** Uses `sentence-transformers` to generate embeddings locally for document chunks and the query.
*   **Relevant Context Retrieval:** Identifies the most relevant document chunks based on semantic similarity to the query.
*   **Local LLM Generation:** Utilizes Ollama (with a model like `gemma3:1b` by default) to generate answers based *only* on the retrieved relevant context and the user's query.
*   **Organized Output:** Saves each answer to a separate `.txt` file and compiles all filenames and answers into a summary `.xlsx` file.

## How it Works (RAG Process)

1.  **Directory Check:** Ensures the `input` and `output_rag` directories exist, creating them if necessary.
2.  **Read & Clean:** The script reads the content from each supported file found in the `input` folder. It attempts to remove trailing reference sections.
3.  **Chunk:** The cleaned text is divided into smaller, overlapping (implicitly by paragraph breaks) chunks.
4.  **Embed:** Each chunk is converted into a numerical vector (embedding) using a `sentence-transformers` model (`all-MiniLM-L6-v2` by default).
5.  **Retrieve:** The user's query is also embedded. The script calculates the cosine similarity between the query embedding and all chunk embeddings to find the `top_k` (default 3) most relevant chunks.
6.  **Generate:** The retrieved chunks (context) and the original query are formatted into a prompt. This prompt is sent to the specified Ollama model (`gemma3:1b` by default) to generate a concise answer based *solely* on the provided context.
7.  **Save:** The generated answer is saved to a `.txt` file (in the `output_rag` folder) named after the original document, and all results are aggregated into an Excel file (`summaries.xlsx`) in the same output folder.

## Requirements

*   **Python:** Version 3.8 or higher recommended.
*   **Ollama:** You need Ollama installed and running on your system. Download it from [https://ollama.com/](https://ollama.com/).
*   **Ollama Model:** The script requires the LLM specified (default: `gemma3:1b`). You need to pull it using Ollama *before* running the script:
    ```
    ollama pull gemma3:1b
    ```
*   **Python Libraries:** Listed in `requirements.txt`.

## Setup

1.  **Clone the Repository:**
    ```
    git clone https://github.com/Ahmed-Sabri/ollama-summerizaer
    cd ollama-summerizaer
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Install the requirements `pip install -r requirements.txt` file with the following content:
    ```
    numpy
    pandas
    ollama
    sentence-transformers
    PyPDF2
    openpyxl # Needed for saving Excel files
    ```
    Then install them:
    ```
    pip install -r requirements.txt
    ```

4.  **Install and Run Ollama:**
    *   Download and install Ollama from [https://ollama.com/](https://ollama.com/).
    *   Ensure the Ollama application or service is running in the background.
    *   Pull the required model:
        ```
        ollama pull gemma3:1b
        ```
        *(Note: The first time you run the Python script, `sentence-transformers` will automatically download the `all-MiniLM-L6-v2` embedding model.)*

5.  **Prepare Input Directory:**
    *   The script will automatically create an `input` folder in the same directory if it doesn't exist.
    *   **You must place the `.txt` and `.pdf` files you want to process into this `input` folder.**

## Usage

1.  **Place Documents:** Ensure your `.txt` and `.pdf` files are inside the `input` folder. If the folder doesn't exist, the script will create it on the first run, but it will be empty, so you'll need to add files and run it again.
2.  **Configure Query (Optional):**
    *   Open the Python script (`your_script_name.py`).
    *   Locate the `main()` function.
    *   Modify the `query` variable if you want to ask something different than the default summarization prompt:
        ```
        query = "What are the main security concerns mentioned in this document?" # Example
        ```
3.  **Run the Script:**
    *   Make sure Ollama is running.
    *   Execute the script from your terminal:
        ```
        python app.py
        ```
    
    *   The script will inform you if it creates the `input` or `output_rag` directories.
    *   If the `input` directory is empty, the script will notify you and exit gracefully.

4.  **Check Output:**
    *   The script will print progress messages to the console for each file processed.
    *   An `output_rag` folder will be present (created automatically if needed).
    *   Inside `output_rag`, you will find:
        *   Individual `.txt` files (e.g., `mydocument_rag_answer.txt`) containing the generated answer for each input file.
        *   An Excel file named `summaries.xlsx` listing all processed filenames and their corresponding generated answers.

## Configuration (Advanced)

You can modify the script's behavior by changing these variables/parameters:

*   **`input_folder` / `output_folder`:** Change the names of the input/output directories defined at the start of `main()`. The script will create these if they don't exist.
*   **`query`:** Modify the question asked about the documents in `main()`.
*   **`max_chunk_length`:** Adjust the target maximum character length for text chunks in `chunk_text()`.
*   **`embedder` model:** Change the Sentence Transformer model name string in `rag_summarize()` (e.g., `"multi-qa-MiniLM-L6-cos-v1"`). Ensure you choose a valid model from the `sentence-transformers` library.
*   **`top_k`:** Change the number of relevant chunks retrieved in `retrieve_relevant_chunks()`.
*   **`ollama.generate(model=...)`:** Change the Ollama model used for generation in `rag_summarize()`. Make sure you have pulled the specified model via `ollama pull <model_name>`.

## License

(Optional) Specify your license here (e.g., MIT, Apache 2.0). If you don't have one, you can omit this section or state "License unspecified".

