import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

def summarize_pdf_with_llama(pdf_path, llama_model="llama3.2:1b"):
    """
    Summarizes a PDF document using a local Llama model via LangChain's load_summarize_chain.
    """
    # 1. Load the PDF document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split the document into chunks (essential for large PDFs and LLM context windows)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    docs = text_splitter.split_documents(documents)

    # 3. Initialize the local Llama LLM (e.g., "llama3")
    # Make sure Ollama is running and the model is pulled
    llm = Ollama(model=llama_model)

    # 4. Define a custom prompt for the map_reduce chain
    # The 'map_reduce' approach is excellent for long documents
    prompt_template = """
    Summarize the following text in a concise and accurate way.
    Focus on key points and main ideas.

    "{text}"

    CONCISE SUMMARY:"""
    custom_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    # 5. Load the summarization chain
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=custom_prompt,
        combine_prompt=custom_prompt,
        verbose=True # Set to True to see intermediate steps
    )

    # 6. Run the chain and get the summary
    summary = chain.invoke({"input_documents": docs}, return_only_outputs=True)
    return summary['output_text']

# Example usage:
# Make sure you have a 'your_document.pdf' file in your directory
pdf_file_path = "whitepaper2.pdf"
try:
    final_summary = summarize_pdf_with_llama(pdf_file_path)
    print("\n--- Final Summary ---")
    print(final_summary)
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure Ollama is running and the specified Llama model is available.")
