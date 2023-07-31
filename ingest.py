import logging
import os
import constants

import click
import torch
import chromadb
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredXMLLoader
from langchain.vectorstores import Chroma
from chromadb.config import Settings

SOURCE_DIRECTORY = f"{constants.ROOT_DIRECTORY}/data/grobid_files"
PERSIST_DIRECTORY = f"{constants.ROOT_DIRECTORY}/DB"
client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

def split_documents(documents: list[Document]) -> list[Document]:
    # split documents into chunks 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_documents = text_splitter.split_documents(documents)
    return text_documents

def load_documents(source_dir: str) -> list[Document]:
    # Load documents from 'source_dir'
    documents = []
    for filename in os.listdir(source_dir):
        # Load XML files (Grobid output)
        if filename.endswith(".xml"):
            document = UnstructuredXMLLoader(os.path.join(source_dir, filename))
            documents.extend(document.load())
        else:
            raise ValueError("Unsupported file type.")
    return documents

# This decorator tells Click that the function below is a command line command    
@click.command()
# Specify options for the command
@click.option(
    "--device-type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(["cpu", "cuda"]),
    help="Device to run on. Defaults to 'cuda' if available, otherwise 'cpu'."
)
def main(device_type):
    # Load documents and split them into chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}...")
    documents = load_documents(SOURCE_DIRECTORY)
    logging.info(f"Loaded {len(documents)} documents.")
    text_documents = split_documents(documents)
    logging.info(f"Split into {len(text_documents)} chunks of text.")
    
    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=constants.EMBEDDING_MODEL_NAME,
    )
    
    # Create vector store (Chroma)
    db = Chroma.from_documents(
        text_documents, 
        embeddings,
        client=client,
        collection_name="instructor_embeddings",
        persist_directory=PERSIST_DIRECTORY,
    )
    db.persist()
    db = None
    
if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s -  %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()