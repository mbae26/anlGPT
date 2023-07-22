import os
import sys
import openai
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain

load_dotenv()
api = os.getenv('OPEN_AI_KEY')
os.environ["OPENAI_API_KEY"] = api

def load_docs(data_dir):
    # Load the documents
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(data_dir, file))
            documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents


if __name__ == "__main__":
    # store path to data and storage directory
    data_dir = "./data/txts"
    #persist_dir = "./chroma"
    persist_dir = "./faiss"
    
    if not os.path.exists(persist_dir):
        chunked_documents = load_docs(data_dir)
        db = FAISS.from_documents(chunked_documents, embedding=OpenAIEmbeddings())
        os.makedirs(persist_dir)  # Create persist_dir before saving
        db.save_local(os.path.join(persist_dir, "faiss_index"))  # Save faiss_index inside persist_dir

        #vectordb = Chroma.from_documents(chunked_documents, embeddings=OpenAIEmbeddings, persist_directory=persist_dir)
        #vectordb.persist()
    else:
        #vectordb = Chroma(persist_directory=persist_dir)
        db = FAISS.load_local(os.path.join(persist_dir, "faiss_index"), embeddings=OpenAIEmbeddings())
    
    #qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo"), vectordb.as_retriever(search_kwargs={'k': 6}), return_source_documents=True, verbose=False)
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo"), db.as_retriever())
    
    chat_history = []
    # Start conversations
    while True:
        query = input("chat with GPT (or type 'exit' to stop): ")
        # stop if input is exit, quit, q, or f
        if query == "exit" or query == "quit" or query == "q" or query == "f":
            print("Exiting...")
            sys.exit()
        # skip if input is empty
        if query == '':
            continue
        # get response from GPT and print it out
        response = qa({'question': query, 'chat_history': chat_history})
        print("Answer: " + response["answer"])
        # Add response to chat history
        chat_history.append((query, response["answer"]))
        print()
        print("---------------------------------------------------------------------------------------------------------------------------------")
