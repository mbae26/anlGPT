import os
import sys
import model_const
import openai
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader
from langchain.retrievers.multi_query import MultiQueryRetriever


load_dotenv()
api = os.getenv('OPEN_AI_KEY')
os.environ["OPENAI_API_KEY"] = api

llm = ChatOpenAI(temperature=0, model_name='gpt-4')

def load_docs(data_dir):
    text_loader = DirectoryLoader(data_dir, glob="*.txt")
    documents = []
    documents.extend(text_loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    chunked_documents = text_splitter.split_documents(documents)
    
    return chunked_documents

def chatbot(question):
    # gpt-4-32k
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    qa = RetrievalQA.from_chain_type(llm, "stuff", retriever=vectordb.as_retriever(search_kwargs={'k': 5}))
    # stop if input is exit, quit, q, or f
    if question == "exit" or question == "quit" or question == "q" or question == "f":
        print("Exiting...")
        sys.exit()
    # skip if input is empty
    if question == '':
        return
    response = qa.run(question)
    
    return response

if __name__ == "__main__":
    # store path to data and storage directory
    data_dir = '/Users/minseokbae/ANL/gpt3_finetune/data/txts'
    
    persist_dir = "./chroma_db/first_13_papers"
    
    chunked_documents = load_docs(data_dir)
    
    vectordb = Chroma.from_documents(
        chunked_documents,
        OpenAIEmbeddings(),
        collection_name="first_13_papers",
        persist_directory=persist_dir,
    )
    
    vectordb.persist()
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Start conversations
    while True:
        question = input("chat with GPT (or type 'exit' to stop): ")
        print("Response: " + chatbot(question) + "\n")
        print("---------------------------------------------------------------------------------------------------------------------------------")
