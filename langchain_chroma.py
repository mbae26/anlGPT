import os
import sys
import constants
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
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)
    
    return chunked_documents

def print_sources(response):
    for source in response['source_documents']:
        print(source.metadata['source'])
        
    return 

def chatbot(question):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=constants.QA_TEMPLATE)
    
    qa_chain = RetrievalQA.from_chain_type(llm, "stuff", retriever=vectordb.as_retriever(search_kwargs={'k': 5}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}, return_source_documents=True)
    
    # stop if input is exit, quit, q, or f
    if question == "exit" or question == "quit" or question == "q" or question == "f":
        print("Exiting...")
        sys.exit()
    # skip if input is empty
    if question == '':
        return
    response = qa_chain({"query": question})
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
        response = chatbot(question)
        print("Response: " + response['result'] + "\n")
        print_sources(response)
        print("---------------------------------------------------------------------------------------------------------------------------------")
