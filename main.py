import os
from llama_index import SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index import ServiceContext 
from llama_index import GPTVectorStoreIndex
import openai

os.environ["OPENAI_API_KEY"] = 'Your openAI API key'

def build_storage(data_dir):
    documents = SimpleDirectoryReader(data_dir).load_data()
    
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist()
    return index

def read_from_storage(persist_dir):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    return load_index_from_storage(storage_context)

def adding_data_to_GPT():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    data_dir = "./pdfs/txts"
    persist_dir = "./storage"
    index = None
    
    if os.path.exists(persist_dir):
        index = read_from_storage(persist_dir)
    else:
        index = build_storage(data_dir)
        query_engine = index.as_query_engine()
        
    query = input("Ask GPT a question: ")
    
    response = query_engine.query(query)
    
    print(response)
    
    
if __name__ == '__main__':
    adding_data_to_GPT()