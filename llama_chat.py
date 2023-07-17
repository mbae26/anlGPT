import os
from llama_index import SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index import ServiceContext 
from llama_index import GPTVectorStoreIndex
import openai
import logging
import sys

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ["OPENAI_API_KEY"] = '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'

def build_storage(data_dir):
    # Load the documents
    documents = SimpleDirectoryReader(data_dir).load_data()
    # Construct index 
    index = GPTVectorStoreIndex.from_documents(documents)
    # Store index in the storage
    index.storage_context.persist()
    return index

def read_from_storage(persist_dir):
    # Access to the storage
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    # Return the index from the storage
    return load_index_from_storage(storage_context)

def adding_data_to_GPT():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    # Relative path to the directory of data
    data_dir = "./pdfs/txts"
    # Relative path to the storage of index
    persist_dir = "./storage"
    index = None
    
    if os.path.exists(persist_dir):
        index = read_from_storage(persist_dir)
    else:
        index = build_storage(data_dir)
    # streaming=True -> the generation of the responses is displayed.  
    # mix_rate = 0.9 -> gives more weight to the responses from the indexed documents
    chat_engine = index.as_chat_engine(
        chat_mode='condense_question',
        verbose=True,
        mix_rate=0.9
        # streaming=True
    )
    
    while True:    
        query = input("chat with GPT (or type 'exit' to stop): ")
        
        if query.lower() == "exit":
            break
        
        response = chat_engine.chat(query)
        print(response)
        # response.print_response_stream()
        print('\n')
        print("----------------------------------------------------------------")
    
    
if __name__ == '__main__':
    adding_data_to_GPT()