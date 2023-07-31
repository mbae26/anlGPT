import os
import sys
import constants
import openai
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredXMLLoader
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.retrievers.document_compressors import EmbeddingsFilter

load_dotenv()
api = os.getenv('OPEN_AI_KEY')
os.environ["OPENAI_API_KEY"] = api
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

def load_docs(data_dir):
    # Load the documents
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".xml"):
            loader = UnstructuredXMLLoader(os.path.join(data_dir, file))
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

def chatbot(question):
    prompt = PromptTemplate(
        input_variables=['question'],
        template=constants.TEMPLATE,
    )
    
    # embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    # compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)

    # compressed_docs = compression_retriever.get_relevant_documents("What did the president say about Ketanji Jackson Brown")
    # pretty_print_docs(compressed_docs)
    
    question_generator = LLMChain(llm=llm, prompt=prompt) 
    doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    # stop if input is exit, quit, q, or f
    if question == "exit" or question == "quit" or question == "q" or question == "f":
        print("Exiting...")
        sys.exit()
    # skip if input is empty
    if question == '':
        return
    chain = ConversationalRetrievalChain(
        retriever=db.as_retriever(search_kwargs={'k': 4}),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
    )
    response = chain({"question": question, "chat_history": chat_history})
    # Add response to chat history
    chat_history.append((question, response["answer"]))
    return response["answer"]

if __name__ == "__main__":
    # store path to data and storage directory
    data_dir = "./data/grobid_files"
    persist_dir = "./faiss_grobid"
    
    if not os.path.exists(persist_dir):
        chunked_documents = load_docs(data_dir)
        db = FAISS.from_documents(chunked_documents, embedding=OpenAIEmbeddings())
        os.makedirs(persist_dir)  # Create persist_dir before saving
        db.save_local(os.path.join(persist_dir, "faiss_index"))  # Save faiss_index inside persist_dir

    else:
        db = FAISS.load_local(os.path.join(persist_dir, "faiss_index"), embeddings=OpenAIEmbeddings())

    chat_history = []
    # Start conversations
    while True:
        question = input("chat with GPT (or type 'exit' to stop): ")
        print("Response: " + chatbot(question) + "\n")
        print("---------------------------------------------------------------------------------------------------------------------------------")
