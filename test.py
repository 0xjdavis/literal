import streamlit as st
from pinecone import Pinecone
import llama_index
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext, Document, VectorStoreIndex, set_global_handler
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core import Settings
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.llms.together import TogetherLLM  # Changed this import

TOGETHER_API_KEY = st.secrets['TOGETHER_API_KEY']
LLAMA_MODEL = "meta-llama/Llama-Vision-Free"

def completion_to_prompt(completion: str) -> str:
    return f"<s>[INST] {completion} [/INST] </s>\n"

def run_rag_completion(
    document_dir: str,
    query_text: str,
    embedding_model: str ="togethercomputer/m2-bert-80M-8k-retrieval",
    generative_model: str ="mistralai/Mixtral-8x7B-Instruct-v0.1"
    ) -> str:

    # Configure the embedding model before loading documents
    Settings.embed_model = TogetherEmbedding(
        model_name=embedding_model
    )

    # Initialize the LLM using TogetherLLM
    llm = TogetherLLM(
        model=LLAMA_MODEL,
        temperature=0.1,
        max_tokens=512,
        api_key=TOGETHER_API_KEY
    )
    
    # Set the LLM in Settings
    Settings.llm = llm

    documents = SimpleDirectoryReader(document_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    response = index.as_query_engine(similarity_top_k=5).query(query_text)

    return str(response)

query_text = "How many asylum cases are pending at present?"
document_dir = "data"

response = run_rag_completion(document_dir, query_text)
st.write(response)