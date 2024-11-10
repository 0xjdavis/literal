import streamlit as st
from pinecone import Pinecone
import llama_index
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import (
    SimpleDirectoryReader, StorageContext, Document, 
    VectorStoreIndex, set_global_handler
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core import Settings
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.llms.together import TogetherLLM
from datetime import datetime
import json
import pandas as pd

# Configuration
TOGETHER_API_KEY = st.secrets['TOGETHER_API_KEY']
LLAMA_MODEL = "meta-llama/Llama-Vision-Free"
DATA_DIR = "data"

def completion_to_prompt(completion: str) -> str:
    return f"<s>[INST] {completion} [/INST] </s>\n"

def run_rag_completion(
    document_dir: str,
    query_text: str,
    embedding_model: str = "togethercomputer/m2-bert-80M-8k-retrieval",
    generative_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
) -> str:
    Settings.embed_model = TogetherEmbedding(
        model_name=embedding_model
    )
    
    llm = TogetherLLM(
        model=LLAMA_MODEL,
        temperature=0.1,
        max_tokens=512,
        api_key=TOGETHER_API_KEY
    )
    
    Settings.llm = llm
    
    documents = SimpleDirectoryReader(document_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    response = index.as_query_engine(similarity_top_k=5).query(query_text)
    
    return str(response)

# New Functions for Script Testing
def generate_questions(index):
    """Generate questions based on script content"""
    query_engine = index.as_query_engine()
    questions_prompt = """
    Based on the script content, generate 10 challenging questions that will test 
    someone's ability to stay on script. Questions should be specific and probe for 
    detailed knowledge of the script content. Format the output as a numbered list.
    """
    response = query_engine.query(questions_prompt)
    questions = str(response).split('\n')
    questions = [q.strip() for q in questions if q.strip() and any(c.isdigit() for c in q[:2])]
    return questions[:10]

def evaluate_response(index, question, response):
    """Evaluate response alignment with script"""
    query_engine = index.as_query_engine()
    evaluation_prompt = f"""
    Question: {question}
    Response: {response}
    
    Evaluate if this response aligns with the script content. Consider:
    1. Accuracy of information
    2. Adherence to script messaging
    3. Completeness of response
    
    Return the evaluation in this JSON format:
    {{
        "aligned": true/false,
        "feedback": "detailed feedback here",
        "score": "numerical score 0-100"
    }}
    """
    result = query_engine.query(evaluation_prompt)
    return json.loads(str(result))

# Initialize Session State
if 'mode' not in st.session_state:
    st.session_state.mode = 'rag'  # Default to RAG mode
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'evaluation_complete' not in st.session_state:
    st.session_state.evaluation_complete = False
if 'index' not in st.session_state:
    st.session_state.index = None

# Streamlit UI
st.title("Literal")
st.subheader("Self Help for United States Asylum Seekers")
# Sidebar for mode selection and setup
with st.sidebar:
    st.header("Mode Selection")
    mode = st.radio("Select Mode", ['RAG Query', 'Script Testing'])
    st.session_state.mode = mode
    
    st.header("Upload Testimonal Script")
    uploaded_file = st.file_uploader("Upload Document", type=['txt', 'md', 'pdf'])
    if uploaded_file:
        with open(f"{DATA_DIR}/mock-testimonials/current_document.txt", 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Initialize index
        Settings.embed_model = TogetherEmbedding(
            model_name="togethercomputer/m2-bert-80M-8k-retrieval"
        )
        llm = TogetherLLM(
            model=LLAMA_MODEL,
            temperature=0.1,
            max_tokens=512,
            api_key=TOGETHER_API_KEY
        )
        Settings.llm = llm
        
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        st.session_state.index = VectorStoreIndex.from_documents(documents)
        st.success("Document processed successfully!")

# Main content area
if st.session_state.mode == 'RAG Query':
    st.header("RAG Query Mode")
    query_text = st.text_input("Enter your query:")
    if st.button("Run Query") and query_text:
        response = run_rag_completion(DATA_DIR, query_text)
        st.write(response)

elif st.session_state.mode == 'Testimonial Practice':
    st.subheader("Testimionial Practice")
    
    if st.session_state.index:
        if len(st.session_state.questions) == 0:
            if st.button("Generate Questions"):
                with st.spinner("Generating questions..."):
                    st.session_state.questions = generate_questions(st.session_state.index)
        
        if len(st.session_state.questions) > 0 and not st.session_state.evaluation_complete:
            st.subheader(f"Question {st.session_state.current_question + 1} of 10")
            st.write(st.session_state.questions[st.session_state.current_question])
            
            response = st.text_area("Your response:")
            if st.button("Submit Response"):
                if response:
                    st.session_state.responses.append(response)
                    
                    if st.session_state.current_question < 9:
                        st.session_state.current_question += 1
                    else:
                        with st.spinner("Evaluating responses..."):
                            evaluations = []
                            total_score = 0
                            
                            for q, r in zip(st.session_state.questions, st.session_state.responses):
                                eval_result = evaluate_response(st.session_state.index, q, r)
                                evaluations.append(eval_result)
                                total_score += float(eval_result['score'])
                            
                            st.session_state.evaluation_complete = True
                            st.session_state.final_results = {
                                "evaluations": evaluations,
                                "final_score": total_score / 10
                            }
                            st.experimental_rerun()
        
        elif st.session_state.evaluation_complete:
            st.header("Test Results")
            st.subheader(f"Final Score: {st.session_state.final_results['final_score']:.1f}%")
            
            results_df = pd.DataFrame({
                "Question": st.session_state.questions,
                "Response": st.session_state.responses,
                "Score": [eval_result['score'] for eval_result in st.session_state.final_results['evaluations']],
                "Feedback": [eval_result['feedback'] for eval_result in st.session_state.final_results['evaluations']]
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            if st.button("Start New Test"):
                st.session_state.current_question = 0
                st.session_state.questions = []
                st.session_state.responses = []
                st.session_state.evaluation_complete = False
                st.experimental_rerun()
    
    else:
        st.info("Please upload a script document to begin testing.")