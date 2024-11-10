import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from streamlit_webrtc import webrtc_streamer
import together
import llama_index




# Storage
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext, Document, VectorStoreIndex, set_global_handler, Settings

#from llama_index import  ServiceContext




from llama_index.llms import Together
from llama_index.embeddings import HuggingFaceEmbedding
import os


# RAG
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore

# LLM
# from llama_index.llms.openai import OpenAI

# Workflow
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
)

# Workflow Graph
from llama_index.utils.workflow import draw_all_possible_flows

# Embeddings
from llama_index.embeddings.openai import OpenAIEmbedding

# Configure Together AI
together.api_key = st.secrets["TOGETHER_API_KEY"]
LLAMA_MODEL = "meta-llama/llama-2-70b-chat"

# Initialize session state variables
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'responses' not in st.session_state:
    st.session_state.responses = []
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'evaluation_complete' not in st.session_state:
    st.session_state.evaluation_complete = False
if 'script_content' not in st.session_state:
    st.session_state.script_content = None
if 'index' not in st.session_state:
    st.session_state.index = None

# Create necessary directories
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def setup_llama_index():
    """Setup LlamaIndex with Together AI model"""
    llm = Together(
        model=LLAMA_MODEL,
        temperature=0.1,
        max_tokens=512
    )
    
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        chunk_size=1024,
        chunk_overlap=20
    )
    
    return service_context

def generate_questions(index):
    """Generate questions using LlamaIndex and Together AI"""
    try:
        # Create a query engine
        query_engine = index.as_query_engine()
        
        # Generate questions prompt
        questions_prompt = """
        Based on the script content, generate 10 challenging questions that will test 
        someone's ability to stay on script. Questions should be specific and probe for 
        detailed knowledge of the script content. Format the output as a numbered list.
        """
        
        # Get response from the model
        response = query_engine.query(questions_prompt)
        
        # Parse the response into individual questions
        questions = str(response).split('\n')
        questions = [q.strip() for q in questions if q.strip() and any(c.isdigit() for c in q[:2])]
        
        # Ensure we have exactly 10 questions
        questions = questions[:10]
        return questions
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return []

def evaluate_response(index, question, response):
    """Evaluate if the response aligns with the script using LlamaIndex"""
    try:
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
    except Exception as e:
        st.error(f"Error evaluating response: {str(e)}")
        return {"aligned": False, "feedback": "Error in evaluation", "score": 0}

def save_interview_data(data):
    """Save interview data to a file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = DATA_DIR / f"interview_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(data, f)

# Streamlit UI
st.title("Script Adherence Testing App")

# Sidebar
with st.sidebar:
    st.header("Setup")
    
    # Script upload
    uploaded_file = st.file_uploader("Upload Script", type=['txt'])
    if uploaded_file:
        # Save uploaded file
        script_path = DATA_DIR / "current_script.txt"
        with open(script_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        # Initialize LlamaIndex
        service_context = setup_llama_index()
        documents = SimpleDirectoryReader(input_files=[script_path]).load_data()
        st.session_state.index = VectorStoreIndex.from_documents(
            documents,
            service_context=service_context
        )
        
        if st.button("Generate Questions"):
            with st.spinner("Generating questions..."):
                st.session_state.questions = generate_questions(st.session_state.index)
                st.session_state.current_question = 0
                st.session_state.responses = []
                st.session_state.evaluation_complete = False

    # Video recording setup
    webrtc_ctx = webrtc_streamer(
        key="interview",
        video_processor_factory=None,
        async_processing=True
    )

# Main content area
if st.session_state.index and len(st.session_state.questions) > 0:
    if not st.session_state.evaluation_complete:
        # Display current question
        st.subheader(f"Question {st.session_state.current_question + 1} of 10")
        st.write(st.session_state.questions[st.session_state.current_question])
        
        # Response input
        response = st.text_area("Your response:")
        
        if st.button("Submit Response"):
            if response:
                st.session_state.responses.append(response)
                
                if st.session_state.current_question < 9:
                    st.session_state.current_question += 1
                else:
                    # Evaluate all responses
                    with st.spinner("Evaluating responses..."):
                        evaluations = []
                        total_score = 0
                        
                        for q, r in zip(st.session_state.questions, st.session_state.responses):
                            eval_result = evaluate_response(st.session_state.index, q, r)
                            evaluations.append(eval_result)
                            total_score += float(eval_result['score'])
                        
                        final_score = total_score / 10
                        
                        # Save interview data
                        interview_data = {
                            "questions": st.session_state.questions,
                            "responses": st.session_state.responses,
                            "evaluations": evaluations,
                            "final_score": final_score,
                            "timestamp": datetime.now().isoformat()
                        }
                        save_interview_data(interview_data)
                        
                        st.session_state.evaluation_complete = True
                        st.session_state.final_results = {
                            "evaluations": evaluations,
                            "final_score": final_score
                        }
    
    else:
        # Display results
        st.header("Interview Results")
        
        # Score
        st.subheader(f"Final Score: {st.session_state.final_results['final_score']:.1f}%")
        
        # Detailed evaluation
        results_df = pd.DataFrame({
            "Question": st.session_state.questions,
            "Response": st.session_state.responses,
            "Score": [eval_result['score'] for eval_result in st.session_state.final_results['evaluations']],
            "Feedback": [eval_result['feedback'] for eval_result in st.session_state.final_results['evaluations']]
        })
        
        st.dataframe(results_df, use_container_width=True)
        
        if st.button("Start New Interview"):
            st.session_state.current_question = 0
            st.session_state.questions = []
            st.session_state.responses = []
            st.session_state.evaluation_complete = False
            st.experimental_rerun()

else:
    st.info("Please upload a script and generate questions to begin the interview.")
