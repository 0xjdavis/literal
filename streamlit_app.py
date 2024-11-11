import streamlit as st
from pathlib import Path
import os
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, Settings,
    ServiceContext
)
from llama_index.embeddings.together import TogetherEmbedding
from llama_index.llms.together import TogetherLLM
import time
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
TOGETHER_API_KEY = st.secrets['TOGETHER_API_KEY']
# LLAMA_MODEL = "meta-llama/Llama-Vision-Free"
LLAMA_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
DATA_DIR = Path("data")
MAX_RETRIES = 3
RETRY_DELAY = 2
LLM_TIMEOUT = 30

class DocumentProcessor:
    """Handles document processing and storage operations"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.ensure_directories()
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            os.makedirs(self.data_dir / "results", exist_ok=True)
            logger.info(f"Directories created/verified at {self.data_dir}")
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            raise
    
    def save_document(self, uploaded_file) -> Path:
        """Save uploaded document with error handling and validation"""
        try:
            file_path = self.data_dir / "mock-testimonials" / "current_document.txt"
            
            # Ensure the mock-testimonials directory exists
            os.makedirs(self.data_dir / "mock-testimonials", exist_ok=True)
            
            content = uploaded_file.getvalue()
            
            # Validate content
            if not content:
                raise ValueError("Empty document uploaded")
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            logger.info(f"Document saved successfully to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving document: {e}")
            raise
        
class LLMManager:
    """Manages LLM operations with retry logic and error handling"""
    
    def __init__(self):
        self.setup_complete = False
    
    def setup_llm(self) -> bool:
        """Initialize LLM with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                Settings.embed_model = TogetherEmbedding(
                    model_name="togethercomputer/m2-bert-80M-8k-retrieval"
                )
                llm = TogetherLLM(
                    model=LLAMA_MODEL,
                    temperature=0.1,
                    max_tokens=512,
                    api_key=TOGETHER_API_KEY,
                    timeout=LLM_TIMEOUT
                )
                Settings.llm = llm
                self.setup_complete = True
                logger.info("LLM setup completed successfully")
                return True
            except Exception as e:
                logger.warning(f"LLM setup attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error("LLM setup failed after all retries")
                    raise
    
    @staticmethod
    def run_with_retry(func, *args, **kwargs) -> any:
        """Execute LLM operations with retry logic"""
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Operation attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error("Operation failed after all retries")
                    raise

class ScriptTester:
    """Handles script testing functionality"""
    
    def __init__(self, index: VectorStoreIndex):
        self.index = index
    
    def generate_questions(self) -> List[str]:
        """Generate test questions with validation"""
        try:
            query_engine = self.index.as_query_engine()
            questions_prompt = """
            Based on the script content, generate 10 challenging questions that will test 
            someone's ability to stay on script. Questions should be specific and probe for 
            detailed knowledge of the script content. Format the output as a numbered list.
            """
            response = LLMManager.run_with_retry(
                query_engine.query, questions_prompt
            )
            
            questions = str(response).split('\n')
            questions = [q.strip() for q in questions if q.strip() and any(c.isdigit() for c in q[:2])]
            
            if len(questions) < 10:
                raise ValueError("Insufficient questions generated")
                
            return questions[:10]
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            raise
    
    def evaluate_response(self, question: str, response: str) -> Dict:
        """Evaluate response with safe JSON parsing"""
        try:
            query_engine = self.index.as_query_engine()
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
            
            result = LLMManager.run_with_retry(
                query_engine.query, evaluation_prompt
            )
            
            try:
                evaluation = json.loads(str(result))
                # Validate evaluation structure
                required_keys = {"aligned", "feedback", "score"}
                if not all(key in evaluation for key in required_keys):
                    raise ValueError("Invalid evaluation structure")
                return evaluation
            except json.JSONDecodeError:
                logger.error("JSON parsing failed for evaluation")
                return {
                    "aligned": False,
                    "feedback": "Error parsing evaluation response",
                    "score": 0
                }
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return {
                "aligned": False,
                "feedback": f"Evaluation error: {str(e)}",
                "score": 0
            }

class SessionManager:
    """Manages Streamlit session state"""
    
    @staticmethod
    def initialize_session_state():
        """Initialize all session state variables"""
        defaults = {
            'mode': 'rag',
            'current_question': 0,
            'questions': [],
            'responses': [],
            'evaluation_complete': False,
            'index': None,
            'final_results': None,
            'error_state': None
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def reset_test_state():
        """Reset test-related session state"""
        st.session_state.current_question = 0
        st.session_state.questions = []
        st.session_state.responses = []
        st.session_state.evaluation_complete = False
        st.session_state.final_results = None
        st.session_state.error_state = None

def main():
    """Main application logic"""
    try:
        # Initialize session state
        SessionManager.initialize_session_state()
        
        # Initialize components
        doc_processor = DocumentProcessor(DATA_DIR)
        llm_manager = LLMManager()
        
        st.title("Literal")
        st.write("Self Help for 🇺🇸 United States Asylum Seekers")
        
        # Initialize base index for Q&A mode if not already done
        if 'base_index' not in st.session_state:
            with st.spinner("Loading knowledge base..."):
                try:
                    if llm_manager.setup_llm():
                        # Load documents from data directory, excluding the results and mock-testimonials subdirectories
                        documents = SimpleDirectoryReader(
                            input_dir=str(DATA_DIR),
                            exclude_hidden=True,
                            recursive=True,
                            filename_as_id=True,
                            required_exts=['.txt', '.md', '.pdf']
                        ).load_data()
                        if not documents:
                            raise ValueError("No documents found in data directory")
                        st.session_state.base_index = VectorStoreIndex.from_documents(documents)
                        logger.info(f"Base knowledge index created successfully with {len(documents)} documents")
                except Exception as e:
                    logger.error(f"Error loading knowledge base: {traceback.format_exc()}")
                    st.error("Error loading knowledge base...")
                    return
    
        # Sidebar setup
        with st.sidebar:
            st.header("Mode Selection")
            mode = st.radio("Select Mode", ['Q&A', 'Testimonial Practice'])
            st.session_state.mode = mode
            
            # Only show document upload for Testimonial Practice mode
            if mode == 'Testimonial Practice':
                st.header("Document Upload")
                uploaded_file = st.file_uploader("Upload Testimonial", type=['txt', 'md', 'pdf'])
                
                if uploaded_file:
                    with st.spinner("Processing testimonial..."):
                        try:
                            file_path = doc_processor.save_document(uploaded_file)
                            if llm_manager.setup_llm():
                                documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
                                st.session_state.index = VectorStoreIndex.from_documents(documents)
                                st.success("Testimonial processed successfully!")
                        except Exception as e:
                            st.error(f"Error processing document: {str(e)}")
                            logger.error(f"Document processing error: {traceback.format_exc()}")
        
        # Main content area
        if st.session_state.mode == 'Q&A':
            handle_rag_mode(st.session_state.base_index)
        else:
            handle_script_testing_mode(st.session_state.index)
    
    except Exception as e:
        logger.error(f"Application error: {traceback.format_exc()}")
        st.error("An unexpected error occurred. Please try again or contact support.")

def handle_rag_mode(index: Optional[VectorStoreIndex]):
    """Handle RAG query mode"""
    st.subheader("Asylum Interview Q&A")
    
    if not index:
        st.error("Knowledge base not available. Please contact support.")
        return
    
    # Create a container for chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    query_engine = index.as_query_engine()
                    response = LLMManager.run_with_retry(
                        query_engine.query, prompt
                    )
                    st.markdown(str(response))
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": str(response)})
                except Exception as e:
                    logger.error(f"Query error: {traceback.format_exc()}")
                    st.error(f"Error processing query: {str(e)}")

def handle_script_testing_mode(index: Optional[VectorStoreIndex]):
    """Handle script testing mode"""
    st.header("Testimonial Practice")
    
    if not index:
        st.info("Please upload your testimonial document to begin practice.")
        return
    
    script_tester = ScriptTester(index)
    
    if len(st.session_state.questions) == 0:
        if st.button("Generate Questions"):
            with st.spinner("Generating questions..."):
                try:
                    st.session_state.questions = script_tester.generate_questions()
                except Exception as e:
                    logger.error(f"Question generation error: {traceback.format_exc()}")
                    st.error(f"Error generating questions: {str(e)}")
                    return
    
    if len(st.session_state.questions) > 0 and not st.session_state.evaluation_complete:
        display_current_question(script_tester)
    elif st.session_state.evaluation_complete:
        display_results()

def display_current_question(script_tester: ScriptTester):
    """Display current test question and handle response"""
    st.subheader(f"Question {st.session_state.current_question + 1} of 10")
    st.write(st.session_state.questions[st.session_state.current_question])
    
    response = st.text_area("Your response:")
    if st.button("Submit Response"):
        if not response:
            st.warning("Please provide a response before submitting.")
            return
        
        st.session_state.responses.append(response)
        
        if st.session_state.current_question < 9:
            st.session_state.current_question += 1
            st.rerun()
        else:
            process_test_completion(script_tester)

def process_test_completion(script_tester: ScriptTester):
    """Process test completion and calculate results"""
    with st.spinner("Evaluating responses..."):
        try:
            evaluations = []
            total_score = 0
            
            for q, r in zip(st.session_state.questions, st.session_state.responses):
                eval_result = script_tester.evaluate_response(q, r)
                evaluations.append(eval_result)
                total_score += float(eval_result['score'])
            
            st.session_state.evaluation_complete = True
            st.session_state.final_results = {
                "evaluations": evaluations,
                "final_score": total_score / 10
            }
            
            # Save results
            save_test_results(st.session_state.questions, 
                            st.session_state.responses, 
                            evaluations, 
                            total_score / 10)
            
            st.rerun()
        except Exception as e:
            logger.error(f"Test completion error: {traceback.format_exc()}")
            st.error(f"Error processing test results: {str(e)}")

def display_results():
    """Display test results"""
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
        SessionManager.reset_test_state()
        st.rerun()

def save_test_results(questions: List[str], 
                     responses: List[str], 
                     evaluations: List[Dict], 
                     final_score: float):
    """Save test results to file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "timestamp": timestamp,
            "questions": questions,
            "responses": responses,
            "evaluations": evaluations,
            "final_score": final_score
        }
        
        file_path = DATA_DIR / "results" / f"test_results_{timestamp}.json"
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving test results: {e}")
        # Don't raise the error since this is not critical for the user experience

if __name__ == "__main__":
    main()
