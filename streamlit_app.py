import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from streamlit_webrtc import webrtc_streamer
from together import Together
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings import HuggingFaceEmbedding
import os

# Configure Together AI client
client = Together()

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

def generate_questions(script_content):
    """Generate questions using Together AI"""
    try:
        prompt = f"""Given this script, generate 10 challenging questions that will test 
        someone's ability to stay on script. Questions should be specific and probe for 
        detailed knowledge of the script content.

        Script content:
        {script_content}

        Generate 10 numbered questions that will test how well someone knows and can stick to this script.
        Make the questions challenging and specific to the script content."""
        
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are an expert interviewer creating questions to test script adherence."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract questions from response
        questions_text = response.choices[0].message.content
        questions = [q.strip() for q in questions_text.split('\n') if q.strip() and any(c.isdigit() for c in q[:2])]
        return questions[:10]  # Ensure we have exactly 10 questions
    
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return []

def evaluate_response(script_content, question, response):
    """Evaluate if the response aligns with the script using Together AI"""
    try:
        evaluation_prompt = f"""
        Evaluate if this response aligns with the given script.

        Script content:
        {script_content}

        Question: {question}
        Response: {response}

        Evaluate the response considering:
        1. Accuracy of information compared to the script
        2. Adherence to script messaging
        3. Completeness of response

        Provide your evaluation in valid JSON format with these exact keys:
        {{"aligned": true/false, "feedback": "detailed feedback", "score": numeric_score_0_to_100}}
        """

        eval_response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            messages=[
                {"role": "system", "content": "You are an expert evaluator assessing script adherence. Provide evaluation in valid JSON format."},
                {"role": "user", "content": evaluation_prompt}
            ]
        )
        
        # Parse the JSON response
        eval_text = eval_response.choices[0].message.content
        return json.loads(eval_text)
    
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
        script_content = uploaded_file.getvalue().decode('utf-8')
        st.session_state.script_content = script_content
        
        # Save uploaded file
        script_path = DATA_DIR / "current_script.txt"
        with open(script_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        if st.button("Generate Questions"):
            with st.spinner("Generating questions..."):
                st.session_state.questions = generate_questions(script_content)
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
if st.session_state.script_content and len(st.session_state.questions) > 0:
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
                            eval_result = evaluate_response(st.session_state.script_content, q, r)
                            evaluations.append(eval_result)
                            total_score += float(eval_result['score'])
                        
                        final_score = total_score / 10
                        
                        # Save interview data
                        interview_data = {
                            "script": st.session_state.script_content,
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
