import streamlit as st
import openai
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from streamlit_webrtc import webrtc_streamer
import av
import threading
import queue
import time

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

# Create necessary directories
Path("data").mkdir(exist_ok=True)

def generate_questions(script_content):
    """Generate questions using OpenAI API based on the script content"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert interviewer. Generate 10 challenging questions that will test how well someone knows and can stick to a given script."},
                {"role": "user", "content": f"Generate 10 questions based on this script:\n\n{script_content}"}
            ]
        )
        questions = response.choices[0].message['content'].split('\n')
        questions = [q.strip() for q in questions if q.strip()][:10]
        return questions
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return []

def evaluate_response(script_content, question, response):
    """Evaluate if the response aligns with the script"""
    try:
        evaluation = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are evaluating if a response aligns with a given script. Return a JSON with 'aligned' (boolean) and 'feedback' (string)."},
                {"role": "user", "content": f"Script:\n{script_content}\n\nQuestion:{question}\n\nResponse:{response}"}
            ]
        )
        return json.loads(evaluation.choices[0].message['content'])
    except Exception as e:
        st.error(f"Error evaluating response: {str(e)}")
        return {"aligned": False, "feedback": "Error in evaluation"}

def save_interview_data(data):
    """Save interview data to a file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/interview_{timestamp}.json"
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
        script_content = uploaded_file.read().decode()
        st.session_state.script_content = script_content
        
        if st.button("Generate Questions"):
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
                    evaluations = []
                    score = 100
                    
                    for q, r in zip(st.session_state.questions, st.session_state.responses):
                        eval_result = evaluate_response(st.session_state.script_content, q, r)
                        evaluations.append(eval_result)
                        if not eval_result['aligned']:
                            score -= 10
                    
                    # Save interview data
                    interview_data = {
                        "script": st.session_state.script_content,
                        "questions": st.session_state.questions,
                        "responses": st.session_state.responses,
                        "evaluations": evaluations,
                        "score": score,
                        "timestamp": datetime.now().isoformat()
                    }
                    save_interview_data(interview_data)
                    
                    st.session_state.evaluation_complete = True
    
    else:
        # Display results
        st.header("Interview Results")
        
        # Score
        st.subheader(f"Final Score: {score}%")
        
        # Detailed evaluation
        results_df = pd.DataFrame({
            "Question": st.session_state.questions,
            "Response": st.session_state.responses,
            "Feedback": [eval_result['feedback'] for eval_result in evaluations]
        })
        
        st.table(results_df)
        
        if st.button("Start New Interview"):
            st.session_state.current_question = 0
            st.session_state.questions = []
            st.session_state.responses = []
            st.session_state.evaluation_complete = False
            st.experimental_rerun()

else:
    st.info("Please upload a script and generate questions to begin the interview.")
