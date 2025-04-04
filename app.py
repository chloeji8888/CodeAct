import sys
import os
from io import StringIO
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Function to get API key from environment or user input
def get_api_key():
    """Get OpenAI API key from environment variable or user input."""
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # In Streamlit, get from session state or environment
    if not api_key and 'openai_api_key' in st.session_state:
        api_key = st.session_state['openai_api_key']
        
    return api_key

# Real OpenAI LLM function
def openai_llm(prompt):
    """Generate code using OpenAI API."""
    api_key = get_api_key()
    
    # If no API key is available, return an error message
    if not api_key:
        return "# Error: No OpenAI API key provided\nprint('Please provide an OpenAI API key to use this feature')"
    
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful coding assistant. Respond only with executable Python code without explanations or markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_message = f"Error calling OpenAI API: {str(e)}"
        st.error(error_message)
        return f"# {error_message}\nprint('Error connecting to OpenAI API')"

# Mock LLM for fallback or testing
def mock_llm(prompt):
    """Simulate an LLM by returning Python code for specific tasks."""
    if "sum of 3 and 5" in prompt:
        return "print(3 + 5)"
    elif "factorial of 5" in prompt:
        return """
result = 1
for i in range(1, 6):
    result *= i
print(result)
"""
    elif "square root of 16" in prompt:
        return "import math\nprint(math.sqrt(16))"
    else:
        return "print('Unknown task or example task')"

# Agent function to generate and execute code
def agent(task_prompt, use_openai=True):
    """Generate Python code from the task prompt and execute it."""
    # Get the generated code from the LLM
    if use_openai:
        generated_code = openai_llm(task_prompt)
    else:
        generated_code = mock_llm(task_prompt)
    
    # Redirect stdout to capture output
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    
    # Execute the code with error handling
    try:
        exec(generated_code)
        output = buffer.getvalue()
    except Exception as e:
        output = f"Error: {str(e)}"
    finally:
        # Restore original stdout
        sys.stdout = old_stdout
    
    return generated_code, output

# Initialize session state variables
def init_session_state():
    if 'use_openai' not in st.session_state:
        st.session_state['use_openai'] = True
    if 'task_history' not in st.session_state:
        st.session_state['task_history'] = []

# Streamlit app
def main():
    st.set_page_config(
        page_title="CodeAct Agent with OpenAI",
        page_icon="🤖",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("🤖 CodeAct Agent with OpenAI")
    st.markdown("""
    This app generates and executes Python code based on your task description.
    Enter a task like "calculate the sum of 3 and 5" or "calculate the factorial of 5".
    """)
    
    # API Key input in sidebar
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("OpenAI API Key", type="password", 
                               help="Enter your OpenAI API key here. It will not be stored permanently.")
        if api_key:
            st.session_state['openai_api_key'] = api_key
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Toggle between OpenAI and Mock LLM
        st.subheader("LLM Selection")
        use_openai = st.toggle("Use OpenAI API", value=st.session_state['use_openai'])
        st.session_state['use_openai'] = use_openai
        st.write(f"Using: **{'OpenAI' if use_openai else 'Mock'} LLM**")
        
        # Show a warning if OpenAI is selected but no API key
        if use_openai and not get_api_key():
            st.warning("⚠️ Please provide an OpenAI API key to use OpenAI's models.")
    
    # Main area for task input and results
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Task Description")
        task_description = st.text_area(
            "Enter your task:",
            placeholder="E.g., calculate the sum of 3 and 5",
            height=100
        )
        
        submit_button = st.button("Generate & Execute Code", type="primary")
    
    # Handle task submission
    if submit_button and task_description:
        # Construct the prompt for the LLM
        prompt = f"Write a Python code snippet that {task_description} and prints the result. Respond only with executable code, no explanations."
        
        # Add to history
        task_entry = {"task": task_description, "timestamp": pd.Timestamp.now()}
        
        with st.spinner("Generating and executing code..."):
            try:
                generated_code, execution_result = agent(prompt, st.session_state['use_openai'])
                
                # Store results in the task entry
                task_entry["code"] = generated_code
                task_entry["result"] = execution_result
                
                # Add to history
                st.session_state['task_history'].insert(0, task_entry)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Display results in the second column
    with col2:
        if st.session_state['task_history']:
            latest = st.session_state['task_history'][0]
            
            st.subheader("Generated Code")
            st.code(latest["code"], language="python")
            
            st.subheader("Execution Result")
            st.text(latest["result"])
    
    # Show history
    if st.session_state['task_history']:
        st.header("Task History")
        for i, entry in enumerate(st.session_state['task_history']):
            if i > 0:  # Skip the most recent one as it's already displayed
                with st.expander(f"Task: {entry['task']} ({entry['timestamp'].strftime('%H:%M:%S')})"):
                    st.code(entry["code"], language="python")
                    st.text(f"Result: {entry['result']}")

if __name__ == "__main__":
    import pandas as pd
    main()