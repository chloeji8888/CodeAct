import streamlit as st
import sys
from io import StringIO
import json
import time
import os

# Try to import word2number, install if not available
try:
    from word2number import w2n
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "word2number"])
    from word2number import w2n

# Page configuration
st.set_page_config(
    page_title="CodeAct Agent UI",
    page_icon="🤖",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .code-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        font-family: monospace;
        white-space: pre-wrap;
        margin-bottom: 10px;
    }
    .success {
        color: green;
        font-weight: bold;
    }
    .error {
        color: red;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

TASK_TEMPLATES = {
    "calculate_average": {
        "description": "Calculate the average of numeric values in a list",
        "default_data": "[1, 'two', 3, None, 5]",
        "initial_code": "print(sum(data) / len(data))"
    },
    "find_max": {
        "description": "Find the maximum numeric value in a list",
        "default_data": "[10, 'large', -5, None, 42, '100']",
        "initial_code": "print(max(data))"
    },
    "count_by_type": {
        "description": "Count the occurrences of each data type in a list",
        "default_data": "[1, 'two', 3.0, None, True, [1,2], {'a': 1}]",
        "initial_code": "print({str(type(x)): 1})"
    }
}

# Function to convert word numbers to integers
def word_to_number(word):
    """Convert word representation of numbers to actual numbers using word2number library."""
    if isinstance(word, str):
        try:
            return w2n.word_to_num(word)
        except ValueError:
            # If word2number can't convert it, return None
            return None
    return None

# Default error handling strategies
DEFAULT_STRATEGIES = {
    "type_error": """
# Convert string numbers to actual numbers and filter non-numeric values
numeric_data = []
for x in data:
    if isinstance(x, (int, float)):
        numeric_data.append(x)
    elif isinstance(x, str):
        # First try direct conversion to float
        try:
            numeric_data.append(float(x))
            continue
        except ValueError:
            pass
            
        # Then try word-to-number conversion using the library
        try:
            num_value = w2n.word_to_num(x)
            numeric_data.append(num_value)
            continue
        except ValueError:
            pass
    else:
        # Try other conversions
        try:
            numeric_data.append(float(x))
        except (ValueError, TypeError):
            pass  # Skip items that can't be converted

if numeric_data:
    if task == "calculate_average":
        print(sum(numeric_data) / len(numeric_data))
    elif task == "find_max":
        print(max(numeric_data))
    elif task == "count_by_type":
        type_counts = {}
        for item in data:
            item_type = str(type(item))
            if item_type in type_counts:
                type_counts[item_type] += 1
            else:
                type_counts[item_type] = 1
        print(type_counts)
    else:
        print(sum(numeric_data) / len(numeric_data))  # Default to average
else:
    print('No valid numbers')
""",
    "zero_division": """
# Handle division by zero (empty list or all zeros)
numeric_data = []
for x in data:
    if isinstance(x, (int, float)):
        numeric_data.append(x)
    else:
        try:
            numeric_data.append(float(x))
        except (ValueError, TypeError):
            pass

if not numeric_data:
    print('No numeric values')
elif task == "calculate_average" and sum(numeric_data) == 0 and len(numeric_data) > 0:
    print('Average is 0.0')
elif task == "find_max":
    if numeric_data:
        print(max(numeric_data))
    else:
        print('No values to find maximum')
elif task == "count_by_type":
    type_counts = {}
    for item in data:
        item_type = str(type(item))
        if item_type in type_counts:
            type_counts[item_type] += 1
        else:
            type_counts[item_type] = 1
    print(type_counts)
else:  # Default to average
    if numeric_data:
        print(sum(numeric_data) / len(numeric_data))
    else:
        print('No numeric values')
""",
    "name_error": """
# Handle undefined variable errors
try:
    if 'data' in globals() or 'data' in locals():
        print(f"Data exists but may be inaccessible: {type(data)}")
    else:
        print('Error: data not found')
except:
    print('Error: Cannot access data variable')
""",
    "index_error": """
# Handle index out of range errors
if not data:
    print('Empty data')
elif len(data) == 0:
    print('Empty list')
else:
    # Convert string numbers to actual numbers and filter non-numeric values
    numeric_data = []
    for i, x in enumerate(data):
        try:
            if isinstance(x, (int, float)):
                numeric_data.append(x)
            else:
                numeric_data.append(float(x))
        except (ValueError, TypeError):
            pass  # Skip values that can't be converted
    
    if not numeric_data:
        print('No numeric values')
    elif task == "calculate_average":
        print(sum(numeric_data) / len(numeric_data))
    elif task == "find_max":
        print(max(numeric_data))
    elif task == "count_by_type":
        type_counts = {}
        for item in data:
            item_type = str(type(item))
            if item_type in type_counts:
                type_counts[item_type] += 1
            else:
                type_counts[item_type] = 1
        print(type_counts)
    else:
        print(sum(numeric_data) / len(numeric_data))  # Default to average
""",
    "attribute_error": """
# Handle attribute errors (trying to access attributes on objects)
numeric_data = []
for x in data:
    try:
        # Handle various data types
        if hasattr(x, '__float__'):
            numeric_data.append(float(x))
        elif isinstance(x, (int, float)):
            numeric_data.append(x)
        elif isinstance(x, str):
            numeric_data.append(float(x))
    except (ValueError, TypeError, AttributeError):
        pass  # Skip items that can't be processed

if not numeric_data:
    print('No valid numbers')
elif task == "calculate_average":
    print(sum(numeric_data) / len(numeric_data))
elif task == "find_max":
    print(max(numeric_data))
elif task == "count_by_type":
    type_counts = {}
    for item in data:
        item_type = str(type(item))
        if item_type in type_counts:
            type_counts[item_type] += 1
        else:
            type_counts[item_type] = 1
    print(type_counts)
else:
    print(sum(numeric_data) / len(numeric_data))  # Default to average
""",
    "value_error": """
# Handle value errors during conversion
numeric_data = []
for x in data:
    try:
        if isinstance(x, (int, float)):
            numeric_data.append(x)
        elif isinstance(x, str) and x.strip():  # Check if string is not empty
            # Try multiple conversion methods
            try:
                numeric_data.append(float(x))
            except ValueError:
                # Try handling special formats like '1,000.00' or '1.000,00'
                cleaned = x.replace(',', '.')
                if cleaned.count('.') > 1:  # If multiple dots after replacement
                    cleaned = cleaned.replace('.', '', cleaned.count('.')-1)
                try:
                    numeric_data.append(float(cleaned))
                except ValueError:
                    pass
    except Exception:
        pass  # Skip problematic values

if not numeric_data:
    print('No valid numbers')
elif task == "calculate_average":
    print(sum(numeric_data) / len(numeric_data))
elif task == "find_max":
    print(max(numeric_data))
elif task == "count_by_type":
    type_counts = {}
    for item in data:
        item_type = str(type(item))
        if item_type in type_counts:
            type_counts[item_type] += 1
        else:
            type_counts[item_type] = 1
    print(type_counts)
else:
    print(sum(numeric_data) / len(numeric_data))  # Default to average
""",
    "key_error": """
# Handle dictionary access errors
if isinstance(data, dict):
    numeric_values = []
    for key, value in data.items():
        try:
            if isinstance(value, (int, float)):
                numeric_values.append(value)
            else:
                numeric_values.append(float(value))
        except (ValueError, TypeError):
            pass
    
    if not numeric_values:
        print('No numeric values in dictionary')
    elif task == "calculate_average":
        print(sum(numeric_values) / len(numeric_values))
    elif task == "find_max":
        print(max(numeric_values))
    elif task == "count_by_type":
        type_counts = {}
        for key, value in data.items():
            item_type = str(type(value))
            if item_type in type_counts:
                type_counts[item_type] += 1
            else:
                type_counts[item_type] = 1
        print(type_counts)
    else:
        print(sum(numeric_values) / len(numeric_values))  # Default to average
else:
    # Process as a list
    numeric_data = []
    for x in data:
        try:
            if isinstance(x, (int, float)):
                numeric_data.append(x)
            else:
                numeric_data.append(float(x))
        except (ValueError, TypeError):
            pass

    if not numeric_data:
        print('No valid numbers')
    elif task == "calculate_average":
        print(sum(numeric_data) / len(numeric_data))
    elif task == "find_max":
        print(max(numeric_data))
    elif task == "count_by_type":
        type_counts = {}
        for item in data:
            item_type = str(type(item))
            if item_type in type_counts:
                type_counts[item_type] += 1
            else:
                type_counts[item_type] = 1
        print(type_counts)
    else:
        print(sum(numeric_data) / len(numeric_data))  # Default to average
""",
    "input_format_error": """
# Handle input format errors
try:
    # Try to safely process the data regardless of format
    numeric_data = []
    
    # Check if data is a string that needs evaluation
    if isinstance(data, str):
        try:
            # Attempt to evaluate as Python literal
            import ast
            evaluated_data = ast.literal_eval(data)
            if isinstance(evaluated_data, (list, tuple)):
                data = evaluated_data
        except:
            # If evaluation fails, treat as a single string
            try:
                numeric_data.append(float(data))
            except:
                pass
    
    # Process the data
    if isinstance(data, (list, tuple)):
        for item in data:
            try:
                if isinstance(item, (int, float)) and item is not None:
                    numeric_data.append(item)
                else:
                    numeric_data.append(float(item))
            except:
                pass
    
    if not numeric_data:
        print('No valid numbers')
    elif task == "calculate_average":
        print(sum(numeric_data) / len(numeric_data))
    elif task == "find_max":
        print(max(numeric_data))
    elif task == "count_by_type":
        type_counts = {}
        for item in data:
            item_type = str(type(item))
            if item_type in type_counts:
                type_counts[item_type] += 1
            else:
                type_counts[item_type] = 1
        print(type_counts)
    else:
        print(sum(numeric_data) / len(numeric_data))  # Default to average
except Exception as e:
    print(f'Error processing input: {str(e)}')
""",
    "fallback": """
# General fallback strategy for any other errors
try:
    # Convert string numbers to actual numbers and filter non-numeric values
    valid_data = []
    
    # Handle different input types
    if isinstance(data, (list, tuple)):
        # Process each item in the list/tuple
        for x in data:
            try:
                if isinstance(x, (int, float)) and x is not None:
                    valid_data.append(x)
                elif isinstance(x, str):
                    # Try direct conversion first
                    try:
                        valid_data.append(float(x))
                        continue
                    except ValueError:
                        pass
                    
                    # Try word-to-number conversion
                    word_num = word_to_number(x)
                    if word_num is not None:
                        valid_data.append(word_num)
                elif x is not None:
                    valid_data.append(float(x))
            except:
                pass
    elif isinstance(data, dict):
        # Extract values from dictionary
        for value in data.values():
            try:
                if isinstance(value, (int, float)) and value is not None:
                    valid_data.append(value)
                elif isinstance(value, str):
                    # Try direct conversion first
                    try:
                        valid_data.append(float(value))
                        continue
                    except ValueError:
                        pass
                    
                    # Try word-to-number conversion
                    word_num = word_to_number(value)
                    if word_num is not None:
                        valid_data.append(word_num)
                elif value is not None:
                    valid_data.append(float(value))
            except:
                pass
    elif data is not None:
        # Try to handle a single value
        try:
            if isinstance(data, str):
                # Try direct conversion first
                try:
                    valid_data.append(float(data))
                except ValueError:
                    # Try word-to-number conversion
                    word_num = word_to_number(data)
                    if word_num is not None:
                        valid_data.append(word_num)
            else:
                valid_data.append(float(data))
        except:
            pass
    
    if not valid_data:
        print('No valid numbers')
    elif task == "calculate_average":
        print(sum(valid_data) / len(valid_data))
    elif task == "find_max":
        print(max(valid_data))
    elif task == "count_by_type":
        type_counts = {}
        for item in data:
            item_type = str(type(item))
            if item_type in type_counts:
                type_counts[item_type] += 1
            else:
                type_counts[item_type] = 1
        print(type_counts)
    else:
        print(sum(valid_data) / len(valid_data))  # Default to average
except Exception as e:
    print(f'Calculation failed: {str(e)}')
"""
}



# Agent class
class CodeActAgent:
    def __init__(self, max_attempts=100, strategies=None):
        self.attempts = 0
        self.max_attempts = max_attempts
        self.previous_code = ""
        self.previous_error = ""
        self.history = []
        self.strategies = strategies or DEFAULT_STRATEGIES.copy()

    def reset(self):
        self.attempts = 0
        self.previous_code = ""
        self.previous_error = ""
        self.history = []

    def update_max_attempts(self, max_attempts):
        self.max_attempts = max_attempts
        
    def update_strategies(self, strategies):
        self.strategies = strategies

    def generate_code(self, data, feedback=None):
        """Generate Python code based on feedback."""
        self.attempts += 1
        if self.attempts > self.max_attempts:
            return None  # Stop if max attempts reached

        if not feedback:  # Initial attempt
            self.previous_code = self.strategies["fallback"]
            return self.previous_code
        
        # Analyze feedback for self-debugging
        if "TypeError" in feedback and "unsupported operand" in feedback:
            # Handle non-numeric values
            self.previous_code = self.strategies["type_error"]
        elif "ZeroDivisionError" in feedback:
            # Handle empty list or division by zero
            self.previous_code = self.strategies["zero_division"]
        elif "NameError" in feedback and "data" in feedback:
            # Handle undefined variable
            self.previous_code = self.strategies["name_error"]
        elif "IndexError" in feedback:
            # Handle index errors
            self.previous_code = self.strategies["index_error"]
        elif "AttributeError" in feedback:
            # Handle attribute access errors
            self.previous_code = self.strategies["attribute_error"]
        elif "ValueError" in feedback:
            # Handle value conversion errors
            self.previous_code = self.strategies["value_error"]
        elif "KeyError" in feedback:
            # Handle dictionary key errors
            self.previous_code = self.strategies["key_error"]
        elif "INPUT_FORMAT_ERROR" in feedback:
            # Handle input format errors
            self.previous_code = self.strategies["input_format_error"]
        else:
            # If feedback is unclear, try a safer approach
            self.previous_code = self.strategies["fallback"]
            
        return self.previous_code

    def execute_code(self, code, data_input, task="calculate_average"):
        """Execute the generated code and return output or error."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            # Convert the input string to a Python object
            try:
                data = eval(data_input)
            except:
                return "INPUT_FORMAT_ERROR: Invalid data format. Please provide a valid Python list."
            
            # Execute the code with the data, task, and word_to_number in scope
            global_vars = {
                "data": data, 
                "task": task, 
                "word_to_number": word_to_number,
                "w2n": w2n
            }
            exec(code, global_vars)
            output = mystdout.getvalue().strip()
            return output if output else "No output"
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            self.previous_error = error
            return error
        finally:
            sys.stdout = old_stdout

    def run_agent(self, data_input, task="calculate_average"):
        """Run the agent with the provided data input."""
        feedback = None
        history = []
        
        while self.attempts < self.max_attempts:
            code = self.generate_code(data_input, feedback)
            if code is None:
                history.append({
                    "attempt": self.attempts,
                    "step": "Max attempts reached",
                    "code": None,
                    "result": "Agent stopping",
                    "success": False
                })
                break
            
            result = self.execute_code(code, data_input, task)
            success = "Error" not in result and "No valid" not in result and "No numeric" not in result
            
            history.append({
                "attempt": self.attempts,
                "code": code,
                "result": result,
                "success": success,
                "task": task
            })
            
            if success:
                break
            
            feedback = result  # Use result as feedback for next iteration
        
        self.history = history
        return history

def display_code_editor(key, default_code, height=150):
    """Display a code editor with syntax highlighting"""
    return st.text_area(
        "Code:",
        value=default_code,
        height=height,
        key=key
    )

# Load sample data if available
def load_sample_data():
    try:
        with open('sample_data.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"samples": []}

# Main function
def main():
    st.title("🤖 CodeAct Agent UI")
    st.subheader("A self-debugging code agent")
    
    # Load samples
    sample_data = load_sample_data()
    samples = sample_data.get("samples", [])
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Agent Configuration")
        
        # Max attempts slider
        max_attempts = st.slider("Maximum Attempts", 1, 100, 50, help="Set the maximum number of attempts for the agent")
        
        # Sample data selection if available
        if samples:
            st.subheader("📊 Sample Data")
            sample_names = ["None"] + [sample["name"] for sample in samples]
            selected_sample = st.selectbox("Load sample data:", sample_names)
            
            if selected_sample != "None":
                sample = next((s for s in samples if s["name"] == selected_sample), None)
                if sample:
                    st.info(f"Sample: {sample['description']}")
        
        # Advanced options toggle
        show_advanced = st.checkbox("Show Advanced Options")
        
        if show_advanced:
            st.subheader("Error Handling Strategies")
            st.caption("Customize how the agent responds to different errors")
            
            # Create tabs for different error strategies
            tabs = st.tabs(["Type Error", "Zero Division", "Name Error", "Index Error", "Fallback"])
            
            strategies = DEFAULT_STRATEGIES.copy()
            
            with tabs[0]:
                strategies["type_error"] = display_code_editor("type_error", DEFAULT_STRATEGIES["type_error"])
            
            with tabs[1]:
                strategies["zero_division"] = display_code_editor("zero_division", DEFAULT_STRATEGIES["zero_division"])
                
            with tabs[2]:
                strategies["name_error"] = display_code_editor("name_error", DEFAULT_STRATEGIES["name_error"])
                
            with tabs[3]:
                strategies["index_error"] = display_code_editor("index_error", DEFAULT_STRATEGIES["index_error"])
                
            with tabs[4]:
                strategies["fallback"] = display_code_editor("fallback", DEFAULT_STRATEGIES["fallback"])
        else:
            strategies = DEFAULT_STRATEGIES.copy()
    
    # Initialize agent
    if 'agent' not in st.session_state:
        st.session_state.agent = CodeActAgent(max_attempts=max_attempts, strategies=strategies)
    else:
        # Update agent settings if they've changed
        st.session_state.agent.update_max_attempts(max_attempts)
        st.session_state.agent.update_strategies(strategies)
    
    # Reset button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Reset Agent"):
            st.session_state.agent.reset()
            st.session_state.pop('history', None)
            st.experimental_rerun()
    
    # Task selection
    task_options = list(TASK_TEMPLATES.keys())
    task_descriptions = [TASK_TEMPLATES[t]["description"] for t in task_options]
    task_display = [f"{desc}" for desc in task_descriptions]
    
    selected_index = st.radio(
        "Select a task for the agent:",
        range(len(task_options)),
        format_func=lambda x: task_display[x]
    )
    
    selected_task = task_options[selected_index]
    task_config = TASK_TEMPLATES[selected_task]
    
    # Input data
    with st.expander("Configure Input Data", expanded=True):
        # Set default data from sample if selected
        default_data = task_config["default_data"]
        if 'selected_sample' in locals() and selected_sample != "None":
            sample = next((s for s in samples if s["name"] == selected_sample), None)
            if sample:
                default_data = str(sample["data"])
        
        data_input = st.text_area("Enter your data as a Python list:", 
                                 value=default_data, 
                                 height=100,
                                 help=f"Example: {default_data}")
        
        st.info(f"The agent will attempt to {task_config['description'].lower()}.")
        
        # Show expected result if using a sample
        if 'selected_sample' in locals() and selected_sample != "None":
            sample = next((s for s in samples if s["name"] == selected_sample), None)
            if sample and "expected_result" in sample:
                st.success(f"Expected result: {sample['expected_result']}")
        
        run_col1, run_col2 = st.columns([4, 1])
        with run_col2:
            run_button = st.button("Run Agent", type="primary")
    
    # Run the agent when button is clicked
    if run_button:
        with st.spinner("Agent is processing..."):
            # Simulate some processing time for better UX
            time.sleep(0.5)
            history = st.session_state.agent.run_agent(data_input, selected_task)
            st.session_state.history = history
    
    # Display results
    if 'history' in st.session_state and st.session_state.history:
        st.markdown("### 📊 Agent Execution Results")
        
        # Progress bar
        total_attempts = len(st.session_state.history)
        final_attempt = st.session_state.history[-1]
        success = final_attempt.get("success", False)
        
        st.progress(total_attempts / st.session_state.agent.max_attempts)
        
        # Results in an expandable container
        with st.expander("Detailed Execution Steps", expanded=True):
            for i, attempt in enumerate(st.session_state.history):
                with st.container():
                    st.markdown(f"<div class='attempt-header'><h4>Attempt {attempt['attempt']}/{st.session_state.agent.max_attempts}</h4></div>", unsafe_allow_html=True)
                    
                    # Show code with copy button
                    st.markdown("**Generated Code:**")
                    if attempt['code']:
                        st.code(attempt['code'], language="python")
                    
                    # Show result
                    if attempt.get('success', False):
                        st.markdown(f"<p><strong>Result:</strong> <span class='success'>{attempt['result']}</span></p>", unsafe_allow_html=True)
                        st.success("Task completed successfully!")
                    else:
                        st.markdown(f"<p><strong>Result:</strong> <span class='error'>{attempt['result']}</span></p>", unsafe_allow_html=True)
                    
                    # Add separator except for last item
                    if i < len(st.session_state.history) - 1:
                        st.markdown("---")
        
        # Final summary
        st.markdown("### 📋 Summary")
        if success:
            st.success(f"Agent completed the task successfully in {total_attempts} attempts.")
        else:
            st.error(f"Agent failed to complete the task after {total_attempts} attempts.")
            
        # Metrics display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Attempts Used", total_attempts, f"{st.session_state.agent.max_attempts - total_attempts} remaining")
        with col2:
            st.metric("Success Rate", f"{100 if success else 0}%")
        with col3:
            st.metric("Task", task_config["description"])
            
# Run the app
if __name__ == "__main__":
    main() 