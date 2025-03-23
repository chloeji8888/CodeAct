# CodeAct Agent UI

A Streamlit-based user interface for a self-debugging code agent that demonstrates the CodeAct paradigm.

## What is CodeAct?

CodeAct is a paradigm where AI agents can generate, execute, and self-debug code based on feedback. This application demonstrates how an agent can:

1. Generate code to solve a problem (calculating an average from a list)
2. Execute the code and observe the results
3. Debug and improve the code based on error messages
4. Iterate until the problem is solved successfully

## Setup

1. Make sure you have Python 3.8+ installed
2. Create and activate a conda environment (optional but recommended):
   ```bash
   conda create -n codeact python=3.11
   conda activate codeact
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

This will open the web interface in your default browser.

## How to Use

1. Enter your data as a Python list in the input box (default: `[1, 'two', 3, None, 5]`)
2. Click "Run Agent" to start the agent
3. Watch as the agent attempts to calculate the average of numeric values in your list
4. See the agent's self-debugging process across multiple attempts
5. View the final result and summary

## Features

- Interactive UI with code highlighting
- Step-by-step visualization of the agent's reasoning
- Ability to input custom data for testing
- Reset functionality to start fresh

## Example

The default example has mixed data types: `[1, 'two', 3, None, 5]`

The agent will:

1. First try a simple average calculation (which fails due to type errors)
2. Adapt by filtering for numeric values only
3. Provide the correct answer (3.0)
