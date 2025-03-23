#!/bin/bash

# Activate conda environment if it exists
if [ -d "$HOME/anaconda3/bin" ]; then
    source "$HOME/anaconda3/bin/activate" codeact
elif [ -d "$HOME/miniconda3/bin" ]; then
    source "$HOME/miniconda3/bin/activate" codeact
fi

# Install requirements if needed
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py 