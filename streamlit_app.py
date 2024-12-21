import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re
import sys
import io
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import uuid  # To generate unique session IDs

# Load environment variables
load_dotenv()

st.set_page_config(layout="wide")

# Initialize the Google Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=1,
)

# Define system prompt
system_prompt = """You are a great data analyst. You will reply to the user's messages and provide the user with only code in python or with very short notes.
The user will ask you to provide the code to answer any question about the dataset.
Besides, here are some requirements:
1: The pandas dataframe is already loaded in the variable "df".
2: Do not load the dataframe in the generated code!
3: The code has to generate visualizations using Plotly and return the figure objects for display.
4: Provide explanations along with the code on how important the visualization is and what insights can be gained from it.
5: If the user asks for suggestions of analysis, just provide the possible analyses without the code.
6: For any visualizations, write only one block of code. Do not write code for python notebook, write as a python script.
7: The available fields in the dataset "df" and their types are: {}"""

# Helper functions
def get_dt_columns_info(df):
    """Get column names and their types."""
    column_types = df.dtypes
    return ", ".join(f"{col}({dtype})" for col, dtype in column_types.items())

def extract_code(gpt_response):
    """Extract code block from the GPT response."""
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, gpt_response, re.DOTALL)
    if matches:
        return matches[-1]
    return None

def filter_rows(text):
    """Filter out unwanted lines from the code."""
    lines = text.split('\n')
    filtered_lines = [line for line in lines if "pd.read_csv" not in line and "pd.read_excel" not in line and ".show()" not in line]
    return '\n'.join(filtered_lines)

def interpret_code(gpt_response, user_df):
    """Interpret and execute the code from GPT response."""
    if "```" in gpt_response:
        just_code = extract_code(gpt_response)
        
        if just_code.startswith("python"):
            just_code = just_code[len("python"):]

        just_code = filter_rows(just_code)
        print("CODE part:{}".format(just_code))
        
        local_scope = {'df': user_df, 'go': go}
        
        # Redirect standard output to capture print statements
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        try:
            exec(just_code, {}, local_scope)
        except Exception as e:
            sys.stdout = old_stdout
            return None, e, None
        
        # Restore original standard output
        sys.stdout = old_stdout
        
        # Check if 'df' is in the local scope and return the modified DataFrame
        modified_df = local_scope.get('df')
        figures = [value for value in local_scope.values() if isinstance(value, go.Figure)]
        return modified_df, new_stdout.getvalue(), figures

    return None, "", []

# Assign a unique session ID for each user
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# App title
st.title("Data Analysis Assistant")
st.markdown("Upload a dataset and ask questions about it. The app will generate visualizations and insights.")

# File upload
uploaded_file = st.file_uploader("Upload your CSV/XLSX dataset file", type=["csv", "xlsx"])

if uploaded_file:
    # Load the dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Store the dataset in the session state
    st.session_state[f"{st.session_state.session_id}_df"] = df

    st.success(f"File `{uploaded_file.name}` uploaded successfully!")
    st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    st.write(f"Columns: {get_dt_columns_info(df)}")

    # Initialize session history
    if f"{st.session_state.session_id}_history" not in st.session_state:
        st.session_state[f"{st.session_state.session_id}_history"] = []

    # Store the system context
    system_context = system_prompt.format(get_dt_columns_info(df))
    session_history = st.session_state[f"{st.session_state.session_id}_history"]
    if not session_history:
        session_history.append({"role": "system", "content": system_context})

    # User query input
    user_query = st.text_input("Ask a question about the dataset:", "")
    if user_query:
        # Add user query to the session history
        session_history.append({"role": "user", "content": user_query})
        
        # LLM response
        with st.spinner("Generating response..."):
            response = llm.invoke(session_history)
            gpt_response = response.content
            st.markdown("### LLM Response")
            st.write(gpt_response)

            # Interpret and execute code
            try:
                modified_df, output, figures = interpret_code(gpt_response, df)
                if output:
                    st.markdown("### Execution Output")
                    st.code(output)

                if figures:
                    st.markdown("### Visualizations")
                    for fig in figures:
                        st.plotly_chart(fig)
            except Exception as e:
                st.error("Error while executing code: " + str(e))

    # Save session history
    st.session_state[f"{st.session_state.session_id}_history"] = session_history
