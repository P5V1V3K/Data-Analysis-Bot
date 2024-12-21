import pandas as pd
import chainlit as cl
import re
import sys
import io
import os
import plotly.graph_objects as go  # Import Plotly Graph Objects
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Initialize the Google Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=1,
)

def get_dt_columns_info(df):
    column_types = df.dtypes
    column_types_list = column_types.reset_index().values.tolist()
    infos = ""
    for column_name, column_type in column_types_list:
        infos += "{}({}),\n".format(column_name, column_type)
    return infos[:-1]

@cl.on_chat_start
async def start_chat():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload your CSV/XLSX dataset file to begin!", 
            accept=["csv", "xlsx"], 
            max_size_mb=100
        ).send()
    
    file = files[0]
    
    if "csv" in file.path:
        df = pd.read_csv(file.path)
    else:
        df = pd.read_excel(file.path, index_col=0)    
    
    cl.user_session.set("user_df", df)
    
    await cl.Message(
        content=f"`{file.name}` uploaded correctly!\n it contains {df.shape[0]} Rows and {df.shape[1]} Columns where each column type are:\n [{get_dt_columns_info(df)}]"
    ).send()

    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": system_prompt.format(get_dt_columns_info(df))}],
    )

def extract_code(gpt_response):
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, gpt_response, re.DOTALL)
    if matches:
        return matches[-1]
    else:
        return None

def filter_rows(text):
    lines = text.split('\n')
    filtered_lines = [line for line in lines if "pd.read_csv" not in line and "pd.read_excel" not in line and ".show()" not in line]
    filtered_text = '\n'.join(filtered_lines)
    return filtered_text

def interpret_code(gpt_response, user_df):
    if "```" in gpt_response:
        just_code = extract_code(gpt_response)
        
        if just_code.startswith("python"):
            just_code = just_code[len("python"):]
        
        just_code = filter_rows(just_code)
        print("CODE part:{}".format(just_code))
        
        local_scope = {'df': user_df, 'go': go}  # Include Plotly Graph Objects
        
        # Redirect standard output to capture print statements
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        
        try:
            exec(just_code, {}, local_scope)
        except Exception as e:
            sys.stdout = old_stdout
            return None, e, None  # Return None for DataFrame, error message, and None for figure
        
        # Restore original standard output
        sys.stdout = old_stdout
        
        # Check if 'df' is in the local scope and return the modified DataFrame
        if 'df' in local_scope:
            modified_df = local_scope['df']
        else:
            modified_df = None
        
        # Collect all figures created in the local scope
        figures = [value for key, value in local_scope.items() if isinstance(value, go.Figure)]
        
        return modified_df, new_stdout.getvalue(), figures  # Return modified DataFrame, output, and list of figures
    
    return None, "", []  # Return None for DataFrame, empty output, and empty list for figures if no code block found

@cl.on_message
async def main(message: str):

     # Display a loading message
    loading_message = await cl.Message(content="Generating your results, please wait...").send()
    # Retrieve the user's DataFrame from their session
    user_df = cl.user_session.get("user_df", None)
    
    if user_df is None:
        await cl.Message(content="Please upload your dataset first.").send()
        return

    # Add the user's message to the history
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content}) 
    
    # Response of the LLM model using the invoke method
    response = llm.invoke(message_history)
    gpt_response = response.content
    print("Gemini response:{}".format(gpt_response))

    # Simulate typing effect for the response
    await loading_message.remove()  # Remove loading message
    msg = cl.Message(content="")
    for char in gpt_response:
        await msg.stream_token(char)

    await msg.send()


    # Execute the code and get the modified DataFrame, output, and figures
    try:
        modified_df, output, figures = interpret_code(gpt_response, user_df)
        await cl.Message(content=output).send()
    except Exception as e:
        await cl.Message(content="lengthy Response. Code Execution Failed").send()
        return

    
    # Commenting out the DataFrame update functionality
    # if modified_df is not None:
    #     cl.user_session.set("user_df", modified_df)
        
    #     new_system_message = system_prompt.format(get_dt_columns_info(modified_df))
    #     cl.user_session.set("message_history", [{"role": "system", "content": new_system_message}])
        
    

    # Handle multiple figures
    if figures:
        for i, figure in enumerate(figures):
            elements = [cl.Plotly(name=f"chart_{i}", figure=figure, display="inline", size="large")]
            await cl.Message(content=f"Here is the generated plot {i + 1}:", elements=elements).send()