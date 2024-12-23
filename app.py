import pandas as pd
import chainlit as cl
import re
import sys
import io
import os
import plotly.graph_objects as go
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio
import uuid  # Import UUID for unique session IDs
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage



# Load environment variables
load_dotenv()

system_prompt = """You are a highly skilled data analyst and machine learning expert. You will reply to the user's messages and provide the user with Python code or very short notes. The user will ask you to perform data mining and machine learning tasks using the given dataset.
Here are the requirements for your responses:
1: The pandas DataFrame is already loaded in the variable "df".
2: Do not load the dataset again in the generated code!
3: The code should perform clear and complete data mining and machine learning tasks and, where appropriate, generate visualizations using Plotly. Return the figure objects for display.
4: Provide brief explanations along with the code on how the tasks or visualizations are important, what they achieve, and the insights they provide.
5: If the user asks for suggestions for tasks, just provide the possible analyses or model ideas without writing the code.
6: Write the code as a single script. Do not write notebook-specific code. Always write code for the given columns and with respect to their datatypes.
7: For data mining and machine learning tasks, use sklearn, XGBoost or other suitable libraries for modeling, and clearly explain the evaluation metrics.
"""

# Initialize the Google Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
)

def get_dt_columns_info(df):
    column_types = df.dtypes
    column_types_list = column_types.reset_index().values.tolist()
    infos = ""
    for column_name, column_type in column_types_list:
        infos += "{}({}),\n".format(column_name, column_type)
    return infos[:-1]



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




async def interpret_code_async(gpt_response, user_df):
    """Asynchronously execute the code extracted from GPT response."""
    if "```" in gpt_response:
        just_code = extract_code(gpt_response)

        if just_code.startswith("python"):
            just_code = just_code[len("python"):]

        just_code = filter_rows(just_code)
        

        local_scope = {'df': user_df, 'go': go, 'px': px}  # Include Plotly Graph Objects

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
        figures = [
            (value) for value in local_scope.values() if isinstance(value, go.Figure) 
        ]

        return modified_df, new_stdout.getvalue(), figures  # Return modified DataFrame, output, and list of figures

    return None, "", []  # Return None for DataFrame, empty output, and empty list for figures if no code block found


# Define a new graph
workflow = StateGraph(state_schema=MessagesState)

# Define the function that calls the model
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


@cl.on_chat_start
async def start_chat():
    session_id = str(uuid.uuid4())  # Generate a unique session ID
    cl.user_session.set("session_id", session_id)

    input_messages = [SystemMessage(content=system_prompt),HumanMessage(content="Hi")]
    config = {"configurable": {"thread_id": f"{cl.user_session.get('session_id')}"}}
    # Invoke the LangGraph workflow
    output=app.invoke({"messages": input_messages},config)
    gpt_response = output["messages"][-1].content
    
    await cl.Message(content=f"Bot:\n{gpt_response}").send()
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
    
   

    # Initialize a stack to store DataFrame states
    cl.user_session.set("df_stack", [df])  # Start with the initial DataFrame
    
    cl.user_session.set("user_df", df)
    
    await cl.Message(
        content=f"`{file.name}` uploaded correctly!\n it contains {cl.user_session.get('user_df').shape[0]} Rows and {cl.user_session.get('user_df').shape[1]} Columns where each column type are:\n [{get_dt_columns_info(cl.user_session.get('user_df'))}]"
    ).send()

    

@cl.on_message
async def main(message: str):
    # Display a loading message
    loading_message = await cl.Message(content="Generating your results, please wait...").send()
    
    # Retrieve the user's DataFrame from their session
    user_df = cl.user_session.get("user_df", None)

    if user_df is None:
        await cl.Message(content="Please upload your dataset first.").send()
        return

   
     # Prepare input messages for the LangGraph workflow
    input_messages = [HumanMessage(content=f"{message.content}\nThe available fields in the dataset df and their types are:{get_dt_columns_info(cl.user_session.get('user_df'))}")]
    config = {"configurable": {"thread_id": f"{cl.user_session.get('session_id')}"}}
    # Invoke the LangGraph workflow
    output = app.invoke({"messages": input_messages},config)
    gpt_response = output["messages"][-1].content  # Get the last message from the output
    

    # Simulate typing effect for the response
    await loading_message.remove()  # Remove loading message
    await cl.Message(content=f"Bot:\n{gpt_response}").send()
    

    # Execute the code asynchronously
    try:
        modified_df, output, figures = await interpret_code_async(gpt_response, user_df)
        if modified_df is not None and not modified_df.equals(user_df):
            cl.user_session.set("user_df", modified_df)
            cl.user_session.get("df_stack").append(modified_df)  # Push the new state onto the stack
            await cl.Message(content="DataFrame Updated").send()
        
    except Exception as e:
        await cl.Message(content=f"Code Execution Failed: {e}").send()
        return

    # Handle multiple figures
    if figures:
        await cl.Message(content="Visualizations:").send()
        for i, figure in enumerate(figures):
            elements = [cl.Plotly(name=f"Plot {i+1}", figure=figure, display="page")]
            await cl.Message(content=f"Here is the generated Plot {i+1}:", elements=elements).send()

    if output:
        await cl.Message(content="Execution Output:").send()
        await cl.Message(content=output).send()

    # Define an action button
    actions = [
        cl.Action(
            name="Yes",
            value="revert_button",
            description="Revert DataFrame",
        )
    ]
    await cl.Message(content="Revert DataFrame?", actions=actions).send()

@cl.action_callback("Yes")
async def on_action(action):
    df_stack = cl.user_session.get("df_stack", [])
    if len(df_stack) > 1:
        last_state = df_stack[-2]  # Get the second last state
        cl.user_session.set("user_df", last_state)
        cl.user_session.set("df_stack", df_stack[:-1])  # Remove the last state from the stack
        await cl.Message(content="Reverted to the last state of the DataFrame.").send()
    else:
        await cl.Message(content="No previous state to revert to.").send()

    await cl.Message(
        content=f"DataFrame contains {cl.user_session.get('user_df').shape[0]} Rows and {cl.user_session.get('user_df').shape[1]} Columns where each column type are:\n [{get_dt_columns_info(cl.user_session.get('user_df'))}]"
    ).send()
