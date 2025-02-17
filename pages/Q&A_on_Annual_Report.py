
# Standard Library Imports
import os
import time
import shutil
import getpass
import json
import re
import random
import shelve
import traceback
import ast
import io
from pathlib import Path
from collections import defaultdict
from typing import Annotated, List, Tuple, Dict, Union, Set, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from datetime import datetime
import datetime as dt
import operator

# Third-Party Libraries
import dotenv
import numpy as np
import pandas as pd
import yfinance as yf
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
from PIL import Image

# Streamlit
import streamlit as st
from st_clickable_images import clickable_images

# LangChain & AI
from langchain import hub
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Technical Analysis (TA)
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volume import volume_weighted_average_price

# Initialize the callback handler
callback_handler = OpenAICallbackHandler()
# Load the .env file
dotenv.load_dotenv()


if "get_feedback" not in st.session_state:
  st.session_state.get_feedback = False

def remove_shelve_file(file_name):
  """Remove the shelve database and associated files if they exist."""
  if os.path.exists(file_name):
    os.remove(file_name)




def append_shelve_data(key, new_entry):
    # session_id = str(st.session_state.session_id)  # Convert session_id to string
    file_name = f'document_file_{session_id}'  # Append session_id to filename
    # file_name = 'dummy_file'
    with shelve.open(file_name, writeback=True) as db:
        if key not in db:
            db[key] = []  # Initialize if key doesn't exist
        if key in ['file_path','output_path']:
          db[key] = new_entry
        else:
          db[key].extend(new_entry)  # Extend new data
        db.sync()  # Ensure data is saved

def save_feedback():
  # Save feedback to CSV when provided
  # Create feedback dictionary
  if st.session_state.feedback==0:
    feedback_received = 'Negative'
  elif st.session_state.feedback==1:
    feedback_received = 'Positive'
  else:
    return ''
  my_feedback_dict = {
      "input_query": st.session_state.prompt,
      "response": st.session_state.response,
      "chat_history":st.session_state.messages,
      "feedback": feedback_received,
      "timestamp": st.session_state.timestamp,
      "session_id": st.session_state["session_id"]
  }

  # Check if the file exists
  feedback_file = "feedback.csv"
  if os.path.exists(feedback_file):
      # Append to existing CSV
      existing_df = pd.read_csv(feedback_file)
      updated_df = pd.concat([existing_df, pd.DataFrame([my_feedback_dict])], ignore_index=True)
  else:
      # Create a new CSV
      updated_df = pd.DataFrame([my_feedback_dict])

  # Save the updated DataFrame to CSV
  updated_df.to_csv(feedback_file, index=False)
  # resetting
  st.toast("Your feedback has been recorded. Thank you!")
  st.session_state.get_feedback = False



hide_streamlit_style = """
                <style>
                div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stDecoration"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
                #MainMenu {
                visibility: hidden;
                height: 0%;
                }
                header {
                visibility: hidden;
                height: 0%;
                }
                footer {
                visibility: hidden;
                height: 0%;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


custom_button_style = """
    <style>
        .stButton > button {
            background-color: #00C853; /* Bright Neon Green */
            color: #121212; /* Dark background for contrast */
            border: 2px solid #00C853;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease; /* Smooth transitions */
        }
        .stButton > button:hover {
            background-color: #1B5E20; /* Deep Forest Green */
            color: white;
            border-color: #00C853;
        }
        .stButton > button:active {
            background-color: #1B5E20; /* Deep Forest Green */
            color: #00C853; /* Neon Green */
            border-color: #1B5E20;
            transform: scale(0.98); /* Slight shrink effect */
        }
        .stButton > button:focus {
            outline: none;
            box-shadow: 0 0 4px 2px #00C853; /* Green Glow effect */
        }
    </style>
"""


st.markdown(custom_button_style, unsafe_allow_html=True)
if np.logical_or(st.session_state["company_name"] == 'Default Name',st.session_state["ticker_name"] == None):
  st.header("Please go to the 'Home' page and select a company name & a ticker name to generate the Financial report")
  if st.button("Go to Home"):
    st.switch_page("main.py")
  st.stop()
else:
  st.session_state['status'] = 3


def save_images_with_boxes(pdf_page, segments, output_path):
    """
    Plots a PDF page as an image with bounding boxes overlayed for the given segments.

    Args:
        pdf_page: A PyMuPDF page object.
        segments: List of segment dictionaries containing coordinates and category info.
        output_path: Pathlib Path object where the image will be saved.
    """
    try:
      # Convert the PDF page to a high-resolution image
      pix = pdf_page.get_pixmap(dpi=300)  # Set DPI for higher resolution
      pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

      # Calculate appropriate figure size based on image resolution
      figsize = (pix.width / 100, pix.height / 100)  # Scale to maintain original resolution

      # Set up the plot
      fig, ax = plt.subplots(figsize=figsize, dpi=100)
      ax.imshow(pil_image)

      # Predefine categories and their colors
      category_to_color = {
          "Title": "orchid",
          "Image": "forestgreen",
          "Table": "tomato",
      }
      default_color = "yellow"
      categories = set()

      for segment in segments:
          coordinates = segment.get("coordinates", {})
          points = coordinates.get("points", [])
          layout_width = coordinates.get("layout_width", pix.width)
          layout_height = coordinates.get("layout_height", pix.height)

          # Scale points to match the PDF page dimensions
          scaled_points = [
              (x * pix.width / layout_width, y * pix.height / layout_height)
              for x, y in points
          ]

          # Determine box color and add to categories
          # box_color = category_to_color.get(segment.get("category", ""), default_color)
          categories.add(segment.get("category", "Text"))
          # Add the polygon patch for bounding box
          if scaled_points:
              rect = patches.Polygon(
                  scaled_points, linewidth=0, edgecolor=None, facecolor='yellow', alpha=0.3
              )
              ax.add_patch(rect)

      # Create a legend for the categories
      legend_handles = [
          patches.Patch(color=category_to_color.get(cat, default_color), label=cat)
          for cat in sorted(categories)
      ]
      ax.axis("off")
      ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
      plt.tight_layout()

      # Save the image
      random_number = random.randint(1, 1000)
      page_number =  1  # 1-based page number
      image_format = "jpeg"
      image_path = output_path / f"{random_number}_highlighted_page_{page_number}.{image_format}"
      fig.savefig(image_path)
      # fig.savefig(image_path, bbox_inches="tight", dpi=300)
      plt.close(fig)
    except Exception as e:
      st.sidebar.write(f"Error while saving images with boxes: {e}")



def save_page(file_path: str, doc_list: list, page_numbers: set, output_path: str , print_text=True) -> None:
  """
  Renders a specific page from a PDF file with associated metadata.

  Args:
      file_path (str): Path to the PDF file.
      doc_list (list): List of document metadata for rendering.
      page_number (int): Page number to render (1-indexed).
      print_text (bool): Whether to print text from the page.
  """
  for page_number in page_numbers:
    pdf_page = fitz.open(file_path).load_page(page_number - 1)  # Page numbers are 0-indexed in fitz
    page_docs = [doc for doc in doc_list if doc.metadata.get("page_number") == page_number]
    segments = [doc.metadata for doc in page_docs]
    save_images_with_boxes(pdf_page, segments,output_path)

    if print_text:
        for segment in segments:
            print(segment.get("text", "No text available"))


def extract_page_numbers_from_chunk(chunk):
    """
    Extracts unique page numbers from a chunk's metadata.

    Args:
        chunk: A chunk object with metadata containing orig_elements.

    Returns:
        set: A set of unique page numbers.
    """
    # Validate input
    if not hasattr(chunk, "metadata") or not hasattr(chunk.metadata, "orig_elements"):
        raise ValueError("Invalid chunk format: 'metadata.orig_elements' not found.")

    # Extract unique page numbers
    elements = chunk.metadata.orig_elements
    page_numbers = {
        int(element.metadata.page_number)
        for element in elements
        if hasattr(element, "metadata") and hasattr(element.metadata, "page_number")
    }

    return page_numbers


def save_chunk_pages(chunk, file_path, output_path):
    """
    Displays pages and visualizes elements in a chunk.

    Args:
        chunk: A chunk object with metadata containing orig_elements.
        file_path (str): Path to the PDF file.
    """
    try:
        # Extract page numbers
        page_numbers = extract_page_numbers_from_chunk(chunk)
        print(f"Pages: {', '.join(map(str, sorted(page_numbers)))}")

        # Prepare document objects
        docs = []
        for element in chunk.metadata.orig_elements:
            if not hasattr(element, "metadata"):
                continue

            # Extract metadata and determine category
            metadata = element.metadata.to_dict()
            if "Table" in str(type(element)):
                metadata["category"] = "Table"
            elif "Image" in str(type(element)):
                metadata["category"] = "Image"
            else:
                metadata["category"] = "Text"

            metadata["page_number"] = int(element.metadata.page_number)

            # Append to document list
            docs.append(Document(page_content=element.text, metadata=metadata))

        # # Render pages
        # for page_number in sorted(page_numbers):
        #     save_page(file_path, docs, page_number, output_path , print_text=False)
        # write file as text
        output_file_path = 'dummy_text_file.txt'
        with open(output_file_path, 'w') as output_file:
            output_file.write(f"Pages: {', '.join(map(str, sorted(page_numbers)))}")

        # Save to a shelve database
        append_shelve_data('documents', docs)
        append_shelve_data('output_path', output_path)
        append_shelve_data('file_path', file_path)
        

    
    except Exception as e:
        # write file in text
        output_file_path = 'error.txt'
        with open(output_file_path, 'w') as output_file:
            output_file.write(f"Error while displaying chunk pages: {e}")
        st.toast(f"Error while displaying chunk pages")


# Function to load images from a folder
def load_images_from_folder(folder_path):
    images = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')):
            images.append(os.path.join(folder_path, file_name))
    return images


def highlight_text_and_save_images(retrieved_documents, input_pdf, output_folder,chunks):
    """
    Highlight text from documents on a PDF and save the highlighted pages as images.

    Parameters:
        documents (list[Document]): List of Document objects with metadata and content.
        input_pdf (str): Path to the input PDF.
        output_folder (str): Folder to save the output images.
        image_format (str): Format of the output images (e.g., 'jpeg', 'png').
        dpi (int): Resolution of the output images (default: 150).

    Returns:
        None
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find and print the matching chunks from `chunks`
    for retrieved_doc in retrieved_documents:
        retrieved_element = retrieved_doc.metadata['element_id']
        retrieved_filename = retrieved_doc.metadata['source']
        # chunks = st.session_state['chunks']

        for i, chunk in enumerate(chunks, start=0):
          chunk_dict = chunk.to_dict()
          chunk_element = chunk_dict.get('element_id', '').strip()
          chunk_filename = chunk_dict.get('metadata', {}).get('filename', 'Unknown Source')
          if np.logical_and(retrieved_element==chunk_element, retrieved_filename==chunk_filename):
            save_chunk_pages(chunks[i],input_pdf,output_path)

    print(f"All highlighted pages saved to '{output_folder}'.")


with st.sidebar:
  st.markdown(custom_button_style, unsafe_allow_html=True)
  # Add title in the sidebar
  st.header('How would you like to analyse?')

  # Add buttons to the sidebar
  financial_summary_button = st.button('Stock Analysis', key='financial_summary_button', help="Navigate to Stock Analysis",use_container_width=True, )
  qa_button = st.button('Ask Questions', key='qa_button', help="Navigate to the Q&A agent", use_container_width=True, type="primary")

  # Display the corresponding page based on the button clicked
  if qa_button:
      # Placeholder for Q&A page content
      st.switch_page("pages/Q&A_on_Annual_Report.py")

  elif financial_summary_button:
      # Placeholder for Financial Summary page content
      st.switch_page("pages/Stock_Analysis.py")




# Access the keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")


#Enter your OpenAI API Key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_TRACING_V2"] = str(LANGCHAIN_TRACING_V2)
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

if TAVILY_API_KEY==False:
    st.info("Please add your Tavily API key to continue.")
if LANGCHAIN_API_KEY==False:
    st.info("Please add your Langchain API key to continue.")
if OPENAI_API_KEY==False:
    st.info("Please add your OpenAI API key to continue.")

# To use wide mode in Streamlit, you can set the layout to "wide" with the following command:
# st.set_page_config(layout="wide")

# Clean text
def clean_and_format_text(input_text):
    """
    Cleans the input text by removing text between angle brackets and markdown formatting,
    including handling double ** used for bold text.

    Args:
        input_text (str): The input text containing markdown and special characters.

    Returns:
        str: The cleaned plain text.
    """
    # Step 1: Remove text between < and >
    text = re.sub(r"<.*?>", "", input_text)

    # Step 2: Remove markdown formatting, including double **
    text = re.sub(r"[*_`]+", "", text)

    return text.strip()

    # # Step 3: Fix character splitting (e.g., "b i l l i o n" -> "billion")
    # text = re.sub(r"(?<!\S)(\w)(\s)(\w)", lambda m: m.group(1) + m.group(3), text)

    return text.strip()
################################# ANNUAL REPORT RETRIVER MULTI AGENT #################################
# Initialize the LLM
format_llm = OpenAI(temperature=0.7)

# Implement the Generate Chain
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert in formatting data in markdown format.\n
    You would need to correct the markdown formatting of the *input_response*. \n
    First remove any markdown formatting. Second remove unnecessary character splitting into new lines.\n
    Third the text should be in mark down format.<|eot_id|><|start_header_id|>user<|end_header_id|>
    input_response: {input_response}
    output_response: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["input_response"],
)

# Chain
response_format_chain = LLMChain(llm=format_llm, prompt=prompt, output_parser=StrOutputParser())


#Implement Web Search tool
# Implement Web Search tool
def web_search(question: str) -> str:
    """
    Perform a web search based on the question.

    Args:
        question (str): The question to perform the web search for.

    Returns:
        web_results (str): The results of the web search as a concatenated string.
    """
    try:
        # Web search invocation
        docs = web_search_tool.invoke({"query": question})

        # Extract URLs from the search results
        web_urls = "\n".join([d.get("url", "") for d in docs])
        output_file_path = output_folder_path + 'web_url.txt'
        
        # Save the URLs to a file
        with open(output_file_path, 'w') as output_file:
            output_file.write(web_urls)

        # Extract content from the search results
        web_results_content = "\n".join([d.get("content", "") for d in docs])

        # Print and return the results
        print("---WEB SEARCH RESULTS---")
        print(web_results_content)

        return web_results_content
    except Exception as e:
        print(f"An error occurred during web search: {e}")
        return ""




# Implement retriever tool for SEC Report
def annual_reports_search(statement: str) -> dict[str, list[Document]]:
    """Retrieve documents based on a given query.

    This function uses a retriever to fetch relevant documents from annual reports for a given query.


    Args:
        statement (str): The question to search in annual reports

    Returns:
        dict[str, list[Document]]: A dictionary with a 'documents' key containing the list of retrieved documents.
    """
    # with st.spinner("Searching the annual report for answer"):
    try:
      #Instantiate the retriever
      # chroma_path = st.session_state['vectorstore']
      # st.sidebar.write(f"Vectorstore path: {chroma_path}")
      vectorstore = Chroma(persist_directory = chroma_path, embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))
      retriever = vectorstore.as_retriever(search_kwargs={"k":3})
      # st.session_state.retriever = retriever()
      # Retrieval
      documents = retriever.invoke(statement)
      try:
        # Highlighting text in folder
        highlight_text_and_save_images(documents, input_pdf_path, output_folder_path,chunks)
        # # Define dummy text to save
        # dummy_text = "Success"
        # # Specify the output file path (unique for each element)
        # output_file_path = "dummy_text_file_.txt"
        # # Save the dummy text to the file
        # with open(output_file_path, "w") as file:
        #     file.write(dummy_text)
      except Exception as a:
        # Handle the error gracefully
        # Define dummy text to save
        dummy_text = "An error occurred while saving highlighted images:"+ str(a)
        # Specify the output file path (unique for each element)
        output_file_path = "dummy_text_file_.txt"
        # Save the dummy text to the file
        with open(output_file_path, "w") as file:
            file.write(dummy_text)
        st.sidebar.write(f"An error occurred while saving highlighted images:", {str(a)})

    except Exception as e:
      # Handle the error gracefully
      st.sidebar.write(f"An error occurred while retrieving documents:", {str(e)})
      return "An error occurred while retrieving documents:" + str(e)

    print("---RETRIEVED---")
    print(documents)
    return documents


# creating web tool
web_search_tool = TavilySearchResults(k=3)

# creating tools for llms to use
tools = [web_search, annual_reports_search]
# tools = [ annual_reports_search]





######################################### Define our Execution Agent ######################
# Modify the prompt (example: changing a section of the prompt text)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", 'You are a helpful assistant whose job is to first conduct a search in annual report and then conduct a web search.'),
    ("human",'{messages}' ),
])
# chat_prompt.pretty_print()
# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-4-turbo-preview")
agent_executor = create_react_agent(llm, tools, state_modifier=chat_prompt)

# Define the State
# Let's now start by defining the state the track for this agent.
# First, we will need to track the current plan. Let's represent that as a list of strings.
# Next, we should track previously executed steps. Let's represent that as a list of tuples (these tuples will contain the step and then the result)
# Finally, we need to have some state to represent the final response as well as the original input.

class PlanExecute(TypedDict):
    company_name: str
    input: str
    previous_conversations: list[dict]
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    router: str
    router_logic: str

######################################### Define our Router Agent ######################
#Implement the Router
prompt = PromptTemplate(
template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a professional at routing inquiries.
A user will come to you with an inquiry about the financial performance of {company_name}. Your first job is to classify what type of inquiry it is. The types of inquiries you should classify it as are:

## `more-info`
Classify a user inquiry as this if you need more information before you will be able to help them. Examples include:
- The user is asking to compare profits for two years, but dosent mention the year itself
- The user says something isn't working but doesn't explain why/how it's not working

## `research`
Classify a user inquiry as this if it can be answered by looking up information related to the stock on Annual report or a simple web search. The user is likely inquiring
about financial performance (revenue, profit, growth), market position, risk factors, strategic initiatives, corporate governance, sustainability efforts, future outlook, and investor sentiment.
This will provide insights into the {company_name} overall health, competitive standing, and future prospects.

## `general`
Classify a user inquiry as this if it is just a general question.

Provide a tertiary choice: 'more-info', 'research', or 'general' based on the question along witht he logic for routing. Return a JSON with a two keys 'datasource' and 'router_logic'
with no premable or explaination. Mention the routing choice under 'datasource' key and the logic/reason for selecting the route under the 'router_logic' key.
Previous Conversations : {previous_conversations}
Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question","company_name","previous_conversations"],
)


# start = time.time()
question_router = prompt | llm | JsonOutputParser()


#Define Conditional Edges
def route_question(state: PlanExecute) -> Dict[str, str]:
    """Determine the next step based on the query classification.

    Args:
        state (PlanExecute): The current state of the agent, including the router's classification.

    Returns:
        Dict[str, str]: The next step to take, containing the 'datasource' and 'router_logic' keys.
    """
    with st.spinner("Routing Question"):
      print("---ROUTE QUESTION & LOGIC---")
      question = state["input"]
      company_name = state["company_name"]
      history = state["previous_conversations"]
      history_convo = ''
      for convo_dict in history:
          # Ensure proper key-value handling
          role = convo_dict.get("role", "unknown")  # Default to "unknown" if key is missing
          content = convo_dict.get("content", "")
          temp_convo = f"{role}: {content}"  # Format the role and content
          history_convo += '\n' + temp_convo
      source = question_router.invoke({"question": question,"company_name": company_name,"previous_conversations":history_convo})
      return {"router": source['datasource'], "router_logic": source['router_logic']}


def route_direction(state: PlanExecute) -> Literal["research", "more-info", "general"]:
    """Determine the next step based on the query classification.

    Args:
        state (AgentState): The current state of the agent, including the router's classification.

    Returns:
        Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]: The next step to take.

    Raises:
        ValueError: If an unknown router type is encountered.
    """
    with st.spinner("Routing Question"):
      print("---ROUTING---")
      routing_response = state["router"]
      print(routing_response)
      if routing_response == 'general':
          print("---ROUTE QUESTION TO respond_to_general_query---")
          return "general"
      elif routing_response == 'more-info':
          print("---ROUTE QUESTION TO ask_for_more_info---")
          return "more-info"
      elif routing_response == 'research':
          print("---ROUTE QUESTION TO create_research_plan---")
          return "research"
      else:
          raise ValueError(f"Unknown router type: {routing_response}")



######################################### Define our More info Agent ######################

prompt = PromptTemplate(
template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a Financial analyst. Your job is help answer questions using {company_name} Annual Report.
Your boss has determined that more information is needed before doing any research on behalf of the user.
Respond to the user and try to get any more relevant information. Do not overwhelm them! Be nice, and only ask them a single follow up question. <|eot_id|><|start_header_id|>user<|end_header_id|>
User_Question: {question}
Logic_More_Info: {context}
More_Info_Question: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
input_variables=["question", "context", "company_name"],
)
# Chain
start = time.time()
more_info_chain = prompt | llm | StrOutputParser()


# print('\n ################################################################### \n')

# async def ask_for_more_info(state: PlanExecute) -> dict[str]:
def ask_for_more_info(state: PlanExecute) -> dict[str]:
    """Generate a response asking the user for more information.

    This node is called when the router determines that more information is needed from the user.

    Args:
        state (PlanExecute): The current state of the agent, including conversation history and router logic.

    Returns:
        dict[str, list[str]]: A dictionary with a 'response' key containing the generated response.
    """
    with st.spinner("Generating response"):
      question = state["input"]
      doc_txt = state["router_logic"]
      company_name = state["company_name"]
      print('-- ASk FOR MORE INFO --')
      # generation = await more_info_chain.ainvoke({"question": question, "context": doc_txt})
      generation =  more_info_chain.invoke({"question": question, "context": doc_txt, "company_name":company_name})

      return {"response": generation}


######################################### Define our General Response Agent ######################

prompt = PromptTemplate(
template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a Financial analyst. Your job is help people using Company Annual Report any questions they are running into.
Your boss has determined that the user is asking a general question.
Respond to the user. Politely decline to answer and tell them you can only answer questions about topics covered {company_name} Annual Report, and that if their question is can be anaswered using Annual Report they should clarify how it is.\
Be nice to them though - they are still a user! <|eot_id|><|start_header_id|>user<|end_header_id|>
User_Question: {question}
Logic_More_Info: {context}
More_Info_Question: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
input_variables=["question", "context"],
)
# Chain
start = time.time()
general_chain = prompt | llm | StrOutputParser()

# async def respond_to_general_query(state: PlanExecute) -> dict[str]:
def respond_to_general_query(state: PlanExecute) -> dict[str]:
    """Generate a response to a general query not related to LangChain.

    This node is called when the router classifies the query as a general question.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    with st.spinner("Generating response"):
      question = state["input"]
      doc_txt = state["router_logic"]
      company_name = state["company_name"]
      print('-- RESPOND TO GENERAL QUERY --')
      # generation = await general_chain.ainvoke({"question": question, "context": doc_txt})
      generation = general_chain.invoke({"question": question, "context": doc_txt, "company_name":company_name})

      return {"response": generation}




######################################### Define our Planning Agent ######################
# Let's now think about creating the planning step. This will use function calling to create a plan.
# Using Pydantic with LangChain
# This notebook uses Pydantic v2 BaseModel, which requires langchain-core >= 0.3. Using langchain-core < 0.3 will result in errors due to mixing of Pydantic v1 and v2 BaseModels.

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
You have access to the following tools: ['web_search', 'annual_reports_search']
ALWAYS try to first use annual_reports_search and if you are unable to find the answer in annual report then you must do a web_search.\
NEVER search an annual report on web_search. Directly search for key words on google.\
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Plan)



######################################### Define our RePlanner Agent ######################
# Now, let's create a step that re-does the plan based on the result of the previous step.
from typing import Union


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


# replanner_prompt = ChatPromptTemplate.from_template(
#     """For the given objective, come up with a simple step by step plan. \
# This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
# The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

# Your objective was this:
# {input}

# Your original plan was this:
# {plan}

# You have currently done the follow steps:
# {past_steps}

# Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
# )

replanner_prompt = ChatPromptTemplate.from_template(
    """**Objective**:
{input}

**Original Plan**:
{plan}

**Completed Steps**:
{past_steps}

**Instructions**:
- If there are more steps needed to achieve the objective, return a **Plan** with the remaining steps.
- If all necessary steps have been completed, return a **Response** to the user based on the information gathered.
- Do **not** include any steps that have already been completed in the new plan.
- Do **not** return an empty plan; if no further steps are needed, you **must** return a **Response**.
- Ensure **Response** is formatted in plain text. Dont split single character text in new lines
- Ensure your output is in the correct structured format as per the `Act` model
- If you are unable to find the answer in annual report then you must do a web_search.

**Remember**:
- The `Act` can be either a `Plan` or a `Response`.
- A `Plan` contains a list of steps that still need to be done.
- A `Response` contains the final answer to the user.

**Provide your output below:**"""
)

replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)


############################ CREATING THE EXECUTION PLAN #######################################
from typing import Literal
from langgraph.graph import END


# async def execute_step(state: PlanExecute):
def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    with st.spinner(f"Executing step: {task}"):
      task_formatted = f"""For the following plan:
  {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
      # agent_response = await agent_executor.ainvoke(
      #     {"messages": [("user", task_formatted)]}
      # )
      agent_response =  agent_executor.invoke(
          {"messages": [("user", task_formatted)]}
      )
      return {
          "past_steps": [(task, agent_response["messages"][-1].content)],
      }


# async def plan_step(state: PlanExecute):
def plan_step(state: PlanExecute):
  with st.spinner("Creating a plan to answer the question"):
    # plan = await planner.ainvoke({"messages": [("user", state["input"])]})
     # Prepare the messages with historical conversations
    history = state["previous_conversations"]  # This should be a list of tuples (role, content)
    user_message = state["input"]  # The current user input
    history_list = []
    for convo_dict in history:
        # Ensure proper key-value handling
        role = convo_dict.get("role", "unknown")  # Default to "unknown" if key is missing
        content = convo_dict.get("content", "")
        history_list.append((role, content))

    # Combine historical messages with the current input
    messages = history_list + [("user", user_message)]

    # Pass the combined messages to the planner
    plan = planner.invoke({"messages": messages})
    # plan = planner.invoke({"messages": [("user", state["input"])]})

    return {"plan": plan.steps}


# async def replan_step(state: PlanExecute):
def replan_step(state: PlanExecute):
  with st.spinner("Evaluating if re-planning is required"):
    # output = await replanner.ainvoke(state)
    output = replanner.invoke(state)

    if isinstance(output.action, Response):
        # save to to a text file
        # with open('check.txt', "w") as file:
        #     file.write(output.action.response)

        # generation = response_format_chain.run({"input_response": output.action.response})
        # generation = clean_and_format_text(output.action.response)

        return {"response": output.action.response }
        # return {"response": generation}
    else:
        return {"plan": output.action.steps}



def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"




################################## CREATING THE GRAPH #########################
#### BUILDING NODES
from langgraph.graph import StateGraph, START

workflow = StateGraph(PlanExecute)
# Add the router node
workflow.add_node("route_question", route_question)

# workflow.add_node("route_direction", route_direction)

# Add the ask more info node
workflow.add_node("ask_for_more_info", ask_for_more_info)

# Add the general response node
workflow.add_node("respond_to_general_query", respond_to_general_query )

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("agent", execute_step)

# Add a replan node
workflow.add_node("replan", replan_step)

# **************************************************************************************************************************************************
### BUILDING EDGES
# workflow.add_edge(START, "planner")
workflow.add_edge(START, "route_question")

workflow.add_conditional_edges(
    "route_question",
    route_direction,
    {
        "general": "respond_to_general_query",
        "more-info": "ask_for_more_info",
        "research": "planner"
    },
)

# From plan we go to agent
workflow.add_edge("planner", "agent")

# From agent, we replan
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["agent", END],
)

# From more_info we go to END
workflow.add_edge("ask_for_more_info", END)

# From general response we go to END
workflow.add_edge("respond_to_general_query", END)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()


##################################################  APP UI ########################################################################
# st.title("Fin-Report-GPT")
################## START THE APP

# config = {"recursion_limit": 50}
# inputs = {"input": "what is the total revenue for 2023?"}
# async for event in app.astream(inputs, config=config):
#     for k, v in event.items():
#         if k != "__end__":
#             print(v)


# Two-column layout in Streamlit
# intilisation
input_pdf_path = st.session_state['pdf_path']
session_id = st.session_state['session_id']
chunks = st.session_state['chunks']
thread_id = st.session_state['thread_id']
# Extract directory and filename without extension
base_directory = os.path.dirname(input_pdf_path)
file_name = os.path.splitext(os.path.basename(input_pdf_path))[0]
# Create output folder path
output_folder_path = os.path.join(base_directory, session_id,file_name + "_output")


# if path exists then delete
if os.path.exists(output_folder_path):
    shutil.rmtree(output_folder_path)
os.makedirs(output_folder_path)
# if file exist then delete
# Define the path to the text file
output_folder_path_text = output_folder_path + 'web_url.txt'
if os.path.exists(output_folder_path_text):
  os.remove(output_folder_path_text)

# remove the log file
remove_shelve_file(f"document_file_{int(st.session_state['session_id'])}.db")

# st.sidebar.write("Input pdf path ",input_pdf_path)
# st.sidebar.write("Output folder path ",output_folder_path)
# st.sidebar.write('path for vectorstore',st.session_state['vectorstore'])

chroma_path = st.session_state['vectorstore']
ticker_name = st.session_state['ticker_name']
selected_company = st.session_state['company_name']


######################################## CHAT AGENT #####################################################

st.markdown(f"""
    <h1 style="text-align: center; font-size: 32px;">
        Annual Report Chat Agent for  <span style="color: #39ff14;">{selected_company}</span>
    </h1>
""", unsafe_allow_html=True)

icon_dict = {"assistant":'https://i.imgur.com/vUru7bM.png',
             "user":"https://i.imgur.com/AykTJiQ.png"}


# Initialize chat history
if "messages" not in st.session_state:
    # ensuring compiler run once per company
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": f"I am an AI assistant here to help answer questions about the annual report for {selected_company}. Feel free to ask any queries!"
        }
    ]
if "feedback" not in st.session_state:
    st.session_state.feedback = []



# Configuration for graph interaction
config = {"recursion_limit": 50 ,"callbacks": [callback_handler],"configurable": {"thread_id": thread_id}}

# Display chat messages from history on app rerun
if "messages" in st.session_state:
  for message in st.session_state.messages:
      with st.chat_message(message["role"],avatar=icon_dict[message["role"]]):
          st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What is your query?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user",avatar="https://i.imgur.com/AykTJiQ.png"):
        st.markdown(prompt)

    # Try to process the graph-based response
    try:
        with st.chat_message("assistant", avatar='https://i.imgur.com/vUru7bM.png'):
            # Interact with the graph asynchronously
            # async def process_graph_input():
            def process_graph_input():
                # async for event in app.astream({"input": prompt}, config=config):
                # filtering the dictionary for last 5 messages excluding the last one
                previous_conversations = st.session_state.messages[-6:]
                if len(previous_conversations) > 5:
                    previous_conversations = previous_conversations[:-1]

                # print(st.session_state.messages)
                for event in app.stream({"previous_conversations": previous_conversations,"input": prompt,"company_name":selected_company}, config=config):
                    for k, v in event.items():
                        if k != "__end__":
                            yield v  # Yield each chunk of the response

            # Collect response in a stream
            response = ''
            query_start_time = time.time()
            run = process_graph_input()
            for chunk in run:
                if 'response' in chunk.keys():
                  response += str(chunk['response'])
                # response += str(chunk)
            query_end_time = time.time()

            # save to check.txt
            with open('check.txt', "w") as file:
                file.write(response)
            # # read from check.txt
            # with open('check.txt', "r") as file:
            #     response = file.read()
            # generation = response_format_chain.run({"input_response": response})
            # generation = clean_and_format_text(generation)
            # fixing the markdown format issue caused by a $ sign
            response = response.replace('$', '\\$')
            st.markdown(response)  # Update the displayed response dynamically
            st.empty()

            # Append the final response to the session state
            st.session_state.messages.append({"role": "assistant", "content": response})

            ################### FEEDBACK COLLECTION ##########################
            colA, colB = st.columns(2)
            with colA:
              # Feedback section directly below the AI response
              st.write("Was this response helpful?")
              st.session_state.get_feedback = True
              st.session_state.prompt = prompt
              st.session_state.response = response
              st.session_state.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
              if st.session_state.get_feedback:
                st.feedback("thumbs", key="feedback", on_change=save_feedback)


            with colB:
              # st.write(f'<p style="font-size: 10px; text-align: right; color: #2EA9E0;">Query Cost (USD): ${np.round(callback_handler.total_cost, 4)}</p>', unsafe_allow_html=True)
              # Obtain time in seconds
              query_time = query_end_time - query_start_time
              # st.write(f'<p style="font-size: 10px; text-align: right; color: #2EA9E0;">Query Run Time (s): {np.round(query_time, 0)}</p>', unsafe_allow_html=True)
              # add cost and runtime in oneline
              st.write(f'<p style="font-size: 12px; text-align: right; color: #2EA9E0;">${np.round(callback_handler.total_cost, 4)}, {int(query_time)} secs</p>', unsafe_allow_html=True)

    except Exception as e:
        # Handle any errors that occur during processing
        st.error(f"An error occurred while processing your query: {e}")



# Load from shelve
with st.spinner("Updating Sources used and Session Metrics"):
  if os.path.exists(f"document_file_{int(st.session_state['session_id'])}.db"): 
    with shelve.open(f"document_file_{int(st.session_state['session_id'])}") as db:
        loaded_data = db['documents']
        files = db['file_path']
        path = db['output_path']
    # extract page numbers
    page_numbers = set()
    for data in loaded_data:
      # extract all page numbers
      page_numbers.add(data.metadata['page_number'])
    # saving images
    save_page(file_path= files, doc_list=loaded_data, page_numbers= page_numbers, output_path=path , print_text=False)




with st.sidebar:
  # List of options
  source_options = ["Annual Report","Web"]
  body = "Sources Used"
  st.sidebar.header(body, divider=True)
  # Multi-select dropdown
  # selected_sources_options = st.pills("Select Source for Q&A", source_options, selection_mode="multi" ,default = [ "Annual Report","Web"])
  # st.write('Select Source for Q&A')
  c1,c2 = st.columns(2)
  with c1:
    ARoption_1 = st.checkbox("Annual Report",value = True)
  with c2:
    WBoption_2 = st.checkbox("Web",value = True)




if ARoption_1:
  # Check if the folder exists
  if os.path.isdir(output_folder_path):
      image_paths = load_images_from_folder(output_folder_path)

      # # Clean up folder if necessary
      # if os.path.exists(output_folder_path):
      #     shutil.rmtree(output_folder_path)

      # # Creating directory
      # os.makedirs(output_folder_path)

      if image_paths:
          with st.sidebar.expander("Source: Annual Report Pages", expanded=False):
              # Display image thumbnails
              thumbnails = [Image.open(img_path).resize((100, 100)) for img_path in image_paths]
              selected_index = clickable_images(
                  [f"data:image/png;base64,{img_path}" for img_path in thumbnails],
                  titles=[f"Image #{str(i)}" for i in range(len(thumbnails))]
              )

              # if selected_index is not None:
              #     # Show the clicked image in a dialog
              #     selected_image_path = image_paths[selected_index]
              #     # with st.dialog(title="Image"):
              for i in range(len(thumbnails)):
                  selected_image_path = image_paths[i]
                  display_name = "Image_" + str(i)
                  st.image(selected_image_path, caption=display_name)
      else:
          with st.sidebar.expander("Source: Annual Report Pages", expanded=False):
            # st.sidebar.warning("No search conducted on Annual Report")
            print("No search conducted on Annual Report")
  else:
    with st.sidebar.expander("Source: Annual Report Pages", expanded=False):
      # st.sidebar.warning("No search conducted on Annual Report")
      print("No search conducted on Annual Report")



if WBoption_2:
  # presenting web urls
  try:
    # Read the text file
    with open(output_folder_path_text, "r") as file:
        urls = file.readlines()

    # Format the URLs for display
    formatted_urls = "\n".join([url.strip() for url in urls])

    # Display the URLs in an expander
    with st.sidebar.expander("Source: Web Search", expanded=False):
        st.code(formatted_urls, language="text")

  except FileNotFoundError:
    with st.sidebar.expander("Source: Web Search", expanded=False):
      # st.sidebar.warning(f"No Web Search conducted")
      print(f"No Web Search conducted")




# Log file path
import csv
from datetime import datetime
log_folder = "log"
log_file = os.path.join(log_folder, "runtime_log.csv")

# Create the log folder if it doesn't exist
os.makedirs(log_folder, exist_ok=True)

# Define the header for the CSV file
header = ["timestamp", "session_id",  "total_tokens", "prompt_tokens", "completion_tokens", "successful_requests", "total_cost"]

# Function to log the data to the CSV file
def log_runtime_info( callback_handler):
    # Collect runtime information
    session_id = st.session_state["session_id"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_tokens = callback_handler.total_tokens
    prompt_tokens = callback_handler.prompt_tokens
    completion_tokens = callback_handler.completion_tokens
    successful_requests = callback_handler.successful_requests
    total_cost = callback_handler.total_cost

    # Check if the log file exists
    file_exists = os.path.exists(log_file)

    # Open the CSV file in append mode
    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write the header if the file is empty
        if not file_exists:
            writer.writerow(header)

        # Append the new data row
        writer.writerow([timestamp, session_id, total_tokens, prompt_tokens, completion_tokens, successful_requests, total_cost])

# Log the runtime information
log_runtime_info(callback_handler=callback_handler)

###################################################################################################################################
# Function to read and filter the log file by session ID
def get_session_metrics(session_id,log_file):
    # if os.path.exists(log_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(log_file)

    # Filter the DataFrame by session ID
    filtered_df = df.loc[df["session_id"] == session_id].reset_index(drop=True)

    # Check if any data was found for the session ID
    if not filtered_df.empty:
        return filtered_df
    else:
        return None
    # else:
    #     return None

# Function to display metrics in an expander
def display_session_metrics(session_id,log_file):
    # Get the metrics for the session_id
    metrics = get_session_metrics(session_id,log_file)

    if metrics is not None:
        body = "Session Metrics"
        st.sidebar.header(body, divider=True)
        with st.sidebar.expander("Select to View Session Metrics", expanded=False):
            # Display the metrics in a beautiful format
            print(f"### Metrics for Session ID: {session_id}")
            st.write(f"**Total Tokens Used:** {metrics['total_tokens'].sum()}")
            st.write(f"**Prompt Tokens:** {metrics['prompt_tokens'].sum()}")
            st.write(f"**Completion Tokens:** {metrics['completion_tokens'].sum()}")
            st.write(f"**Successful Requests:** {metrics['successful_requests'].sum()}")
            st.write(f"**Total Cost (USD):** ${metrics['total_cost'].sum():.2f}")
            st.write(f"**Total Queries:** {len(metrics)}")
    else:
        body = "Session Metrics"
        st.sidebar.header(body, divider=True)
        st.sidebar.warning(f"No metrics found for Session ID: {session_id}")

# Example usage: Use the session_id from callback_handler or provide one manually
session_id = int(st.session_state.session_id)  # Replace with the dynamic session ID or callback_handler session ID

# Call the function to display the metrics in the sidebar expander
display_session_metrics(session_id,log_file)
########################################################################################################

# with st.sidebar:
#   # Access and print the token usage information
#   st.write(f"Total Tokens Used: {callback_handler.total_tokens}")
#   st.write(f"Prompt Tokens: {callback_handler.prompt_tokens}")
#   st.write(f"Completion Tokens: {callback_handler.completion_tokens}")
#   st.write(f"Successful Requests: {callback_handler.successful_requests}")
#   st.write(f"Total Cost (USD): ${callback_handler.total_cost}")


