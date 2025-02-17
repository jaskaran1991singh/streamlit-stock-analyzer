import os
import time
import random
import shutil
import requests
import numpy as np
import pandas as pd
import dotenv
import nltk
from datetime import datetime
from typing import Dict, List, Tuple, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# Streamlit
import streamlit as st

# OpenAI & LangChain
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.vectorstores import Chroma

# Finance & API-related
import yfinance as yf  # Assumes you use yfinance for company data
from sec_api import QueryApi, RenderApi

# Pandas parallel processing
from pandarallel import pandarallel

# PDF Processing & Rendering
from unstructured.partition.pdf import partition_pdf
from weasyprint import HTML

# ChromaDB
import chromadb

# NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load environment variables
dotenv.load_dotenv()


# Access the keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
SEC_API_KEY = os.getenv("SEC_API_KEY")


#Enter your OpenAI API Key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_TRACING_V2"] = str(LANGCHAIN_TRACING_V2)
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
renderApi = RenderApi(SEC_API_KEY)
queryApi = QueryApi(api_key=SEC_API_KEY)

if TAVILY_API_KEY==False:
    st.info("Please add your Tavily API key to continue.")
if LANGCHAIN_API_KEY==False:
    st.info("Please add your Langchain API key to continue.")
if OPENAI_API_KEY==False:
    st.info("Please add your OpenAI API key to continue.")


st.set_page_config(layout="wide")



def generate_session_id():
    # return str(uuid.uuid4())
    from datetime import datetime
    # Get the current datetime
    current_datetime = datetime.now()

    # Convert to string representation in seconds since epoch
    current_datetime_in_seconds = str(int(current_datetime.timestamp()))
    return current_datetime_in_seconds

TIMEOUT_DURATION = 10 * 60  # 10 minutes

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


# function to download 10k filings
def get_10K_filing_urls(ticker_batch):
    if ticker_batch is None:
        return []

    # ticker_batch = row["Tickers"]

    if len(ticker_batch) == 0:
        return []
    # create a query string to search for 10-K filings for the given tickers
    # "(ticker:AAPL OR ticker:MSFT OR ...)"
    ticker_query = " OR ".join([f'ticker:"{ticker}"' for ticker in ticker_batch])
    ticker_query = f"({ticker_query})"
    # search for 10-K filings filed between 2014-01-01 and 2023-12-31
    date_query = "filedAt:[2014-01-01 TO 2023-12-31]"
    # exclude 10-K/A and NT 10-K filings
    form_type_query = 'formType:"10-K" AND NOT formType:"10-K/A" AND NOT formType:"NT"'

    search_query = f"{ticker_query} AND {date_query} AND {form_type_query}"

    search_params = {
        "query": search_query,
        "from": 0,
        "size": 50,
        "sort": [{"filedAt": {"order": "desc"}}],
    }

    print(f"Fetching filings for {ticker_batch[:4]}...\n")

    has_more_filings = True
    filing_urls = []

    while has_more_filings:

        search_results = queryApi.get_filings(search_params)
        filings = search_results["filings"]

        if len(filings) == 0:
            break

        # extract metadata for each filing
        # { "ticker": "...", "cik": "...", "filedAt": "...", "filingUrl": "..." }
        metadata = list(
            map(
                lambda f: {
                    "ticker": f["ticker"],
                    "cik": f["cik"],
                    "filedAt": f["filedAt"],
                    "accessionNo": f["accessionNo"],
                    "filingUrl": f["linkToFilingDetails"],
                },
                filings,
            )
        )

        filing_urls.extend(metadata)

        search_params["from"] += search_params["size"]

    metadata = pd.DataFrame(filing_urls)
    # Convert 'filedAt' to datetime format for sorting
    metadata["filedAt"] = pd.to_datetime(metadata["filedAt"])

    return metadata



def download_filing(row):
    ticker = row["ticker"]
    accessionNo = row["accessionNo"]
    # filedAt = row["filedAt"].split("T")[0]
    filedAt = row["filedAt"].date()
    filing_url = row["filingUrl"]

    try:
        content = renderApi.get_filing(filing_url)

        # check if path "filings/{ticker}/" exists. if not, create it
        if not os.path.exists(f"filings/{ticker}/"):
            os.makedirs(f"filings/{ticker}/")

        file_type = filing_url.split("/")[-1].split(".")[1]
        local_file_name = f"filings/{ticker}/{filedAt}_{accessionNo}.{file_type}"

        with open(local_file_name, "w") as f:
            f.write(content)
        return local_file_name
        print(f"✅ Downloaded {local_file_name}")
    except:
        print(f"❌ {ticker}: downloaded failed for {filing_url}")


def convert_htm_to_pdf(input_html, output_pdf):
    try:
        HTML(input_html).write_pdf(output_pdf)
        print(f"Converted {input_html} to {output_pdf} successfully.")
    except Exception as e:
        print(f"Error during conversion: {e}")



def create_vectorstore(file_path):
  print(file_path)
  filename = os.path.basename(file_path)  # Get the file name with extension
  basename = os.path.splitext(filename)[0]  # Remove the extension
  print(basename)
  # Remove chroma_db folder if exists
  chroma_path = os.path.join(os.getcwd(),st.session_state["session_id"],basename,"chroma_db")
  if os.path.exists(chroma_path):
    shutil.rmtree(chroma_path)
    chromadb.api.client.SharedSystemClient.clear_system_cache()

  chunks = partition_pdf(
    filename=file_path,
    # infer_table_structure=True,            # extract tables
    strategy="fast",                     # mandatory to infer tables

    # extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
    # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

    extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

    chunking_strategy="by_title",          # or 'basic'
    max_characters=10000,                  # defaults to 500
    combine_text_under_n_chars=2000,       # defaults to 0
    new_after_n_chars=6000,

    # extract_images_in_pdf=True,          # deprecated
    )
  # Convert all chunks to LangChain-compatible Document objects
  documents = [
      Document(
          page_content=chunk.to_dict().get('text', ''),  # Extract text safely
          metadata={
              "source": chunk.to_dict().get('metadata', {}).get('filename', 'Unknown Source'),
              "element_id": chunk.to_dict().get('element_id', 'Unknown ID')  # Extract element ID
          }
      )
      for chunk in chunks
      if hasattr(chunk, "to_dict") and chunk.to_dict().get('text', '').strip()  # Ensure chunk has text
  ]


  # Creating a vectorstore
  # st.write("stepC")
  try:
    # Create a Chroma vectorstore with the specified documents and embeddings
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory=chroma_path
    )

  except Exception as e:
    # Handle errors
    st.write(f"Error details: {e}")

  page_number = chunks[len(chunks)-1].to_dict().get('metadata', {}).get('page_number', 0)
  # randomw between 1 to 1000
  thread_id = random.randint(1, 1000)
  return page_number,chroma_path,chunks,thread_id


# Function to fetch ticker symbol dynamically
def get_ticker(company_name):
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    company_code = data['quotes'][0]['symbol']
    return company_code



# Mock data for company names and tickers
COMPANY_DETAIL_DICT = {
    'Shopify': './pdfs/NYSE_SHOP_2023.pdf',
    'Royal Bank Of Canada': './pdfs/ar_2024_e.pdf',
    'Toronto Dominion Bank': './pdfs/2023-annual-report-e.pdf',
    'Enbridge': './pdfs/TSX_ENB_2023.pdf',
    'Brookfield Corporation': './pdfs/Full_Annual_Report_BN.pdf',
    'Thomson Reuters': './pdfs/TRI_2023_Annual _Report_AODA.pdf',
    'Canadian Pacific Railway': './pdfs/TSX_CP_2023.pdf',
    'Constellation Software': './pdfs/Q4-2023-Shareholder-Report.pdf',
    "Alimentation Couche-Tard": './pdfs/COTA78_ACT_Annual-Report_WEB_EN_20240626.pdf',
    'Suncor Energy': './pdfs/2023-annual-report-en.pdf',
    'Lululemon Athletica': './pdfs/lululemon-2023-annual-report.pdf',
    'Imperial Oil': './pdfs/TSX_IMO_2023.pdf',
    'Dollarama': './pdfs/2024-Annual-Information-Form-ENG-Final.pdf'
}
COMPANY_NAME_LIST = list(COMPANY_DETAIL_DICT.keys())

##################################################  APP UI ########################################################################

st.markdown(custom_button_style, unsafe_allow_html=True)

if "status" not in st.session_state:
  st.session_state["status"] = 0
if "vectorstore" not in st.session_state:
  st.session_state["vectorstore"] = None
if "company_name" not in st.session_state:
  st.session_state["company_name"] = 'Default Name'
if "chunks" not in st.session_state:
  st.session_state["chunks"] = None
if "file_name" not in st.session_state:
  st.session_state["file_name"] = None
if "ticker_name" not in st.session_state:
  st.session_state["ticker_name"] = None
if "ticker_placeholder" not in st.session_state:
  st.session_state["ticker_placeholder"] = None
if "session_id" not in st.session_state:
    st.session_state["session_id"] = generate_session_id()
    st.session_state["last_activity"] = time.time()  # Set the last activity time to current time
if "messages" in st.session_state:
  del st.session_state["messages"]

# Get the current time and check for inactivity
current_time = time.time()
inactive_duration = current_time - st.session_state["last_activity"]

# If inactivity exceeds the timeout, reset the session state
if inactive_duration > TIMEOUT_DURATION:
    st.session_state.clear()  # Clear the session state, effectively "closing" the session
    st.warning("Your session has expired due to inactivity. Please refresh the page.")
    st.stop()  # Stop the script to prevent further execution



# st.write("Step1:" + str(st.session_state["status"]) )
# Dropdown or custom input option
body = "How would you like to select the company?"
st.sidebar.header(body, divider=True)  # Add a header with a divider on the sidebar
selection_mode = st.sidebar.radio(
    "Select an option from below",
    options=["Select from list", "Type your own", "Upload an Annual Report"],
    index=0
)

body = "Runtime Status"
st.sidebar.header(body, divider=True)

col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
# st.write("Step1:" + str(st.session_state["status"]) )

with col2:
  st.title("AI-Powered Business Analyst")
  ############################# Select from list ###################################
  if selection_mode == "Select from list":
    # st.title("Company Selection")
    body = "Select a company for analysis"
    st.header(body,divider=True)
    selected_company = st.selectbox("Select a company", COMPANY_NAME_LIST)
    if selected_company!=st.session_state["company_name"]:
      # status update
      st.session_state["status"] = 0

    if np.logical_or(st.button("Confirm Company",type="primary"),st.session_state["status"] == 4):
      st.session_state["company_name"] = selected_company
      st.sidebar.success(f"You have selected {selected_company}")

      # st.write("Step2:" + str(st.session_state["status"]) )

      if st.session_state["status"] == 4:
        ticker_input = st.session_state["ticker_placeholder"]
        len_doc = st.session_state["len_doc"]
        st.sidebar.success(f"The Ticker for {selected_company} is '{ticker_input}'.")
        st.sidebar.success("Annual Report Loaded")
        st.sidebar.success("Vector Database Created")
        # st.write("Moving to the Financial Summary Page")
        st.sidebar.write("Number of pages processed: ", len_doc)
        st.switch_page("pages/Stock_Analysis.py")
      else:
        if st.session_state["status"] == 0:
          # status update
          st.session_state["status"] = 1

        try:
            ticker_placeholder = get_ticker(selected_company)
            if ticker_placeholder is None:
                raise ValueError("Ticker not found")
        except Exception as e:
            st.error(f"Error fetching ticker: {e}")
            ticker_placeholder = "Unable to Find a Ticker Name"  # Fallback value

        st.session_state["ticker_placeholder"] = ticker_placeholder
        st.session_state["ticker_name"] = ticker_placeholder
        ticker_input = ticker_placeholder
        st.sidebar.success(f"The Ticker for {selected_company} is '{ticker_input}'.")
        # status update
        st.session_state["status"] = 2
        # Download the latest annual report
        # Add a spinner to the sidebar
        with st.sidebar:
          with st.spinner("Downloading the latest annual report..."):
            if os.path.exists("filings"):
              shutil.rmtree("filings")
            # obtain pdf location
            input_pdf = COMPANY_DETAIL_DICT[selected_company]
            st.session_state['pdf_path'] = input_pdf
            st.sidebar.success("Annual Report Loaded")
            st.session_state["status"] = 3
          with st.spinner("Creating Vector Database for the Downloaded Annual Report..."):
            # Loading the pdf
            len_doc,st.session_state['vectorstore'],st.session_state['chunks'] ,st.session_state['thread_id']= create_vectorstore(input_pdf)
            # st.sidebar.write("Vector Database Created :",st.session_state['vectorstore'])
            st.sidebar.success("Vector Database Created")
            st.sidebar.write("Number of pages processed: ", len_doc)
            st.session_state["len_doc"] = len_doc
            st.session_state["status"] = 4
        body = f"Click the button below to run financial analysis for **{st.session_state['company_name']}**."
        st.write(body)
        # Inject custom styles
        st.markdown(custom_button_style, unsafe_allow_html=True)
        if st.button("GO",use_container_width=True, type="primary"):
          st.write('Loading ...')
  ############################# Upload an Annual Report ###################################
  elif selection_mode == "Type your own":
    body = "Type a company name for analysis"
    st.header(body,divider=True)
    selected_company = st.text_input("Type your own company name", "Type Here..")
    if selected_company!=st.session_state["company_name"]:
      st.session_state["status"] = 0
    if np.logical_or(st.button("Confirm Company",type="primary"),st.session_state["status"] >= 1):
      st.sidebar.success(f"You have selected {selected_company}")
      st.session_state["company_name"] = selected_company

      if st.session_state["status"] == 4:
        ticker_input = st.session_state["ticker_placeholder"]
        len_doc = st.session_state["len_doc"]
        st.sidebar.success(f"The Ticker for {selected_company} is '{ticker_input}'.")
        st.sidebar.success("Annual Report Loaded")
        st.sidebar.success("Vector Database Created")
        # st.write("Moving to the Financial Summary Page")
        st.sidebar.write("Number of pages processed: ", len_doc)
        st.switch_page("pages/Stock_Analysis.py")

      if st.session_state["status"] == 0:
        st.session_state["status"] = 1

      # Identifying ticker name of the company
      try:
        ticker_placeholder = get_ticker(selected_company)
        if ticker_placeholder is None:
            raise ValueError("Ticker not found")
      except Exception as e:
        st.sidebar.error(f"Error fetching ticker: {e}")
        ticker_placeholder = "Unable to Find a Ticker Name"  # Fallback value

      st.session_state["ticker_placeholder"] = ticker_placeholder

      # Display text input for ticker, allowing user to edit
      body = f"Do we have the right ticker for {selected_company}"
      st.header(body,divider=True)
      ticker_input = st.text_input("Select Ticker Symbol", value=ticker_placeholder, key="ticker_input")
      st.session_state["ticker_name"] = ticker_input
      if st.session_state["status"] == 1:
        st.session_state["status"] = 2

      # Button to confirm the ticker
      if np.logical_or(st.button("Confirm Ticker",type="primary"),st.session_state["status"] >= 3):
        if st.session_state["status"] == 2:
          st.session_state["status"] = 3
        # st.sidebar.success(f"Ticker for {selected_company} is confirmed as '{ticker_input}'.")
        # ticker_placeholder = st.session_state["ticker_placeholder"]
        if ticker_input == ticker_placeholder:
            st.sidebar.success(f"The Ticker for {selected_company} is confirmed as '{ticker_input}'.")
        else:
            st.sidebar.warning(f"Ticker updated to '{ticker_input}'. Ensure this is correct for {selected_company}.")

        if st.session_state["status"] == 3:
          try:
            with st.sidebar:
              with st.spinner("Downloading the latest annual report..."):
                # Downloading Annual Report
                metadata = get_10K_filing_urls([ticker_input])
                # Sort by 'filedAt' in descending order and select the latest row
                filtered_metadata = metadata.sort_values(by="filedAt", ascending=False)
                filtered_metadata= filtered_metadata.reset_index(drop=True)
                latest_row = filtered_metadata.iloc[0]
                # Delete filings folder if exists
                if os.path.exists("filings"):
                    shutil.rmtree("filings")
                # Example usage
                input_html = download_filing(latest_row)
                output_pdf = input_html.rstrip('.htm') + '.pdf'
                # output_pdf= "./temp.pdf"
                st.session_state['pdf_path'] = output_pdf
                convert_htm_to_pdf(input_html, output_pdf)
                st.sidebar.success(f"Annual Report Downloaded")

              with st.spinner("Creating Vector Database for the Downloaded Annual Report..."):
                # Loading the pdf
                len_doc ,st.session_state['vectorstore'],st.session_state['chunks'],st.session_state['thread_id'] = create_vectorstore(output_pdf)
                st.sidebar.success("Vector Database Created")
                st.sidebar.write("Number of pages processed: ", len_doc)
                st.session_state["len_doc"] = len_doc
                st.session_state["status"] = 4

          except Exception as e:
            st.sidebar.error(f"An error occurred: {e}")


        body = f"Click the button below to run financial analysis for **{st.session_state['company_name']}**."
        st.write(body)
        # Inject custom styles
        st.markdown(custom_button_style, unsafe_allow_html=True)
        if st.button("GO",type="primary",use_container_width=True):
          st.write('Loading ...')

  elif selection_mode == "Upload an Annual Report":
    # selected_company = 'NA'
    # st.title("Upload the File here:")
    body = "Upload the file here:"
    st.header(body,divider=True)
    # pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=False)
    uploaded_file = st.file_uploader('Upload your PDF Files and Click on the Submit & Process Button', type="pdf", accept_multiple_files=False)
    if uploaded_file:
      # file_path = os.path.join(os.getcwd(),'pdfs',st.session_state["session_id"],"temp.pdf")
      file_path = st.session_state["session_id"] + '_temp.pdf'
      st.session_state['pdf_path'] = file_path
      with open(file_path, "wb") as file:
        file.write(uploaded_file.getvalue())
        file_name = st.session_state["file_name"]
        st.sidebar.success("File Uploaded")
        if st.button("Submit & Process",type="primary"):
          with st.sidebar:
            with st.spinner("Creating Vector Database"):
              # Loading the pdf
              # st.write(file_path)
              if file_name != uploaded_file.name:
                len_doc,st.session_state['vectorstore'],st.session_state['chunks'],st.session_state['thread_id'] = create_vectorstore(file_path)
                file_name = uploaded_file.name
                st.session_state["file_name"] = file_name
                st.session_state["len_doc"] = len_doc
                st.sidebar.success("Vector Database Created")
                len_doc = st.session_state["len_doc"]
                st.write("Number of pages processed: ", len_doc)
                st.session_state["status"] = 0.5
              else:
                # len_doc = st.session_state["len_doc"]
                st.sidebar.success("Vector Database Created")
                # st.write("Number of pages processed: ", len_doc)
      if st.session_state["status"] >= 0.5:
        temp_company_name = st.session_state["company_name"]
        body = "Confirm company name"
        st.header(body,divider=True)
        selected_company = st.text_input("Type Company Name", temp_company_name)

        if np.logical_or(st.button("Confirm Company",type="primary"),st.session_state["status"] >= 1):
          st.session_state["company_name"] = selected_company
          st.sidebar.success("Vector Database Created")
          st.sidebar.success(f"You have selected {selected_company}")
          if st.session_state["status"] == 0.5:
            st.session_state["status"] = 1

        # Identifying the ticker name
        if st.session_state["status"] >= 1:
          # Identifying ticker name of the company
          try:
            ticker_placeholder = get_ticker(selected_company)
            if ticker_placeholder is None:
                raise ValueError("Ticker not found")
          except Exception as e:
            st.sidebar.error(f"Error fetching ticker: {e}")
            ticker_placeholder = "Unable to Find a Ticker Name"  # Fallback value

          st.session_state["ticker_placeholder"] = ticker_placeholder

          # Display text input for ticker, allowing user to edit
          body = f"Do we have the right ticker for {selected_company}"
          st.header(body,divider=True)
          ticker_input = st.text_input("Select Ticker Symbol", value=ticker_placeholder, key="ticker_input")
          st.session_state["ticker_name"] = ticker_input
          if st.session_state["status"] == 1:
            st.session_state["status"] = 2
          # Button to confirm the ticker
          if np.logical_or(st.button("Confirm Ticker",type="primary"),st.session_state["status"] >= 3):
            if st.session_state["status"] == 2:
              st.session_state["status"] = 3
            # st.sidebar.success(f"Ticker for {selected_company} is confirmed as '{ticker_input}'.")
            # ticker_placeholder = st.session_state["ticker_placeholder"]
            if ticker_input == ticker_placeholder:
                st.sidebar.success(f"The Ticker for {selected_company} is confirmed as '{ticker_input}'.")
            else:
                st.sidebar.warning(f"Ticker updated to '{ticker_input}'. Ensure this is correct for {selected_company}.")


            body = f"Click the button below to run financial analysis for **{st.session_state['company_name']}**."
            st.write(body)
            # Inject custom styles
            st.markdown(custom_button_style, unsafe_allow_html=True)

            if st.button("GO",type="primary",use_container_width=True):
              # st.write('Loading ...')
              ticker_input = st.session_state["ticker_placeholder"]
              st.sidebar.success(f"The Ticker for {selected_company} is '{ticker_input}'.")
              st.write("Moving to the Financial Summary Page")
              len_doc = st.session_state["len_doc"]
              st.sidebar.write("Number of pages processed: ", len_doc)
              st.switch_page("pages/Stock_Analysis.py")


  else:
    st.write("Invalid selection mode")
