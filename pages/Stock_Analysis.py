
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
  st.session_state["status"] = 3



with st.sidebar:
  # Add title in the sidebar
  st.markdown(custom_button_style, unsafe_allow_html=True)
  st.header('How would you like to analyse?')
  # Add buttons to the sidebar
  financial_summary_button = st.button('Stock Analysis', key='financial_summary_button', help="Navigate to Stock Analysis",use_container_width=True,type="primary")
  qa_button = st.button('Ask Questions', key='qa_button', help="Navigate to the Q&A agent",use_container_width=True)

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

# title format
tile_style = """
    <style>
        .tile-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        .tile {
            background-color: #181825; /* Deep Dark Blue-Grey */
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 255, 170, 0.15); /* Soft green glow */
            padding: 20px;
            margin-bottom: 20px;
            text-align: center;
            border: 1px solid #2a2a3a; /* Subtle border */
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .tile:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 15px rgba(0, 255, 170, 0.25); /* Glow effect */
        }
        .tile-header {
            font-size: 1.2rem;
            font-weight: bold;
            color: #00FFA2; /* Vibrant neon green */
            margin-bottom: 10px;
        }
        .tile-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #E5E5E5; /* Soft white */
        }
    </style>
"""



# Custom styles for dark mode containers
custom_style = """
    <style>
        .summary-container {
            background-color: #181825; /* Deep Dark Blue-Grey */
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0, 255, 170, 0.15); /* Subtle Neon Green Glow */
            border: 1px solid #2a2a3a; /* Subtle border */
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .summary-container:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 18px rgba(0, 255, 170, 0.25); /* Stronger Glow on Hover */
        }
        .summary-header {
            font-size: 1.6rem;
            font-weight: bold;
            color: #00FFA2; /* Neon Green for Better Visibility */
            margin-bottom: 12px;
            text-transform: uppercase;
        }
        .summary-content {
            font-size: 1.1rem;
            color: #E5E5E5; /* Soft White for Readability */
            line-height: 1.6;
        }
    </style>
"""

def generate_summary(key, data_summary_dict):
    header = key.replace('_', ' ').title()
    content = data_summary_dict[key]

    summary_html = f"""
    <style>
    .summary-container {{
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 250px;  /* Uniform height */
        padding: 18px;
        border-radius: 10px;
        background-color: #181825; /* Deep Dark Blue-Grey */
        border: 1px solid #2a2a3a;
        box-shadow: 0 4px 12px rgba(0, 255, 170, 0.15); /* Subtle Neon Green Glow */
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .summary-container:hover {{
        transform: translateY(-4px);
        box-shadow: 0 6px 18px rgba(0, 255, 170, 0.25); /* Stronger Glow on Hover */
    }}
    .summary-header {{
        font-size: 1.2rem;
        font-weight: bold;
        color: #00FFA2; /* Neon Green */
        text-transform: uppercase;
        margin-bottom: 8px;
    }}
    .summary-content {{
        font-size: 1rem;
        color: #E5E5E5; /* Soft White */
        flex-grow: 1;
        overflow-y: auto;
        padding-right: 10px;
    }}
    </style>

    <div class="summary-container">
        <div class="summary-header">{header}</div>
        <div class="summary-content">{content}</div>
    </div>
    """

    st.markdown(summary_html, unsafe_allow_html=True)



def convert_to_dataframe(data):
      # Parse the string into a Python dictionary
      data_dict = ast.literal_eval(data)

      # Extract the stock data
      stock_data = data_dict['stock_price']

      # Convert to a DataFrame
      df = pd.DataFrame(stock_data)

      # Flatten MultiIndex columns
      df.columns = [' '.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

      # Return the DataFrame
      return df

import plotly.graph_objects as go
import streamlit as st

def plot_stock_data(df, ticker_input):
    # Define custom colors for each trace
    colors = {
        "Close": "#00FFA2",  # Neon Green
        "High": "#73D0FF",   # Light Blue
        "Low": "#FF73B5",    # Soft Pink
        "Open": "#FFB347"    # Warm Orange
    }

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces with improved styling
    for metric in ["Close", "High", "Low", "Open"]:
        fig.add_trace(go.Scatter(
            x=df["Date"], 
            y=df[f"{metric} {ticker_input}"], 
            mode="lines+markers", 
            name=f"{metric} {ticker_input}",
            line=dict(color=colors[metric], width=2, shape="spline"),  # Smooth curves
            marker=dict(size=6, symbol="circle", line=dict(width=1, color="white"))  # Improved marker styling
        ))

    # Customize the layout for dark mode
    fig.update_layout(
        title=f"{ticker_input} Stock Trends",
        xaxis_title="Date",
        yaxis_title="Price (CAD)",
        legend_title="Metrics",
        template="plotly_dark",
        title_font=dict(size=22),  
        margin=dict(l=40, r=40, t=60, b=40),  # Better spacing
        xaxis=dict(
            showgrid=True,        
            gridcolor="#444",    
            showline=True,        
            linecolor="white",    
            linewidth=2,          
            tickangle=45,
            tickfont=dict(size=12, color="white"),
        ),
        yaxis=dict(
            showgrid=True,        
            gridcolor="#444",     
            showline=True,        
            linecolor="white",    
            linewidth=2,
            tickfont=dict(size=12, color="white"),
        ),
        legend=dict(
            font=dict(size=12, color="white"),
            bgcolor="rgba(0,0,0,0.5)",  # Semi-transparent background
            bordercolor="white",
            borderwidth=1
        ),
        hovermode="x unified",  # Show all values on hover
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)




##################################### FINANCIAL ANALYST SUMMARY ################################
FUNDAMENTAL_ANALYST_PROMPT = """
You are a fundamental analyst specializing in evaluating company (whose symbol is {company}) performance based on stock prices, technical indicators, and financial metrics. Your task is to provide a comprehensive summary of the fundamental analysis for a given stock.

You have access to the following tools:
1. **get_stock_prices**: Retrieves the latest stock price, historical price data and technical Indicators like RSI, MACD, Drawdown and VWAP.
2. **get_financial_metrics**: Retrieves key financial metrics, such as revenue, earnings per share (EPS), price-to-earnings ratio (P/E), and debt-to-equity ratio.

### Your Task:
1. **Input Stock Symbol**: Use the provided stock symbol to query the tools and gather the relevant information.
2. **Analyze Data**: Evaluate the results from the tools and identify potential resistance, key trends, strengths, or concerns.
3. **Provide Summary**: Write a concise, well-structured summary that highlights:
    - Recent stock price movements, trends and potential resistance.
    - Key insights from technical indicators (e.g., whether the stock is overbought or oversold).
    - Financial health and performance based on financial metrics.

### Constraints:
- Use only the data provided by the tools.
- Avoid speculative language; focus on observable data and trends.
- If any tool fails to provide data, clearly state that in your summary.

### Output Format:
Respond in the following format:
"stock": "<Stock Symbol>",
"price_analysis": "<Detailed analysis of stock price trends>",
"technical_analysis": "<Detailed time series Analysis from ALL technical indicators>",
"financial_analysis": "<Detailed analysis from financial metrics>",
"final Summary": "<Full Conclusion based on the above analyses>"
"Asked Question Answer": "<Answer based on the details and analysis above>"

Ensure that your response is objective, concise, and actionable.
"""


@tool
def get_stock_prices(ticker: str) -> Union[Dict, str]:
    """Fetches historical stock price data and technical indicator for a given ticker."""
    # with st.spinner(f'Fetching price data for {ticker}..'):
    try:
      data = yf.download(
          ticker,
          start=dt.datetime.now() - dt.timedelta(weeks=24*3),
          end=dt.datetime.now(),
          interval='1wk'
      )
      df= data.copy()
      df.columns = df.columns.get_level_values(0)
      data.reset_index(inplace=True)
      data.Date = data.Date.astype(str)

      indicators = {}

      # Momentum Indicators
      rsi_series = RSIIndicator(df['Close'], window=14).rsi().iloc[-12:]
      indicators["RSI"] = {date.strftime('%Y-%m-%d'): int(value) for date, value in rsi_series.dropna().to_dict().items()}
      sto_series = StochasticOscillator(
          df['High'], df['Low'], df['Close'], window=14).stoch().iloc[-12:]
      # print(sto_series)
      indicators["Stochastic_Oscillator"] = {date.strftime('%Y-%m-%d'): int(value) for date, value in sto_series.dropna().to_dict().items()}

      macd = MACD(df['Close'])
      macd_series = macd.macd().iloc[-12:]
      # print(macd_series)
      indicators["MACD"] = {date.strftime('%Y-%m-%d'): int(value) for date, value in macd_series.to_dict().items()}
      macd_signal_series = macd.macd_signal().iloc[-12:]
      # print(macd_signal_series)
      indicators["MACD_Signal"] = {date.strftime('%Y-%m-%d'): int(value) for date, value in macd_signal_series.to_dict().items()}

      vwap_series = volume_weighted_average_price(
          high=df['High'],
          low=df['Low'],
          close=df['Close'],
          volume=df['Volume'],
      ).iloc[-12:]
      indicators["vwap"] = {date.strftime('%Y-%m-%d'): int(value) for date, value in vwap_series.to_dict().items()}

      st.sidebar.success(f"Feteched stock price data for: {ticker}")
      return {'stock_price': data.to_dict(orient='records'), 'indicators': indicators}
    except Exception as e:
      st.sidebar.error(f'Error fetching price data for {ticker}: {str(e)}')
      return f"Error fetching price data: {str(e)}"

@tool
def get_financial_metrics(ticker: str) -> Union[Dict, str]:
    """Fetches key financial ratios for a given ticker."""
    # with st.spinner(f'Fetching financial metrics for {ticker}..'):
    try:
      stock = yf.Ticker(ticker)
      info = stock.info
      st.sidebar.success(f"Feteched financial metrics for: {ticker}")
      return {
          'pe_ratio': info.get('forwardPE'),
          'price_to_book': info.get('priceToBook'),
          'debt_to_equity': info.get('debtToEquity'),
          'profit_margins': info.get('profitMargins')
      }

    except Exception as e:
      st.sidebar.error(f'Error fetching ratios: {str(e)}')
      return f"Error fetching ratios: {str(e)}"


class StockState(TypedDict):
    messages: Annotated[list, add_messages]
    stock: str

stock_graph_builder = StateGraph(StockState)

stocktools = [get_stock_prices, get_financial_metrics]
stockllm = ChatOpenAI(model='gpt-4o-mini')
stockllm_with_tool = stockllm.bind_tools(stocktools)

def fundamental_analyst(state: StockState):
    messages = [
        SystemMessage(content=FUNDAMENTAL_ANALYST_PROMPT.format(company=state['stock'])),
    ]  + state['messages']
    return {
        'messages': stockllm_with_tool.invoke(messages)
    }

stock_graph_builder.add_node('fundamental_analyst', fundamental_analyst)
stock_graph_builder.add_edge(START, 'fundamental_analyst')
stock_graph_builder.add_node(ToolNode(stocktools))
stock_graph_builder.add_conditional_edges('fundamental_analyst', tools_condition)
stock_graph_builder.add_edge('tools', 'fundamental_analyst')

# HumanMessage(content=)
# stock_graph_builder.add_edge('fundamental_analyst', END)
stock_graph = stock_graph_builder.compile()






##################################################  APP UI ########################################################################
# st.title("Fin-Report-GPT")
################## START THE APP


# Two-column layout in Streamlit
# intilisation
ticker_name = st.session_state['ticker_name']
selected_company = st.session_state['company_name']
if 'fin_summary_flag' not in st.session_state:
  st.session_state['fin_summary_flag']  = False
if 'ticker_name_sum' not in st.session_state:
  st.session_state['ticker_name_sum'] = ''
# if 'company_name' not in st.session_state:



st.markdown(f"""
    <h1 style="text-align: center; font-size: 32px;">
        Stock Analysis for <span style="color: #39ff14;">{selected_company}</span>
    </h1>
""", unsafe_allow_html=True)



# st.write("This is the selected company name",selected_company)
# col1, col2 = st.columns(2)
# # Display the stream charts
# with col1:# FINANCIAL SUMMARIZER
if np.logical_or(st.session_state['fin_summary_flag'] == False, ticker_name!=st.session_state['ticker_name_sum']):
  events = stock_graph.stream({'messages':[('user', 'Should I buy this stock?')],
                        'stock': ticker_name}, stream_mode='values')
  with st.spinner(f'Analysing Stock Summary for {ticker_name}..'):
    for event in events:
      if 'messages' in event:
        event['messages'][-1].pretty_print()
  st.session_state['event'] =  event

event = st.session_state['event']

# st.sidebar.write("Ticker name:",st.session_state["ticker_name"])


############################# plot metric as tile chart
metric_data = event['messages'][-2].content
metric_dict = json.loads(metric_data)
# Inject custom styles
st.markdown(tile_style, unsafe_allow_html=True)

col11, col12, col13, col14  = st.columns(4)
with col11:
    st.markdown('<div class="tile-header-container">', unsafe_allow_html=True)
    for key, value in metric_dict.items():
        if np.logical_and(key in ['pe_ratio'],value!=None):
          # replace underscore with space and use camelcase
          key = key.replace('_', ' ').title().replace('Pe Ratio', 'PE Ratio')
          tile_html = f"""
              <div class="tile">
                  <div class="tile-header">{key}</div>
                  <div class="tile-value">{value:,.2f}</div>
              </div>
          """
          st.markdown(tile_html, unsafe_allow_html=True)
with col12:
    st.markdown('<div class="tile-header-container">', unsafe_allow_html=True)
    for key, value in metric_dict.items():
        if np.logical_and(key in ['price_to_book'],value!=None):
          # replace underscore with space and use camelcase
          key = key.replace('_', ' ').title().replace('Pe Ratio', 'PE Ratio')
          tile_html = f"""
              <div class="tile">
                  <div class="tile-header">{key}</div>
                  <div class="tile-value">{value:,.2f}</div>
              </div>
          """
          st.markdown(tile_html, unsafe_allow_html=True)

with col13:
    st.markdown('<div class="tile-header-container">', unsafe_allow_html=True)
    for key, value in metric_dict.items():
        if np.logical_and(key in ['debt_to_equity'],value!=None):
          # replace underscore with space and use camelcase
          key = key.replace('_', ' ').title()
          tile_html = f"""
                  <div class="tile">
                      <div class="tile-header">{key}</div>
                      <div class="tile-value">{value:,.2f}</div>
                  </div>
              """
          st.markdown(tile_html, unsafe_allow_html=True)
with col14:
    st.markdown('<div class="tile-header-container">', unsafe_allow_html=True)
    for key, value in metric_dict.items():
        if np.logical_and(key in ['profit_margins'],value!=None):
          # replace underscore with space and use camelcase
          key = key.replace('_', ' ').title()
          if key == 'Profit Margins':
            key = 'Profit Margin'
            value = value * 100
            tile_html = f"""
                <div class="tile">
                    <div class="tile-header">{key}</div>
                    <div class="tile-value">{value:,.0f}%</div>
                </div>
            """
            st.markdown(tile_html, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# with col2:# FINANCIAL SUMMARIZER
###################################### Plotting stock price line chart
data = event['messages'][-3].content
dataframe = convert_to_dataframe(data)
dataframe["Date"] = pd.to_datetime(dataframe["Date"])  # Convert 'Date' to datetime
plot_stock_data(dataframe,ticker_name)


############################## Plotting summary


st.markdown("""
    <h2 style="font-size: 28px;">
        AI Analysis
    </h2>
""", unsafe_allow_html=True)

data_summary = event['messages'][-1].content

# Parse the JSON
try:
    data_summary_dict = json.loads(data_summary.strip("```json\n").strip("```").strip())
except:
  try:
    data_summary_dict = json.loads(data_summary.strip("```json\n  "))
  except json.JSONDecodeError as e:
      st.error(f"Error decoding JSON: {e}")

# drop key 'stock' from dictionary
data_summary_dict.pop('stock', None)

# Inject custom styles
st.markdown(custom_style, unsafe_allow_html=True)


col3, col4 = st.columns(2)
with col3:
    # row11, row12 = st.row(2)
    # with row11:
    generate_summary('financial_analysis',data_summary_dict)
    # with row12:
    generate_summary('price_analysis',data_summary_dict)

    # Display each summary in a styled container
    # for header, content in data_summary_dict.items():
    #   if header in ["financial_analysis","price_analysis"]:
    #     header = header.replace('_', ' ').title().replace('Final Summary', 'Summary')
    #     summary_html = f"""
    #         <div class="summary-container">
    #             <div class="summary-header">{header}</div>
    #             <div class="summary-content">{content}</div>
    #         </div>
    #     """
    #     st.markdown(summary_html, unsafe_allow_html=True)
    # END
    # st.session_state['ticker_name_sum'] = ticker_name
    # st.session_state['fin_summary_flag']  = True

with col4:
  data_summary_dict['Summary'] = data_summary_dict['final Summary'] + "<br>" +  "<b>AI Recommendation</b>" + "<br>" +  data_summary_dict['Asked Question Answer']
                              # + "<br><br>"
                              # + "<i>Disclaimer: <span style='font-size: 10px;'>The following recommendations are generated by an AI. Please ensure thorough due diligence is done before making any buying or selling decisions related to stocks.</span></i>"


  # row21, row22 = st.row(2)
  # with row21:
  generate_summary('technical_analysis',data_summary_dict)
  # with row22:
  generate_summary('Summary',data_summary_dict)

  # END
  st.session_state['ticker_name_sum'] = ticker_name
  st.session_state['fin_summary_flag']  = True