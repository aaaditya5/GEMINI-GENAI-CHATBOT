import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

## Langchain tracking monitoring our model
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a helpful assistant. Please response to the user query"),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    os.environ['GOOGLE_API_KEY'] = api_key
    llm = ChatGoogleGenerativeAI(model=llm, temperature=temperature)
    user_prompt = prompt.format(question=question)
    response = llm.invoke(user_prompt)
    return response.content

## Now we will create the web app
st.title("Enhanced Q&A chatbot with GEMINIPRO")

with st.sidebar:
    st.write("Settings")
    api_key = st.text_input("Enter your Google API key to proceed", type='password')
    # Now we will create the dropdown for various Gemini Pro models
    llm = st.selectbox("Select a Gemini Pro model", ["gemini-pro","gemini-1.0-pro", "gemini-1.0-pro-001", "gemini-1.0-pro-latest"])

    # Adjust the response parameters
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)
    max_tokens = st.slider("Max Tokens", min_value=50, max_value=500, value=125)

# Main interface for the user
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    res = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(res)
else:
    st.write("Please provide the query")