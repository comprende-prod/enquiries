import os
from dataclasses import asdict
from pathlib import Path

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import DataFrameLoader

import streamlit as st
import pandas as pd
import pyperclip

from trademe import make_url, search


# Big thanks to this article, really illuminated the data -> QA pipeline:
# https://python.langchain.com/docs/use_cases/question_answering/


# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

# Functions
@st.cache_resource  # Could not cache_data on listings, unserialisable
def get_listings():
    url = make_url("rent", "", "", "", search_string="comprende")
    listings = search(None, ["--headless=new", "--start-maximized"], url)
    return listings


@st.cache_resource  # Could not cache_data on qa_chain, unserialisable
def build_chain(one_column_dataframe, column_name):
    # Starting with the most abstracted LangcChain pipeline, where you just
    # feed a loader to VectorStoreIndexCreator().
    # See the article above for an explanation on this.
    # If we're not happy with this, we can try using progressively less 
    # abstraction to adjust things for our use case.
    loader = DataFrameLoader(
        one_column_dataframe, 
        page_content_column=column_name
    )
    index = VectorstoreIndexCreator().from_loaders([loader])
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=index.vectorstore.as_retriever()
    )
    return qa_chain


@st.cache_data
def respond(_qa_chain, system_message, tenant_email):
    # Unfortunately, we have to stitch together the system message and user 
    # input pretty cruedly:
    # https://github.com/langchain-ai/langchain/issues/10011
    
    # ^ CAN TRY SPECIFYING A PROMPTTEMPLATE INSTEAD
    input = system_message + "\n\n" + tenant_email
    return _qa_chain.run(input)


# Get data:
listings = get_listings()
one_column_dataframe = pd.DataFrame(listings, columns=["rental_properties"])

# Get system message:
app = Path("app")
with open(app / "system_message.md", "r") as f:
    system_message = f.read()

# Chain:
qa_chain = build_chain(one_column_dataframe, "rental_properties")


# UI:
# Title
st.title("Enquiries")
# Data
display_df = pd.DataFrame([asdict(l) for l in listings])
st.dataframe(display_df, hide_index=True) 
# Tenant email
tenant_email = st.text_area(label="Tenant email")

# LLM response
llm_response = respond(qa_chain, system_message, tenant_email)
st.write(llm_response)

# Copy button
st.button("Copy", on_click=pyperclip.copy(llm_response))

