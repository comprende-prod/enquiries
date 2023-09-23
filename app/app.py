import os
from dataclasses import asdict
from pathlib import Path

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


# Setup:
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
app = Path("app")
with open(app / "system_message.md", "r") as f:
    system_message = f.read()


@st.cache_resource
def get_listings():
    url = make_url("rent", "", "", "", search_string="comprende")
    return search(None, ["--headless=new", "--start-maximized"], url)


@st.cache_resource
def qa_dataframe(listings, column_name) -> pd.DataFrame:
    """Creates one-column DataFrame; used for building QA chain.
    
    Slightly awkward: seems like most convenient method is to convert 
    listings to a one-column DataFrame.
    """
    listing_strings = [str(asdict(listing)) for listing in listings]
    return pd.DataFrame(listing_strings, columns=[column_name])


#@st.cache_data
def build_chain(one_column_dataframe, column_name):
    """Build RetrievalQA chain.

    Uses basically the most high-level/abstracted data -> QA pipeline possible.

    Args:
        dataframe: one-column pandas DataFrame (see get_listings_dataframe()).
        column_name: The name of the column used to build the loader. Note this
            can only be one column at a time.
        
    Returns:
        RetrievalQA chain. Can be queried with .run().
    """
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
    # Can't specify SystemMessage the normal way here, to my knowledge:
    input = system_message + "\n\n" + "Tenant email: " + tenant_email
    return _qa_chain.run(input)


# Set up listings:
listings = get_listings()
# - For display:
display_df = pd.DataFrame([asdict(listing) for listing in listings])
display_df.drop("address", axis=1, inplace=True)
display_df.rename({"title": "address"}, axis=1, inplace=True)
# - For QA chain:
qa_df = qa_dataframe(listings, "rental_properties")


# Set up QA chain:
qa_chain = build_chain(qa_df, "rental_properties")


# UI:
st.subheader("Enquiries")
st.dataframe(display_df, hide_index=True)
tenant_email = st.text_area("Tenant email")


# LLM response:
if tenant_email:
    # Generate:
    llm_response = respond(qa_chain, system_message, tenant_email)
    st.write(llm_response)
    # Copy button:
    st.button("Copy", on_click=pyperclip.copy(llm_response))

