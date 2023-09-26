import os
from dataclasses import asdict
from pathlib import Path
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import streamlit as st
import pyperclip
from helpers import build_ensemble_retriever, get_listings, \
    dataframe_to_string, prepare_data
from constants import EMAIL_TEMPLATE, PROMPT_TEMPLATE, SEARCH_SYSTEM_MESSAGE


BM25_K = 7                           # k for BM25Retriever
SIMILARITY_SCORE_THRESHOLD = 0.6     # Similarity threshold for FAISS retriever
LLM = ChatOpenAI(model="gpt-3.5-turbo")
os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]


# Functions:
get_listings = st.cache_resource(get_listings)

build_ensemble_retriever = st.cache_resource(build_ensemble_retriever)

dataframe_to_string = st.cache_data(dataframe_to_string)

prepare_data = st.cache_data(prepare_data)


# Listings + data for display
listings = get_listings()
data = prepare_data(listings)

# Stuff for similarity search:
retriever = build_ensemble_retriever(
    listings,
    BM25_K,
    SIMILARITY_SCORE_THRESHOLD
)
qa_chain = load_qa_chain(llm=LLM)


# LLMChain:
llmchain = LLMChain.from_string(template=PROMPT_TEMPLATE, llm=LLM)


# UI:
st.subheader("Enquiries")
user_facing_data = st.data_editor(data, hide_index=True)
tenant_email = st.text_area("Tenant email")
gpt_with_search, gpt_only, manual_tab = st.tabs(
    ["GPT with search", "GPT only", "Select properties"]
)
gpt_with_search.write(
    "*Generates email based on similarity search.\n Ideal for larger \
    searches, e.g. if searching from 100+ listings.*"
)
gpt_only.write(
    "*Directly feeds data to LLM, without the use of an intermediate \
    similarity search. Recommended for smaller datasets.*"
)
manual_tab.write(
    "*Generate an email from a static template by selecting properties.*"
)


# Tabs:

# Manual selection tab:
with manual_tab:
    generate_manual = st.button("Generate", key="generate_manual")
    if generate_manual:
        selected_properties = \
            user_facing_data.loc[user_facing_data["selected"] == True]
        manual_response = EMAIL_TEMPLATE.replace(
            "INSERT_PROPERTIES", 
            dataframe_to_string(selected_properties)
        )
        st.write(manual_response)
        st.button(
            "Copy", 
            key="manual_copy", 
            on_click=pyperclip.copy(manual_response)
        )

if tenant_email:
    # GPT with search tab:
    with gpt_with_search:
        run_with_search = st.button("Generate", key="generate_with_search")
        if run_with_search:
            docs = retriever.get_relevant_documents(query=tenant_email)
            with_search_response = qa_chain.run(
                input_documents=docs,
                question=SEARCH_SYSTEM_MESSAGE + "\n\n" + "Tenant email: " + tenant_email
            )
            st.write(with_search_response)
            st.button(
                "Copy", 
                key="gpt_with_search_copy", 
                on_click=pyperclip.copy(with_search_response)
            )

    # GPT only tab:
    with gpt_only:
        run_gpt_only = st.button("Generate", key="generate_gpt_only")
        if run_gpt_only:
            gpt_only_response = llmchain.run(
                {"tenant_email": tenant_email, 
                 "properties": dataframe_to_string(data)}
            )
            st.write(gpt_only_response)
            st.button(
                "Copy",
                key="gpt_only_copy",
                on_click=pyperclip.copy(gpt_only_response)
            )
            
