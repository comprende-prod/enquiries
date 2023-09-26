import os
from dataclasses import asdict
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import streamlit as st
from trademe import search, make_url


os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]


def get_listings():
    url = make_url("rent", "", "", "", search_string="comprende")
    return search(None, ["--headless=new", "--start-maximized"], url)


def dataframe_to_string(data):
    """Make bullet-pointed string from listing DataFrame."""
    if len(data) == 0: raise ValueError("Data must have 1+ rows.")

    data.drop(["selected", "agency"], axis=1, inplace=True)

    output = ""
    for _, row in data.iterrows():
        output += f"{row.pop('address')} \n"
        for index, value in row.items():
            output += f" - {index.capitalize()}: {value}\n" 
        output += "\n"
    return output


def build_ensemble_retriever(
        listings, 
        bm25_k, 
        score_threshold,
        embeddings=OpenAIEmbeddings()
    ):
    """Create EnsembleRetriever using BM25 and FAISS retriever."""
    # Build BM25Retriever:
    listing_strings = [str(asdict(listing)) for listing in listings]
    bm25 = BM25Retriever.from_texts(listing_strings) 
    bm25.k = bm25_k
    # Build FAISS retriever:
    faiss_db = FAISS.from_texts(listing_strings, embeddings)
    faiss_retriever = faiss_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": score_threshold}
    )
    # Create EnsembleRetriever: (50/50 weights by default)
    return EnsembleRetriever(retrievers=[bm25, faiss_retriever])  
