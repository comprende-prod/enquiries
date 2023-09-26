import os
import subprocess
from dataclasses import asdict
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import streamlit as st
import pandas as pd


# Install trademe:
# try:
#     from trademe import search, make_url
# except ModuleNotFoundError:
#     token = st.secrets["token"]
#     #subprocess.Popen([f'{sys.executable} -m pip install git+https://${token}@github.com/comprende-prod/trademe.git'], shell=True)
#     subprocess.Popen([f'{sys.executable} -m pip install git+https://github.com/comprende-prod/trademe.git'], shell=True)
#     time.sleep(90)

# from trademe import search, make_url


#subprocess.check_call(["git", "clone", "https://github.com/comprende-prod/trademe.git"])
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


def prepare_data(listings) -> pd.DataFrame:
    if len(listings) == 0: raise ValueError("Length of `listings` must be >0.")
    data = pd.DataFrame([asdict(listing) for listing in listings])
    data.drop("address", axis=1, inplace=True)
    data["availability"] = data["availability"].str.replace(":", "", regex=False)
    data.rename({"title": "address"}, axis=1, inplace=True)
    data.insert(0, "selected", False)
    return data


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

