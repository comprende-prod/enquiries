

from dataclasses import asdict
import streamlit as st
import pandas as pd
from trademe import make_url, search


st.title("Enquiries")


@st.cache_data
def listing_dataframe():
    """Build DataFrame of listings."""
    # Get listings:
    url = make_url("rent", "", "", "", search_string="comprende")
    listings = search(urls=url)

    # REMEMBER THAT THIS NEEDS TO BE A ONE-COLUMN DF!

    # Convert to list[dict], then create DataFrame
    listing_dicts = [asdict(l) for l in listings]

    return pd.DataFrame(listing_dicts)


@st.cache_data
def vsi():
    """Create VectorStoreIndex."""



