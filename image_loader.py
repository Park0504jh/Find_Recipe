import json
import streamlit as st

@st.cache_data
def load_image_dict():
    path = r"C:\Users\DS\Desktop\recipe1M_layers\layer2.json"
    with open(path, "r", encoding="utf-8") as f:
        layer2 = json.load(f)
    image_dict = {}
    for item in layer2:
        images = item.get("images", [])
        if images:
            image_dict[item["id"]] = images[0]["url"]
    return image_dict
# streamlit run app.py