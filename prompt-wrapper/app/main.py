import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from app.chains import Chain

def create_streamlit_app(llm):
    st.title("Job Posting Summarizer")
    url_input = st.text_input("Enter a URL:", value="")
    submit_button = st.button("Submit")

    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = loader.load().pop().page_content
            jobs = llm.extract_data(data)
            for job in jobs:
                result = llm.define_response(job)
                st.markdown(result)
        except Exception as e:
            st.error(f"{e}")


if __name__ == "__main__":
    chain = Chain()
    st.set_page_config(layout="wide", page_title="Job Posting Summarizer")
    create_streamlit_app(chain)