# [2412]
#

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

import streamlit as st
import numpy as np

import json

st.title("CCC Bot : 2501")

succinct_txt = ("Please provide only a one or two word answer. " +
                "Be as succinct as possible when answering. ")
initial_question = "What shall we discuss?"


with st.sidebar:
    request_succinct = st.checkbox("Request succinct")
    st.text('The Request Succinct option appends this text to the prompt : "' +
             succinct_txt + '"')
    if st.button("Save Transcript"):
        print("save transcript")
        messages = []
        for i in range(len(st.session_state.messages)):
            message = st.session_state.messages[i].copy()
            j = int(i/2)
            message["succinct"] = st.session_state.succinct_hist[j]
            print(message)
            messages.append(message)
        with open('ccc_bot-transcript.json', 'w') as file:
            json.dump(messages, file)

    tab1, tab2 = st.tabs(["Example Questions", "Terms & Privacy"])

    with tab1:
        st.header("Example Questions")
        st.text("• How many districts are there in the California community college system?")
        st.text("• What college is designated a Center of Excellence in bioprocessing?")
        st.text("• How many California community colleges partner with the California Department of Corrections and Rehabilitation (CDCR) to provide in‑person courses?")
        st.text("• ... ")
    with tab2:
        st.header("Terms & Privacy")
        st.text("Add a terms and privacy statement here, note sessions are logged for testing and development.")


st.text("Example question : What is the California community college with the largest student enrollment?")
st.text("Note : make sure the Ollama client is running while using this application.")

st.divider()


llm = OllamaLLM(model="phi3")
# llm = OllamaLLM(model="phi3.5")

template = """
Question: {question}
Answer: 
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

chain = prompt | llm


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.succinct_hist = []

# displays the chat history when app is rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(initial_question):

    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if request_succinct:
            prompt += " " + succinct_txt
            print(prompt)
            st.session_state.succinct_hist.append("True")
        else:
            print(prompt)
            st.session_state.succinct_hist.append("False")
        result = chain.invoke(prompt)

        st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})


# [\] add succinct setting to transcript
# [x] json formatted transcript
# [ ] data collection on impact of succinct option (word counts with and without)
# [ ] other options for the sidebar / chat session
#
