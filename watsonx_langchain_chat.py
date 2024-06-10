# https://python.langchain.com/v0.2/docs/tutorials/chatbot/
# Import relevant functionality from LangChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# General Python utility modules
import os
from dotenv import load_dotenv
import secrets

# watsonx LLM specific modules
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials

# Streamlit modules
import streamlit as st

# These global variables will be updated in get_credentials() functions
watsonx_project_id = ""
api_key = ""
url=""
locale=""
watsonx_client=""




# Read creadentials from local .env file in the same directory as this script
def get_credentials():

    load_dotenv()

    # Update the global variables that will be used for authentication in another function
    globals()["api_key"] = os.getenv("api_key", None)
    globals()["watsonx_project_id"] = os.getenv("project_id", None)
    globals()["url"] = os.getenv("url", None)
    globals()["locale"] = os.getenv("locale", None)

    globals()["gcounter"] = 0
    
    #print("*** Got credentials***")

# Get watsonx.ai Python client using defined credentials
def get_watsonx_ai_client():
    credentials = Credentials(
        url = url,
        api_key = api_key
    )
    client = APIClient(credentials)
    return client

# List all available and supported Foundation Models in watsonx.ai
def list_models():
    modelList = watsonx_client.foundation_models.TextModels
    models = [member.value for member in modelList]
    return models

# Return WatsonxLLM model object with the specific parameters
def get_model(model_type,max_tokens,min_tokens,decoding,stop_sequences,temperature):

    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.STOP_SEQUENCES:stop_sequences,
        GenParams.TEMPERATURE: temperature
    }

    model = WatsonxLLM(
        model_id=model_type,
        params=generate_params,
        url=url,
        apikey=api_key,
        project_id=watsonx_project_id
    )      
    return model

# Filter the number of messages tracked in the message history and included
# in the prompt for the chat interaction to avoid running out of the context
# length of the LLM
# This would return the last k most recent messages
def filter_messages(messages, k=10):
    return messages[-k:]


# Track Session History
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = ChatMessageHistory()
    return st.session_state.chat_store[session_id]

# Setup sidebar for the streamlit app with the list of watsonx.ai models to choose from
def setupSideBar():
    models = list_models()
    selected_model = st.sidebar.selectbox("Select model",models)
    return selected_model

# Create a unique id to track conversation
def create_unique_id():
    """
    Creates a cryptographically-secure, URL-safe string
    """
    return secrets.token_urlsafe(16)  

# Main function
def main():
    gcounter = 0
    # Initialize chat history
    if "str_messages" not in st.session_state:
        st.session_state.str_messages = []
    
    if "chat_store" not in st.session_state:
        st.session_state.chat_store = {}

    if "session_id" not in st.session_state:
        st.session_state.session_id = create_unique_id()

    # Set the api key and project id global variables
    get_credentials()

    # Web app UI - title and input box for the question
    #st.title(st.image('watsonx_main.jpg') + 'watsonx.ai Assistant')

    col1, col2 = st.columns([1,20])
    with col1:
        st.image('watsonx_logo.png', width=30)
    with col2:
        st.markdown("##### *Chat* with ***watsonx.ai***")

    
    # Get list of supported models in watsonx.ai
    globals()['watsonx_client'] = get_watsonx_ai_client()

    ###llm="meta-llama/llama-3-70b-instruct"
    model_type = setupSideBar()
    max_tokens = 400
    min_tokens = 20
    decoding = DecodingMethods.GREEDY
    stop_sequences = ['.', '\n']
    temperature = 0.7
    llm_model = get_model(model_type,max_tokens,min_tokens,decoding,stop_sequences,temperature)
    
    config = {"configurable": {"session_id": st.session_state.session_id}}
    

    prompt = ChatPromptTemplate.from_messages(
        [
            (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability. Do not provide additional details.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    #chain = prompt | llm_model
    chain = (
        RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
        | prompt
        | llm_model
    )

    with_message_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")

    # Display chat messages from history on app rerun
    for str_message in st.session_state.str_messages:
        if str_message["role"] == "assistant":
            with st.chat_message(str_message["role"],avatar="watsonx.jpg"):
                st.markdown(str_message["content"])
        else:
            with st.chat_message(str_message["role"]):
                st.markdown(str_message["content"])

    

    # React to user input
    if user_question := st.chat_input('What would you like to chat about?'):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_question)
        # Add user message to chat history
        st.session_state.str_messages.append({"role": "user", "content": user_question})

        response = with_message_history.invoke(
            {"messages": [HumanMessage(content=user_question)]},
            config=config,
        )
        

        with st.chat_message("assistant",avatar="watsonx.jpg"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.str_messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()