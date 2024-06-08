import os
import streamlit as st
from dotenv import load_dotenv
# watsonx.ai python SDK
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai.foundation_models import ModelInference

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials


DISPLAY_MODEL_LLAMA2 = "llama2"
DISPLAY_MODEL_GRANITE= "granite"
DISPLAY_MODEL_FLAN = "flan"
DISPLAY_MODEL_ELYZA = "elyza"

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

    print("*** Got credentials***")

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

# The get_model function creates an LLM model object with the specified parameters
def get_model(model_type,max_tokens,min_tokens,decoding,stop_sequences):

    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.STOP_SEQUENCES:stop_sequences
    }
    
    model = ModelInference(
        model_id = model_type,
        params=generate_params,
        credentials=Credentials(
                   url = url,
                   api_key = api_key
                  ),
        project_id=watsonx_project_id,
    )
    
    return model

# Create a generic prompt to answer the user question
def get_prompt(question, selected_model):

    # Prompts are passed to LLMs as one string. We are building it out as separate strings for ease of understanding
    # Instruction
    #instruction = "Follow examples and answer the question briefly."
    instruction = "You are a helpful AI assistant. Answer the question below. " 
    # Examples to help the model set the context
    ##examples = "\n\nQuestion: What is the capital of Germany\nAnswer: Berlin\n\nQuestion: What year was George Washington born?\nAnswer: 1732\n\nQuestion: What are the main micro nutrients in food?\nAnswer: Protein, carbohydrates, and fat\n\nQuestion: What language is spoken in Brazil?\nAnswer: Portuguese \n\nQuestion: "
    examples = ""
    # Question entered in the UI
    your_prompt = question
    # Since LLMs want to "complete a document", we're are giving it a "pattern to complete" - provide the answer
    # This format works for all models with the exception of llama
    end_prompt = "\nAnswer:"

    final_prompt = instruction + examples + your_prompt + end_prompt

    return final_prompt


def answer_questions(user_question, selected_model):

    # Get the prompt
    final_prompt = get_prompt(user_question, selected_model)
    
    # Display our complete prompt - for debugging/understanding
    print("***final prompt***")
    print(final_prompt)
    print("***end of final prompt***")

    # Look up parameters in documentation:
    # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#
    model_type = selected_model
    max_tokens = 300
    min_tokens = 50
    decoding = DecodingMethods.GREEDY
    stop_sequences = ['.', '\n']

    # Get the model
    model = get_model(model_type, max_tokens, min_tokens, decoding, stop_sequences)

    # Generate response
    generated_response = model.generate(prompt=final_prompt)
    model_output = generated_response['results'][0]['generated_text']
    # For debugging
    print("Answer: " + model_output)

    return model_output

# Setup sidebar for the streamlit app with the list of watsonx.ai models to choose from
def setupSideBar():
    models = list_models()
    selected_model = st.sidebar.selectbox("Select model",models)

    return selected_model

# Main function for the application
def main():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Set the api key and project id global variables
    get_credentials()

    # Web app UI - title and input box for the question
    st.title('🌠watsonx.ai Assistant')

    # Get list of supported models in watsonx.ai
    globals()['watsonx_client'] = get_watsonx_ai_client()
    
    llm = setupSideBar()
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if user_question := st.chat_input('Ask a question, for example: What is IBM?'):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})

        #response = f"Echo: {prompt}"
        response = answer_questions(user_question,llm)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()