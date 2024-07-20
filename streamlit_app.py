import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import ChatMessage


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# Show title and description.
st.title("Reviuer Assistant")

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

# Create a session state variable to store the chat messages. This ensures that the
# messages persist across reruns.
if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

# Display the existing chat messages via `st.chat_message`.
for message in st.session_state.messages:
    with st.chat_message(message.role):
        st.markdown(message.content)

# Create a chat input field to allow the user to enter a message. This will display
# automatically at the bottom of the page.
if prompt := st.chat_input("What is up?"):

    # Store and display the current prompt.
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream the response to the chat using `st.write_stream`, then store it in 
    # session state.
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        perplexity = ChatPerplexity(temperature=0, pplx_api_key=st.secrets["PERPLEXITY_API_KEY"], streaming=True, callbacks=[stream_handler])
        response = perplexity.invoke([ChatMessage(role="system", content="You are a good agent")] + st.session_state.messages[1:])
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
