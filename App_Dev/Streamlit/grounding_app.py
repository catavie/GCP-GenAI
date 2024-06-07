import streamlit as st
import base64
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
from google.cloud import storage

# Initialize session state and add initial message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello Andrew! \n\nWould you like me to start your typical order from Tony's? I can assist you with building and submitting an order. \n\nI also love feedback, so please provide your thoughts whenver you feel like it. \n\nFor example, you can tell me if any of my recommendations are not helpful. Or even better, you can tell me when they are helpful."}
    ]

def display_image_from_gcs(bucket_name, blob_name):
    """Fetches and displays an image from GCS using a base64 encoded data URL."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    image_bytes = blob.download_as_bytes()
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    # Return the image in Markdown format for seamless integration with the chat message
    return f'<img src="{f"data:image/png;base64,{encoded_image}"}" width="400">'


# Function to generate model response
def generate_response(prompt: str):
    model = GenerativeModel("gemini-1.5-flash-preview-0514",
    system_instruction=["""
                        
                        """])

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    chat = model.start_chat()

    response = (chat.send_message(
      [f"""{prompt}"""],
      safety_settings=safety_settings
  )).text
    
    if "picture" in prompt.lower(): # FIX
        image_markdown = display_image_from_gcs('andrewcooley-genai-tests', 'retrieve_images/image.png')
        response += f"\n\n<div style='text-align: center;'>{image_markdown}</div>"

    return response

# Streamlit UI
st.header("Vertex AI Gemini 1.5 Flash API", divider="rainbow")
st.subheader("Smart Recommendations")

# User input box
prompt = st.chat_input("I really want...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
            
            text_messages = [msg for msg in st.session_state.messages if not "<div styl" in msg["content"]]
            history = "\n\n".join(f"{message['role']}: {message['content']}" for message in text_messages)

            response = generate_response(f"""[history]: {history}.
                                         
                                         user: {prompt}""")

            
            st.session_state.messages.append({"role": "assistant", "content": response})

            

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)