import streamlit as st
from vertexai.preview import reasoning_engines

# Initialize conversation history as a Streamlit session state
if "history" not in st.session_state:
    st.session_state.history = []

# Vertex AI setup (unchanged)
remote_agent = reasoning_engines.ReasoningEngine(
    "projects/619758184732/locations/us-central1/reasoningEngines/6483248322948628480"
)

def query_agent(prompt: str):
    response = remote_agent.query(
        input=f"{prompt}",
        config={"configurable": {"session_id": "demo_01"}},
    )
    return response

def extract_content(data):
    contents = []
    for message in data['history']:
        if 'kwargs' in message and 'content' in message['kwargs']:
            contents.append(message['kwargs']['content'])
    return contents

# Streamlit UI
st.header("Vertex AI Gemini 1.5 Flash API", divider="rainbow")
st.subheader("Smart Recommendations")

# Get user input
if prompt := st.chat_input("I want to watch..."):

    # Query Vertex AI
    response = query_agent(prompt)
    contents = extract_content(response)
    for index, content in enumerate(contents):
        if index % 2 == 0:
            st.markdown(f"**User:** {content}")
        else:
            st.markdown(f"**Assistant:** {content}")
    st.markdown(f"**User:** {prompt}")
    st.markdown(f"**Assistant:** {response['output']}")