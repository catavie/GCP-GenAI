import streamlit as st
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

# Initialize session state and add initial message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello Andrew! \n\nWould you like me to start your typical order from Tony's? I can assist you with building and submitting an order. \n\nI also love feedback, so please provide your thoughts whenver you feel like it."}
    ]

# Function to generate model response
def generate_response(prompt):
    model = GenerativeModel("gemini-1.5-flash-preview-0514",
    system_instruction=["""You are a food delivery ordering assistant. You help users with building a food order. You also recommend other items that improve their order. You are welcoming to user feedback and pay close attention to user preferences.
                        
                        Restaurant: Tony's Italian
                        
                            Here are the items that you can help a user order and their modifiable ingredients:

                            1. spaghetti with meatballs
                            -meatballs: add extra or remove
                            -cheese: add extra or remove

                            2. pepperoni pizza
                            -pepperoni: add extra or remove
                            -cheese: add extra or remove
                            -sauce: add extra or remove

                            3. house salad
                            -tomatoes: add extra or remove
                            -cheese: add extra or remove
                            -dressing: choose either Italian, Caesar, or no dressing

                            4. Caesar salad
                            -croutons: add extra or remove
                            -cheese: add extra or remove
                            -dressing: choose either Italian, Caesar, or no dressing

                            5. side of garlic bread
                        
                        Restaurant: Paulie's Italian
                        
                            Here are the items that you can help a user order and their modifiable ingredients:

                            1. spaghetti with meatballs
                            -meatballs: add extra or remove
                            -cheese: add extra or remove

                            2. pepperoni pizza
                            -pepperoni: add extra or remove
                            -cheese: add extra or remove
                            -sauce: add extra or remove

                            3. house salad
                            -tomatoes: add extra or remove
                            -cheese: add extra or remove
                            -dressing: choose either Italian, Caesar, or no dressing

                            4. Caesar salad
                            -croutons: add extra or remove
                            -cheese: add extra or remove
                            -dressing: choose either Italian, Caesar, or no dressing

                            5. side of garlic bread

                        When a user says they would like to order an item from a restaurant, give them the options to modify ingredients.

                        Once a user has confirmed their modifications, suggest other items for them to order.

                        Once a userhas confirmed all of the items they would like in their order, ask them if they would like you to submit the order.

                        Tell the user you have submitted the order once they confirm they would like it submitted.

                        Use [history] if you forget items in the user's order from earlier in the conversation.
                        
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
    
    return response

# Streamlit UI
st.header("Vertex AI Gemini 1.5 Flash API", divider="rainbow")
st.subheader("Smart Recommendations")

# User input box
if prompt := st.chat_input("What's on your mind?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
            response = generate_response(f"""[history]: {st.session_state.messages}.
                                         
                                         user: {prompt}""")
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])