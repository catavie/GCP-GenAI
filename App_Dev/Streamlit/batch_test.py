import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
import time

# Vertex AI Setup (Replace placeholders with your actual values)
PROJECT_ID = "andrewcooley-test-project" 
LOCATION = "us-central1"

# Streamlit UI
st.title("Gemini Text - Batch Inference App")
st.text("Safety Filters off")
options = ["gemini-1.5-flash-preview-0514", "gemini-1.5-pro-preview-0514", "gemini-1.0-pro-preview-001"]
selected_model = st.selectbox("Select an option:", options)
if selected_model == "gemini-1.0-pro-preview-001":
    pass
else:
    system_instruction = st.text_area("System Instruction:", value="Be extremely friendly.")
temp_slider = st.slider("Temperature:", min_value=0., max_value=2., value=0.5)
if selected_model == "gemini-1.0-pro-preview-001":
    top_k_slider = st.slider("Top-K:", min_value=0, max_value=40, value=40)
else:
    pass
top_p_slider = st.slider("Top-P:", min_value=0., max_value=1., value=0.95)
max_output_tokens_slider = st.slider("Output token limit:", min_value=1, max_value=8192, value=2048)
num_inputs = st.number_input("Number of Text Inputs:", min_value=1, value=3)

text_inputs = []
for i in range(num_inputs):
    text_inputs.append(st.text_area(f"Input #{i+1}:"))

if st.button("Generate Inferences"):
    with st.spinner("Generating inferences..."):
        results = []

        # Gemini API Interaction
        vertexai.init(project="andrewcooley-test-project", location="us-central1")

        if selected_model == "gemini-1.0-pro-preview-001":

            generation_config = {
            "max_output_tokens": max_output_tokens_slider,
            "temperature": temp_slider,
            "top_k": top_k_slider,
            "top_p": top_p_slider,
            "candidate_count": 1
            }

        else:

            generation_config = {
            "max_output_tokens": max_output_tokens_slider,
            "temperature": temp_slider,
            "top_p": top_p_slider,
            "candidate_count": 1
            }
            

        safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }


        for text in text_inputs:
            if selected_model == "gemini-1.0-pro-preview-001":
                model = GenerativeModel(model_name=selected_model)
            else:
                model = GenerativeModel(model_name=selected_model, system_instruction=[system_instruction])
            responses = model.generate_content([text],
            generation_config=generation_config,
            safety_settings=safety_settings)
            results.append(responses.text)
            time.sleep(5)


    st.subheader("Results:")
    for i, (input_text, inference) in enumerate(zip(text_inputs, results)):
        st.write(f"**Input #{i+1}:**")
        st.write(input_text)
        st.write(f"**Inference:**")
        st.write(inference)
