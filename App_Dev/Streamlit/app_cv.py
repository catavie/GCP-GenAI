import os
import re

import streamlit as st
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
from vertexai.preview.vision_models import ImageGenerationModel
from typing import Sequence

import google.cloud.texttospeech as tts

PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
print(f"load project {PROJECT_ID}")
vertexai.init(project=PROJECT_ID, location=LOCATION)

#output_image_file = "my-output.png"
generated_story="In a land where the sun shimmered like molten gold and the stars danced in the night sky, lived a young girl named Zarina. She was known for her bright eyes that sparkled with curiosity and her laughter that echoed like the tinkling of wind chimes. But Zarina harbored a secret dream: to see the world beyond her village, to experience the magic that whispered in the wind and the wonders hidden in the depths of the forest."

@st.cache_resource
def load_models():
    """
    Load the generative models for text and multimodal generation.

    Returns:
        Tuple: A tuple containing the text model and multimodal model.
    """
    text_model_pro = GenerativeModel("gemini-1.5-flash-001")
    multimodal_model_pro = GenerativeModel("gemini-1.0-pro-vision")
    imagen_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
    return text_model_pro, imagen_model, multimodal_model_pro


def get_gemini_pro_text_response(
    model: GenerativeModel,
    contents: str,
    generation_config: GenerationConfig,
    stream: bool = True,
):
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    responses = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )

    final_response = []
    for response in responses:
        try:
            # st.write(response.text)
            final_response.append(response.text)
        except IndexError:
            # st.write(response)
            final_response.append("")
            continue
    return " ".join(final_response)

def get_gemini_imagen_image(
    model: ImageGenerationModel,
    prompt: str,
    seed: int,
    number_of_images: int
):
    prompt = f"""Illustration of a mythical story {prompt}"""

    images = model.generate_images(
        prompt=prompt,
        number_of_images=number_of_images,
        aspect_ratio="1:1",
        safety_filter_level="block_few",
        person_generation="allow_adult",
        add_watermark=False,
    )
    # print(f"Image Type: {type(images)}")
    # print(f"Image detail: {dir(images)}")
    
    return images


def get_gemini_pro_vision_response(
    model, prompt_list, generation_config={}, stream: bool = True
):
    generation_config = {"temperature": 0.1, "max_output_tokens": 2048}
    responses = model.generate_content(
        prompt_list, generation_config=generation_config, stream=stream
    )
    final_response = []
    for response in responses:
        try:
            final_response.append(response.text)
        except IndexError:
            pass
    return "".join(final_response)


def text_to_wav(voice_name: str, text: str):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = tts.SynthesisInput(text=text)
    voice_params = tts.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

    client = tts.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input,
        voice=voice_params,
        audio_config=audio_config,
    )

    return response.audio_content

def extract_chapters(text):
  """Extracts text for each chapter from a given text, including any text before
  chapter 1 and after the last chapter.

  Args:
    text: The text to parse.

  Returns:
    A list of strings, where each element represents the text for a chapter,
    including any text before chapter 1 and the last chapter.
  """
  chapters = re.split(r'Chapter \d+: ', text)  # Split by chapter headings
  chapter_texts = []

  # Add text before chapter 1
    # Add text before chapter 1
  if chapters:  # If there's at least one chapter
    chapter_texts.append(chapters[0].strip())  # First element is before chapter 1
    del chapters[0]  # Remove it from the list

  # Extract chapter content and add them to the list
  for chapter in chapters:
    chapter = chapter.replace('**', '') 
    chapter_texts.append(chapter.strip())

  return chapter_texts

st.header("1001 Night", divider="rainbow")
text_model_pro, imagen_model, multimodal_model_pro = load_models()

st.write("Imaginary StoryTeller powered by Gemini")
st.subheader("Generate a story")

# Story premise
creative_control = st.radio(
    "Select the creativity level: \n\n",
    ["Low", "High"],
    key="creative_control",
    horizontal=True,
)
length_of_story = st.radio(
    "Select the length of the story: \n\n",
    ["Short", "Long"],
    key="length_of_story",
    horizontal=True,
)

story_premise = "Arabian Nights folkstory style"

if creative_control == "Low":
    temperature = 0.30
else:
    temperature = 0.95

max_output_tokens = 2048

prompt = f"""You are a master storyteller, 
Write a {length_of_story} story based on the following premise: \n
inspired by the tales of "One Thousand and One Nights," filled with magic, adventure, and a dash of romance. \n
The story is intended for children from 5 - 13 years old.
If the story is "short", then make sure to have 2 chapters or else if it is "long" then 8 chapters.
Important point is that each chapters should be generated based on the premise given above.
First start by giving the book introduction, chapter introductions and then each chapter. It should also have a proper ending.
The book should have prologue and epilogue.
"""

config = {
    "temperature": 0.8,
    "max_output_tokens": 1024,
}

generate_t2t = st.button("Get my story", key="generate_t2t")

if generate_t2t and prompt:
    # st.write(prompt)
    with st.spinner("Generating your story using Gemini ..."):
        # st.write(generated_story)
        generated_story = get_gemini_pro_text_response(
                text_model_pro,
                prompt,
                generation_config=config,
            )
        if generated_story:
            # st.write("Your story:")
            # st.write(response)
            chapter_texts = extract_chapters(generated_story)

            # Loop through the chapter_texts list
            for i, chapter in enumerate(chapter_texts, 1):
                if i == 1:
                    st.write(chapter)
                else:
                    try:
                        image = get_gemini_imagen_image(imagen_model, chapter, 42, 1)
                        # image[0].save(location=output_file, include_generation_parameters=False)
                        print('writing image and chapter')
                        if image[0] is not None:
                            st.image(image[0]._image_bytes, width=250, output_format="auto")
                    except Exception as e:
                        # Log the error for debugging
                        print(f"Error generating image for chapter {i-1}: {e}")
                        # Continue to the next chapter
                        pass
                    st.write(f"Chapter {i-1}:\n{chapter}\n")


        # Create the audio data
        audio_data = text_to_wav("en-US-Studio-O", generated_story)
        st.audio(audio_data, format="audio/wav", loop=False, autoplay=False)


