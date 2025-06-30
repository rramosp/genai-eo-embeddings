import os
import tempfile
from skimage import io
import pickle
from glob import glob
import numpy as np
import json
from loguru import logger
import google.generativeai as genai

best_gemini_generation_prompt = '''
You are analyzing a satellite image to create a comprehensive textual description for precise image retrieval from a vast da
tabase.  Your description will serve as a detailed visual fingerprint, enabling users to pinpoint this specific image among 
millions using text-based search queries.

Structure your description into the following categories, providing specific details for each:

* **Location and Extent:**  Describe the geographic area covered by the image if discernible (e.g., coordinates, region name
). Estimate the area covered by the image (e.g., square kilometers).

* **Dominant Land Cover:** Identify the most prominent land cover types (e.g., urban, forest, agricultural, water, barren) a
nd estimate the percentage of the image covered by each.

* **Terrain:** Describe the dominant terrain characteristics.  For each feature (mountains, valleys, plateaus, canyons, hill
s, coastlines, etc.), specify:
    - **Type:** (e.g., mountain range, valley, coastal plain)
    - **Location:** Precise location within the image (e.g., "northwest quadrant," "stretching from east to west")
    - **Size/Extent:**  (e.g., "small," "large," "covering the southern third")
    - **Shape and Attributes:** (e.g., "jagged peaks," "steep slopes," "sandy beach")

* **Vegetation:** Describe all significant vegetation types. For each (forests, grasslands, shrublands, agricultural fields,
 etc.), specify:
    - **Type:**  (e.g., "deciduous forest," "coniferous forest," "cultivated fields")
    - **Location:** Precise location
    - **Extent:**  (e.g., "dense," "sparse," "covering the eastern half")
    - **Health and Appearance:** (e.g., "healthy," "dry," "lush green")
    - **Patterns:**  (e.g., "uniform," "patchy," "linear")

* **Water Bodies:**  Describe all significant water features.  For each (rivers, lakes, oceans, reservoirs, glaciers, etc.),
 specify:
    - **Type:** (e.g., "meandering river," "crescent-shaped lake")
    - **Location:** Precise location
    - **Size/Extent:**
    - **Flow Direction (for rivers):**
    - **Shoreline Characteristics:** (e.g., "rocky," "sandy," "developed")

* **Man-Made Structures:** Describe all significant human-made features.  For each (roads, buildings, bridges, dams, urban a
reas, airports, agricultural infrastructure, etc.), specify:
    - **Type:**  (e.g., "highway," "residential area," "industrial complex")
    - **Location:**  Precise location
    - **Size/Extent:**
    - **Arrangement:** (e.g., "linear," "clustered," "grid-like")
    - **Materials (if apparent):** (e.g., "concrete," "metal")

* **Geological Features:**  Describe any unique geological formations.  For each (rock outcrops, fault lines, volcanic features, etc.), specify:
    - **Type:** (e.g., "granite outcrop," "volcanic crater")
    - **Location:** Precise location
    - **Size/Extent:**
    - **Shape, Color, and Texture:**

* **Other Distinctive Features:**  Describe any other visually prominent features not captured above (e.g., unique land cover textures, atmospheric phenomena, signs of natural events).


Focus on features that are most distinctive and useful for differentiating this image.  Provide specific locations and estimate the extent of features. Pay close attention to spatial relationships, noting the proximity and arrangement of features.  Aim for a comprehensive yet concise description to facilitate accurate and efficient image retrieval using text-based searches.  Provide illustrative examples if necessary.

provide a json formated output of 'feature_name': 'percentage of coverage in the image'
'''



class GeminiMultimodalModel:

    def __init__(self, 
                 api_key,
                 model_name = "gemini-1.5-pro-latest", 
                 embeddings_model_name = '',
                 temperature = 1,   
                 top_p = 0.95,       
                 max_output_tokens = 8192,
                 verbose = False):
        """
        api_key: string with the api key of the file name to read it from
        """

        super().__init__()

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_output_tokens = max_output_tokens
        self.verbose = verbose
        self.api_key = api_key
        self.generation_prompt = best_gemini_generation_prompt
        
        # Configure the Gemini API
        if os.path.isfile(self.api_key):
            with open(self.api_key) as f:
                self.api_key = f.read().strip()
    
        genai.configure(api_key = self.api_key) 

        # Model settings for generating responses
        generation_config = {
          "temperature": temperature,              # Creativity (0: deterministic, 1: high variety)
          "top_p": top_p,                          # Focus on high-probability words
          "max_output_tokens": max_output_tokens,  # Limit response length
          "response_mime_type": "text/plain",      # Output as plain text
        }
        
        safety_settings = [
          # Gemini's safety settings for blocking harmful content
          # (Set to "BLOCK_NONE" for no blocking)
          {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
          {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
          {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
          {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        self.model = genai.GenerativeModel(
          model_name=model_name,  
          safety_settings=safety_settings,
          generation_config=generation_config,
        )

    def get_name(self):
        return self.model_name

    def set_generation_prompt(self, prompt):
        self.generation_prompt = prompt

    def generate_description_for_image(self, img):
        # uploads img to gemini so that it is part of the prompt
        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, 'img.jpg')
            io.imsave(img_path, img)
            uploaded_img_path = genai.upload_file(img_path, mime_type=None)
            if self.verbose: 
                logger.info(f"uploaded file image to prompt")
        
        if self.verbose:
            logger.info('querying gemini for description')
        chat_session = self.model.start_chat(
          history=[
            {"role": "user", "parts": [uploaded_img_path]},
          ]
        )
        prompt = self.generation_prompt
        response = chat_session.send_message(prompt)
        
        return response.text


    def generate_answer(self, text):
        chat_session = self.model.start_chat()    
        self.response = chat_session.send_message(text)
        return self.response.text


