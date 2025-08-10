import os
import tempfile
from skimage import io
import pickle
from glob import glob
import numpy as np
import json
from loguru import logger
import google.generativeai as genai
from time import sleep

best_gemini_generation_prompt = '''
You are analyzing a satellite image to create a comprehensive textual description for precise image retrieval from a vast da
tabase.  Your description will serve as a detailed visual fingerprint, enabling users to pinpoint this specific image among 
millions using text-based search queries.

Structure your description into the following categories, providing specific details for each:

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

* **Geological Features:**  Describe any unique geological formations.  For each (rock outcrops, fault lines, volcanic 
features, etc.), specify:
    - **Type:** (e.g., "granite outcrop," "volcanic crater")
    - **Location:** Precise location
    - **Size/Extent:**
    - **Shape, Color, and Texture:**

* **Other Distinctive Features:**  Describe any other visually prominent features not captured above (e.g., unique land cover 
textures, atmospheric phenomena, signs of natural events).


Generate a textual description, focusing on features that are most distinctive and useful for differentiating this image.  
Pay close attention to spatial relationships, noting the proximity and arrangement of features. Aim for a comprehensive 
yet concise description to facilitate accurate and efficient image retrieval using text-based searches.  Provide illustrative 
examples if necessary.

Additionally, append a section titled 'Coverage estimation', and provide in it a json formated output of 
'feature_name': 'percentage of coverage in the image'
'''

class GeminiMultimodalModel:

    def __init__(self, 
                 api_key,
                 generation_model_name = "gemini-2.5-pro", 
                 embeddings_model_name = 'gemini-embedding-001',
                 temperature = 1,   
                 top_p = 0.95,       
                 max_output_tokens = 8192,
                 verbose = False):
        """
        api_key: string with the api key of the file name to read it from
        """

        super().__init__()

        self.generation_model_name = generation_model_name
        self.embeddings_model_name = embeddings_model_name
        self.temperature           = temperature
        self.top_p                 = top_p
        self.max_output_tokens     = max_output_tokens
        self.verbose               = verbose
        self.api_key               = api_key
        self.generation_prompt     = best_gemini_generation_prompt
        

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

        if verbose:
            logger.info(f'using generation model {self.generation_model_name}')
            logger.info(f'using embeddings model {self.embeddings_model_name}')
            logger.info(f'using config {generation_config}')

        safety_settings = [
          # Gemini's safety settings for blocking harmful content
          # (Set to "BLOCK_NONE" for no blocking)
          {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
          {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
          {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
          {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        self.generation_model = genai.GenerativeModel(
          model_name=generation_model_name,  
          safety_settings=safety_settings,
          generation_config=generation_config,
        )

    def set_generation_prompt(self, prompt):
        self.generation_prompt = prompt

    def generate_description_for_image(self, img, max_retries=5, sleep_secs_before_retry=30):

        attempts = 0
        while True:
            try:

                # uploads img to gemini so that it is part of the prompt
                with tempfile.TemporaryDirectory() as tmp:
                    img_path = os.path.join(tmp, 'img.jpg')
                    io.imsave(img_path, img)
                    uploaded_img_path = genai.upload_file(img_path, mime_type=None)
                    if self.verbose: 
                        logger.info(f"uploaded file image to prompt")
                
                if self.verbose:
                    logger.info('querying gemini for description')
                chat_session = self.generation_model.start_chat(
                history=[
                    {"role": "user", "parts": [uploaded_img_path]},
                ]
                )
                prompt = self.generation_prompt
                response = chat_session.send_message(prompt)
                
                return response.text

            except Exception as e:
                if self.verbose:
                    logger.error(f'attempt {attempts+1}, exception {str(e)}')

                attempts += 1
                if attempts > max_retries:
                    return f'<!!error!!>::<!!pending!!>:::\n\n{str(e)}'

                if sleep_secs_before_retry is not None:
                    sleep(sleep_secs_before_retry)


    def get_embedding(self, text, max_retries=5, sleep_secs_before_retry=10, task_type="RETRIEVAL_DOCUMENT"):
    
        attempts = 0
        while True:
            try:
                result = genai.embed_content(
                            model=self.embeddings_model_name,
                            content=text,
                            task_type=task_type
                        )
                return np.r_[result['embedding']]

            except Exception as e:
                attempts += 1
                if attempts > max_retries:
                    return f'TEXT:::{text}:::fdl2025\n\n{str(e)}'

                if sleep_secs_before_retry is not None:
                    sleep(sleep_secs_before_retry)





