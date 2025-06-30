import os
from mistralai import Mistral
from mistralai import SDKError
import numpy as np
from time import sleep

class MistralEmbeddingsAPIModel():

    def __init__(self, api_key):
        self.api_key = api_key


        if os.path.isfile(self.api_key):
            with open(self.api_key) as f:
                self.api_key = f.read().strip()

        self.model = "mistral-embed"

        print(f'model {self.model} configure for API calls')

        self.client = Mistral(api_key=self.api_key)

    def get_name(self):
        return 'mistral-embed-api'

    def generate_embeddings_for_text(self, text):

        
        while True:
            try:
                e = self.client.embeddings.create(
                    model=self.model,
                    inputs=[text],
                )
                return {'api': np.r_[e.data[0].model_dump()['embedding']]}
            except SDKError:
                sleep(2)    
