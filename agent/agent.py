from openai import OpenAI
from langsmith import wrappers
from langsmith import traceable
import os


class LLM:
    def __init__(self, openai_model_name):
        self.llm = wrappers.wrap_openai(OpenAI(api_key=""))
        self.openai_model_name = openai_model_name


    def response_parser(self, response):
        return response.choices[0].message.content

    @traceable
    def step(self, prompt):
        response =  self.llm.chat.completions.create(
            model=self.openai_model_name,
            messages=[
                {"role": "user", "content": prompt}],
        )
        response = self.response_parser(response=response)
        return response
        
