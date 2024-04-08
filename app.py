from prompts.prompts import forecasting_model_inference_prompt,task_assigner_prompt, planner_prompt, data_information, task_modifier_prompt, data_api_information, debugger_prompt, responder_prompt
from sandbox.sandbox import PythonSandbox
from agent.agent_controller import AgentController
# import the OpenAI Python library for calling the OpenAI API
from openai import OpenAI
import os
import json
from jinja2 import Template
from tools.pants_data_api import pants_data_api
from tools.shirts_data_api import shirts_data_api
from tools.forecasting_model_inference import forecasting_model_inference_api

import streamlit as st 

sandbox = PythonSandbox()
# print(sandbox.execute_code("print(pants_data_api())"))

controller = AgentController(planner_prompt=planner_prompt,
                            action_delegator_prompt=task_assigner_prompt,
                            openai_model_name='gpt-3.5-turbo-0125',
                            code_executor=sandbox.execute_code,
                            data=data_api_information,
                            task_modifier_prompt=task_modifier_prompt,
                            debugger_prompt=debugger_prompt,
                            responder_prompt=responder_prompt,
                            forecasting_model_inference_prompt=forecasting_model_inference_prompt
)

controller.streamlit_initiate_chat("can you forecast the sales of pant for january 2024")