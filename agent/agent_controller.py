
from openai import OpenAI
from utils.task import Task
from jinja2 import Template
import json
from .agent import LLM
from jinja2 import Environment

env = Environment(cache_size=0)
import time
import streamlit as st



ANSI_CODES = {
        'PURPLE': '\033[95m',
        'CYAN': '\033[96m',
        'DARKCYAN': '\033[36m',
        'BLUE': '\033[94m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
        'END': '\033[0m'
    }


class AgentController(LLM): 
    def __init__(self, planner_prompt, action_delegator_prompt, openai_model_name, code_executor, data, task_modifier_prompt, debugger_prompt, responder_prompt, forecasting_model_inference_prompt):
        super().__init__(openai_model_name=openai_model_name)
        self.planner_prompt =  planner_prompt
        self.action_delegator_prompt = action_delegator_prompt
        self.tasks = None
        self.task_not_parsed = True
        self.init_user_question = None  
        self.history = []
        self.code_executor = code_executor
        self.data = data
        self.task_modifier_prompt = task_modifier_prompt
        self.debugger_prompt = debugger_prompt
        self.col1 = None
        self.col2 = None
        self.container1 = None
        self.container2 = None
        self.responder_prompt = responder_prompt
        self.forecasting_model_inference_prompt = forecasting_model_inference_prompt


    def responder(self):
        prompt = Template(self.responder_prompt)
        x = prompt.render(context=self.history[-5:], question=self.init_user_question)
        response = self.step(x)
        self.col1 = st.markdown(f":blue[{response}]")


    def chat_window_printer(self, response):
            self.col1 = st.markdown(response)

    def task_printer(self, task):
        with self.container1:
            placeholder = st.empty()
            with placeholder.container():
                st.markdown(task)
            # placeholder.empty()


    def code_printer(self, code):
        self.container2.code(code)

    
    def error_printer(self, error):
        self.container2.error(error, icon='🔥')
    
    def success_print(self, success):
        self.container2.success(success)

    def python_debugger(self, error):
        prompt = Template(self.debugger_prompt)
        x = prompt.render(error=error)
        return  self.step(x)

    def task_modifier(self):
        prompt = Template(self.task_modifier_prompt)
        x = prompt.render( history=self.history, current_task_with_id=self.tasks.get_current_task_with_id())
        # self.print_color(x, "PURPLE")
        output_dict = None
        while output_dict == None:
            response = self.step(x)
            modify_dict = self.actionParser(response)
            if isinstance(modify_dict, dict) and modify_dict['modify'] in ['successful', 'open'] and isinstance(modify_dict['task_number'], int):
                output_dict = modify_dict
                print(output_dict)
        return output_dict


    def print_color(self, message, color):
        print("-"*100)
        print('\n\n')
        print(ANSI_CODES[color] + message + ANSI_CODES['END'])
        print('\n\n')
        print("-"*100)


    def code_executor_handler(self, code):
        results = self.code_executor(code)
      
        return results

    def actionParser(self, action):
        try: 
            action = json.loads(action)
            return action
        except Exception as e:
            return f"Oops, there seems to be an error while parsing the json. Are you sure you reponsd with valid json. Try again!\nYour output: {action}\nError: {e}"



    def action_delegator(self):
        self.task_printer(self.tasks.get_task_template())
        prompt = Template(self.action_delegator_prompt)
        x = prompt.render( history=self.history[-10:], data=self.data, current_task=self.tasks.get_current_task(), forecasting_model=self.forecasting_model_inference_prompt)
        response = response = self.step(x)
        action = self.actionParser(response)
        if isinstance(action, dict):
            self.print_color("Action: " + action['thought'], 'CYAN')
            self.print_color("Action: " + action['python_code'], 'CYAN')
            # here
            self.chat_window_printer(action['thought'])
            self.code_printer(action['python_code'])
            observation = self.code_executor_handler(action['python_code'])
            
            if observation['result'] == "SUCCESS":
                # Memory handler
                self.history.append({
                    'action': action,
                    'observation': observation
                })
                self.print_color("Observation: "  + observation['output'], 'GREEN')
                self.success_print(observation['output'])
                latest_successful_task = self.task_modifier()
                if latest_successful_task['modify'] == 'successful':
                    self.tasks.update_task(latest_successful_task['task_number'], 'successful')                    
            else:
                # debug_message = self.python_debugger(observation)
                self.history.append({
                'action': response,
                'observation': observation
                    })
                self.error_printer(observation['output'])
                self.print_color("Observation: "  + observation['output'], 'RED')
        else:
            self.history.append({
                'action': response,
                'observation': action
            })
            self.print_color("Observation: "  + action, 'RED')
            # latest_successful_task = self.tasks.get_latest_successful_task()
            # self.tasks.update_task(latest_successful_task, 'Unsuccessful. Retry!')
          
    def taskParser(self, task_str):
        # Extracting the substring containing the Python list
        try:
            start_index = task_str.find("[")
            end_index = task_str.find("]") + 1
            python_list_string = task_str[start_index:end_index]

            # Cleaning up the string and removing unnecessary characters
            python_list_string = python_list_string.replace("assistant:", "").strip()

            # Convert the string representation of the list into a Python list
            python_list = eval(python_list_string)

            # Print the extracted Python list
            return python_list
        except Exception as e:
            return f"Oops, there is some error in parsing the task list. Are you sure you output the task in right format. Error: {e}.\nTry again. Best of luck!"


    def planner(self,x):
        response = self.step(x)
        tasks = self.taskParser(response)
        if isinstance(tasks, list):
            self.tasks = Task(tasks)
            self.task_not_parsed = False
            return self.tasks.get_task_template()
        else:
            prompt = Template(self.planner_prompt)
            self.planner_prompt = prompt.render(recent_response=response, feedback=tasks, user_question=self.init_user_question)


    def initiate_chat(self, message, max_turn=25):
        self.init_user_question = message
        for i in range(max_turn):
            time.sleep(1)
            if self.tasks and self.tasks.check_termination():
                self.responder()
                print("FINISHED")
                break
            if i == 0:
                while self.task_not_parsed:
                    prompt = Template(self.planner_prompt)
                    x = prompt.render(recent_response="No Response yet!", feedback='No feedback yet!',user_question=self.init_user_question)
                    tasks = self.planner(x)
                    self.print_color(tasks, 'YELLOW')
            else:
                # Take the template and propose and action 
                 action_response = self.action_delegator()
                 self.print_color(self.tasks.get_task_template(), 'YELLOW')


    def setup_streamlit(self):
        st.set_page_config(page_title="The Data Science Agent", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
        col1, col2 = st.columns([2,1],gap = "medium")
        col1.title("Custom Data Science Agent")
        col1.header("Chat Window")
        col2.header("Working")
        if "codes" not in st.session_state:
            st.session_state.codes = []

        if "responses" not in st.session_state:
            st.session_state.responses = []

        if "tasks" not in st.session_state:
            st.session_state.tasks = []
        with col2:
            self.container1 = st.container(height=200, border=True)
            self.container2 = st.container(height=400, border=True)
        self.col1 = col1
        self.col2 = col2

    def streamlit_initiate_chat(self, max_turn=25):
        self.setup_streamlit()
        
        with self.col1:
            for response in st.session_state.responses:
                st.markdown(response["content"])

            if content:=st.session_state.responses:
                st.session_state.responses.append({"role": "user", "content": content})

                    
            prompt = st.chat_input("Say something")
            if prompt:
                st.chat_message("user").write(prompt)
                self.initiate_chat(message=prompt)





   