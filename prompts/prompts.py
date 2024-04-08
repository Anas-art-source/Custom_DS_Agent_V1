import pandas as pd

data = pd.read_csv('/home/khudi/Desktop/my_own_agent/final_pants_dataset.csv')
data_shirt = pd.read_csv('/home/khudi/Desktop/my_own_agent/Shirts_sales_data.csv')

# print(data.tail())

data_information = f"""
Data path: /home/khudi/Desktop/my_own_agent/final_pants_dataset.csv

SKU ID: {', '.join(map(str, data['SKU ID'].unique()))}\n
Size: {', '.join(map(str, data['Size'].unique()))}\n
Pants Type: {', '.join(map(str, data['Pants Type'].unique()))}\n
Fabric:  \n
Waist: {', '.join(map(str, data['Waist'].unique()))}\n
Front Pockets: {', '.join(map(str, data['Front Pockets'].unique()))}\n
Back Pockets: {', '.join(map(str, data["Back Pockets"].unique()))}\n
Closure: {', '.join(map(str, data["Closure"].unique()))}\n
Belt Loops:  {', '.join(map(str, data["Belt Loops"].unique()))}\n
Cuff: {', '.join(map(str, data["Cuff"].unique()))}\n
Pattern: {', '.join(map(str, data["Pattern"].unique()))}\n
Store: {', '.join(map(str, data["Store"].unique()))}\n
Region:{', '.join(map(str, data["Region"].unique()))}\n
Date: Dates ranging from 01-01-2022 to 01-01-2023\n
Sales: Sales Amount
"""
# def shirt_data_api(start_date:str='2022-01-01', end_date:str='2023-12-31', sku_id: str='all', name:str="all", size:str="all", color:str='all',  material:str='all', pattern:str='all', sleeve_length:str='all', neck_style:str='all', tags:str='all', fit_pocket_style:str='all'):

data_api_information = f"""
* `pants_data_api` - fetch pants sales data. Arguments:
  * `start_date` - starting date of data. Ranging from 2022-01-01 to 2023-12-30
  * `end_date` - ending date of data. ranging from 2022-01-01 to 2023-12-30
  * `sku_id` - default to `all`. Could be one of these: {', '.join(map(str, data['SKU ID'].unique()))}
  * `size` - default to `all`. Could be one of these: {', '.join(map(str, data['Size'].unique()))}
  * `pant_type` - default to `all`. Could be one of these: {', '.join(map(str, data['Pants Type'].unique()))}
  * `fabric` - default to `all`. Could be one of these: {', '.join(map(str, data['Fabric'].unique()))}
  * `waist` - default to `all`. Could be one of these: {', '.join(map(str, data['Waist'].unique()))}
  * `front_pockets` - default to `all`. Could be one of these: {', '.join(map(str, data['Front Pockets'].unique()))}
  * `back_pockets` - default to `all`. Could be one of these: {', '.join(map(str, data["Back Pockets"].unique()))}
  * `closure` - default to `all`. Could be one of these: {', '.join(map(str, data["Closure"].unique()))}
  * `belt_loops` - default to `all`. Could be one of these: {', '.join(map(str, data["Belt Loops"].unique()))}
  * `cuff` - default to `all`. Could be one of these: {', '.join(map(str, data["Cuff"].unique()))}
  * `store` - default to `all`. Could be one of these: {', '.join(map(str, data["Store"].unique()))}
  * `region` - default to `all`. Could be one of these: {', '.join(map(str, data["Region"].unique()))}
    Returns:
      * `Pandas Dataframe`: with two columns only: `date` and `sales`

* `shirts_data_api` - fetch shirts sales data. Arguments:
  * `start_date` - starting date of data. Ranging from 2022-01-01 to 2023-12-30
  * `end_date` - ending date of data. ranging from 2022-01-01 to 2023-12-30
  * `sku_id` - default to `all`. Could be one of these: {', '.join(map(str, data_shirt['sku id'].unique()))}
  * `size` - default to `all`. Could be one of these: {', '.join(map(str, data_shirt['size'].unique()))}
  * `color` - default to `all`. Could be one of these: {', '.join(map(str, data_shirt['color'].unique()))}
  * `material` - default to `all`.Could be one of these: {', '.join(map(str, data_shirt['material'].unique()))}
  * `pattern` - default to `all`.Could be one of these: {', '.join(map(str, data_shirt['pattern'].unique()))}
  * `sleeve_length` - default to `all`.Could be one of these: {', '.join(map(str, data_shirt['sleeve length'].unique()))}
  * `neck_style` - default to `all`.Could be one of these: {', '.join(map(str, data_shirt['neck style'].unique()))}
  * `pocket_style` - default to `all`.Could be one of these: {', '.join(map(str, data_shirt['pocket style'].unique()))}
  * `tags` - default to `all`.Could be one of these: {', '.join(map(str, data_shirt['tags'].unique()))}
    Returns:
      * `Pandas Dataframe`: with two columns only: `date` and `sales`

"""

forecasting_model_inference_prompt = """"

* `forecasting_model_inference_api` - api for forecasting model inference. Arguments:
  * `sales_dataframe` - dataframe use for training and forecasting
  * `start_date` - start date of forecasting
  * `end_date` -  end_date of forecasting
  Returns: dictionary with following key-value:
    * `results` - results
    * `average_forecast` - average forecast
    * `total_forecast` - total forecast


"""

task_modifier_prompt = """
You are an helpful assistant. Your task is to keep the track of conversation and tasks. You take action to modify the status of task by looking into conversation

## Tasks
{{current_task_with_id}}

## Conversation
{{history}}


## Action
What is your next action? Your response must be in JSON format.

It must be an object, and it must contain two fields:
* `modify`, `successful` if you are sure that task is completed successfully, else `open`
* `task_number`, the index of task in the list

Always be 100% sure before marking any task `successful`

"""

# task_assigner_prompt = """
# You are the expert problem solver. You have to solve tasks, one at a time, by taking action.

# Tasks:
# {{ tasks }}


# Current Tasks:
# You need to solve the following task until it is successful. In case of error, retry.

# {{ current_task }}


# ## Data APIs
# Use data api to fetch the data.
# they call be called in python code as such: data = pants_data_api(...)

# {{ data }}


# ## History
# Here is a recent history of actions you've taken   as well as observations you've made. 

# {{ history }}

# Your most recent action is at the bottom of that history.


# ## Action
# What is your next thought and action? Your response must be in JSON format.

# It must be an object, and it must contain two fields:
# * `thought`, what is your thought about the task and python code you are writting
# * `python_code`, python code to achieve the task

# You have to write your own python code.

# What is your next thought and action? Again, you must reply with JSON, and only with JSON.

# """



task_assigner_prompt = """
You are the expert problem solver. You have to solve tasks, one at a time, by taking action.

## Current Task:
You need to solve the following task until it is successful. In case of error, retry.

{{ current_task }}


## Data APIs
Use data APIs to fetch the data.
They can be called in python code as such: data = pants_data_api(...)

{{ data }}


## Forecasting APIs
Use this API to make inference of forecasting model. Always submit complete dataframe to it
It can be called in python code as such: data = forecasting_model_inference_api(...)

{{forecasting_model}}



## History
Here is a recent history of actions you've taken   as well as observations you've made. 

{{ history }}

Your most recent action is at the bottom of that history.


## Action
What is your next thought and action? Your response must be in JSON format.

It must be an object, and it must contain two fields:
* `thought`, what is your thought about the task and python code you are writting
* `python_code`, python code to achieve the task. Always use print to track the results of your work

You have to write your own python code.

What is your next thought and action? Again, you must reply with JSON, and only with JSON.


## Special instruction
* You can use the variable you created in the past (see in history)
* Dont assume any data and data path on your own
* In case of error in the code, stop and think hard before correcting the error
"""


# Available prebuilt python function: 
#  - xgboost_model(data: pd.DataFrame, forecast_start_date: str = "01-01-2023", forecast_end_date: str:  "31-01-2023"): use this pre-loaded function to make forecasting


planner_prompt = """
Your are an expert AI that breaks down user question into smaller step-by-step tasks for other AI to execute.

Return step-by-step task using only tools

**Example**
User Question:  I want to know the sales of pant for june, july, august 2024 for each store

Output: 
[
    "Retrieve Sales Data: Write python script to load data f for pants from June, July, and August 2024. Ensure that the data includes information about the stores where the sales occurred.",
    "Filter Data by Date: Filter the retrieved data to only include sales data for the specified months (June, July, August 2024).",
    "Group Data by Store: Group the filtered data by store to get sales information for each store.",
    "Calculate Total Sales: Calculate the total sales of pants for each store.",
    "Present Results: Present the calculated total sales for each store in a clear and understandable format, such as a table or a chart"
]

User Question: can you forecast the sales of pant with Fabric denim for  month of feb 2024

Output: 
[
    "Extract historical sales data for pants with fabric denim",
    "Preprocess the data (cleaning, formatting, etc.).",
    "Model Inference: infere forecasting api",
    "present the forecasted sales data."
]


User Question: give me the forecasted sales for shirt in store 2 for next 2 months

Output:    
[
    "Get the next 2 months: use python code to fetch the dates of next 2 months",
    "Understand Data Availability: Determine if historical sales data for shirts in store 2 is available.",
    "Data Preprocessing: Preprocess the available data by cleaning, formatting, and organizing it for analysis.",
    "Model Inference: infere forecasting api",
    "Presentation", "Visualize the forecasted sales along with historical data to communicate insights effectively."
]


User question:  can you tell me what will be the sales of pants in upcoming month in north region

Output:  
[   
    "Get the upcoming month: use python code to fetch the dates of upcoming month",
    "Identify the historical sales data of pants.",
    "Filter the historical sales data to include only the sales in the north region.",
    "Model Inference: infere forecasting api",
    "Output the forecasted sales for the upcoming month in the north region."
]
s
**Response Format**
Always respond in python list with each element being a task

**Recent Response**
{{recent_response}}

**Feedback**
{{ feedback }}

User question: {{user_question}}
"""


debugger_prompt = """
You an an helpful assistant. Your task is to take a look at the python code and error below and propose a solution

{{error}}
"""


responder_prompt = """"
You are helpful assistant that responds to human question by considering information is the context. You dont make up any answer on your own

## Context
{{context}}

## Question
{{question}}

Go!
"""
