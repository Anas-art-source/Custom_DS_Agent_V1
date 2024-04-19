import pandas as pd

data = pd.read_csv('/home/khudi/Desktop/my_own_agent/final_pants_dataset.csv')
data_shirt = pd.read_csv('/home/khudi/Desktop/my_own_agent/final_shirts_dataset.csv')

# print(data.tail())

feature_list = f"""
## Shirts Features
Pants Type: such as, {', '.join(map(str, data['Pants Type'].unique()))}\n
Fabric: such as, {', '.join(map(str, data['Fabric'].unique()))}\n
Waist: such as, {', '.join(map(str, data['Waist'].unique()))}\n
Front Pockets: such as, {', '.join(map(str, data['Front Pockets'].unique()))}\n
Back Pockets: such as, {', '.join(map(str, data["Back Pockets"].unique()))}\n
Closure: such as, {', '.join(map(str, data["Closure"].unique()))}\n
Belt Loops:  such as, {', '.join(map(str, data["Belt Loops"].unique()))}\n
Cuff: such as, {', '.join(map(str, data["Cuff"].unique()))}\n
Pattern: such as, {', '.join(map(str, data["Pattern"].unique()))}\n
Store: such as, {', '.join(map(str, data["Store"].unique()))}\n
Region: such as, {', '.join(map(str, data["Region"].unique()))}\n

## Pants Features
* `size` -  such as, {', '.join(map(str, data_shirt['size'].unique()))}
* `color` -  such as, {', '.join(map(str, data_shirt['color'].unique()))}
* `material` - such as, {', '.join(map(str, data_shirt['material'].unique()))}
* `pattern` - such as, {', '.join(map(str, data_shirt['pattern'].unique()))}
* `sleeve length` - such as, {', '.join(map(str, data_shirt['sleeve length'].unique()))}
* `neck style` - such as, {', '.join(map(str, data_shirt['neck style'].unique()))}
* `pocket style` - such as, {', '.join(map(str, data_shirt['pocket style'].unique()))}
* `tags` - such as, {', '.join(map(str, data_shirt['tags'].unique()))}
* `store`- such as, {', '.join(map(str, data_shirt['Store'].unique()))}
* `region`- such as, {', '.join(map(str, data_shirt['Region'].unique()))}
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
  * `is_store_level` - default to `False`. Could be `True` or `False`. `True` when store level data is required else `False`
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
  * `is_store_level` - default to `False`. Could be `True` or `False`. `True` when store level data is required else `False`
  * `store`- default to `all`.Could be one of these: {', '.join(map(str, data_shirt['Store'].unique()))}
  * `region`- default to `all`.Could be one of these: {', '.join(map(str, data_shirt['Region'].unique()))}
    Returns:
      * `Pandas Dataframe`: with two columns only: `date` and `sales`

"""


# print(data_api_information)

forecasting_model_inference_prompt = """"

* `forecasting_model_inference_api` - Complete API for forecasting model inference. Arguments:
  * `sales_dataframe` - dataframe fetched from data API. It should contain date and sales columns. 
  * `start_date` - start date of forecasting, such as '2024-01-01'
  * `end_date` -  end_date of forecasting, such as '2024-03-31'
    `forecasting_model_inference_api` returns python dictionary with following key-value:
      * `results` - results
      * `average_forecast` - average forecast
      * `total_forecast` - total forecast


"""

new_product_simulate_sf_prompt =  """"
* `new_product_forecasting_inference_api` - API for stimulating forecast for new product. Argument
  * `product_description` - the detail description of the product
  * `start_date` - start date of forecasting, such as '2024-01-01'
  * `end_date` -  end_date of forecasting, such as '2024-03-31'
  `new_product_forecasting_inference_api` returns python dictionary with following key-value:
      * `result` - result
      * `analysis` - the analysis of forecasting using SHAP method
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

Task: {{ current_task }}


## Data APIs
Use data APIs to fetch the data.
They can be called in python code as such: data = pants_data_api(...)

{{ data }}


## Forecasting APIs
Use this API to make inference of forecasting model. Always submit complete dataframe to it
It can be called in python code as such: data = forecasting_model_inference_api(...)

{{forecasting_model}}


## Simulate Sales Forecasting API for New Product
Use this API to simulate sales forecast for new product. 
It can be called in python code as such: data = new_product_forecasting_inference_api(...)

{{new_product_simulate_sf_prompt}}


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
* Always print the variable at each step to understand the progress

## Debug Instruction
* Your code will be executed in python interpretor. In case of any error, you will be given error message along with your code. You need to debug the code and make it error-freeof any error, you will be given error message along with your code. You need to debug the code and make it error-free

"""


# Available prebuilt python function: 
#  - xgboost_model(data: pd.DataFrame, forecast_start_date: str = "01-01-2023", forecast_end_date: str:  "31-01-2023"): use this pre-loaded function to make forecasting


planner_prompt = """
Your are an expert AI that breaks down user question into smaller step-by-step tasks for other AI to execute.

Return step-by-step task using only tools


Most recent conversation is at the bottom of that history.

## **Example**
User Question:  I want to know the sales of pant for june, july, august 2024 for each store

Output: 
[
    "Since it is the business intellegence request, retrieve Sales Data: Write python script to load data for `pants` from June, July, and August 2024 per store level",
    "Calculate Total Sales: Calculate the total sales of pants for each store",
    "Lastly, respond the forecasted result to user in natural language"
]

User Question: can you forecast the sales of pant with Fabric denim for  month of feb 2024

Output: 
[
    "Since it is the forecasting request, extract historical sales data for `pants` with fabric denim",
    "Model Inference: infere forecasting api for febuary 2024 with fetched pants data",
    "Lastly, respond the forecast result to user in natural language"
]


User Question: give me the forecasted sales for shirt in store 2 for next 2 months

Output:    
[
    "Get the next 2 months: use python code to fetch the dates of next 2 months",
    "Since it is the forecasting request, fetch the historical data of `shirts` for store 2.",
    "Model Inference: infere forecasting api for the next 2 months with fetched shirts data",
]


User question:  can you tell me what will be the sales of pants in upcoming month in north region

Output:  
[   
    "Get the upcoming month: use python code to fetch the dates of upcoming month",
    "Since it seems to be the forecasting request, fetch the historical sales data of `pants` and filter data for only north region",
    "Model Inference: infere forecasting api for upcoming months with fetched pants data",
    "Lastly, output the forecasted sales of `pants` for the upcoming month in the north region."
]


User question: Forecast the sales of shirts for the month of july 2024, break it down at store level 

Output:  
[   
    "Since it seems to be the forecasting request, fetch the historical sales data for `shirts` per store level",
    "Model Inference: infere forecasting api for months of july 2024 with fetched shirts data",
    "Lastly, output the forecasted sales of `shirts` in natural language."

]

User question: Stimulate the sales for the following product that I am planning to launch:\nAbout Product: I can see a red shirt with crew neck style and pockets

Output:  
[   
    "Since it is the forecasting request for new product, call simulate sale forecasting API for new product with relevant args",
    "Lastly, explain the results to user"

]

User question: Forecast the sales of shirts for every store for the upcoming 2 months

Output:  
[   
    "Since it is the forecasting request, fetch the historical sales data of shirts from API per store level",
    "Model Inference: infere forecasting api for upcoming 2 months with fetched shirts data"
    "Lastly, explain the results to user"
]

User question: Forecast the sales of shirts for july 2024 and break it down at store level. 

Output:  
[   
    "Since it is the forecasting request, fetch the historical sales data of shirts from API per store level",
    "Model Inference: infere forecasting api for july 2024 with fetched shirts data"
    "Lastly, explain the results to user"
]

User question: What was the sales of shirts in north region and also forecast its sales for next quarter 

Output:  
[   
    "This user question contains both BI query and Forecasting request. First, solve the BI request. For this fetch the sales of shirts in norht region and sum it up and respond to user"
    "Secondly, address the forecasting request. For this first, get the date of next quarter/3 month: use python code to fetch the dates of next quarter/3 month",
    "Thirdly, Model Inference: infere forecasting api for next 3 month/quarter with fetched shirts data"
    "Lastly, explain the results to user"
]


## **Response Format**
Always respond in python list with each element being a task

## **Recent Response**
{{recent_response}}

## **Feedback**
{{ feedback }}

## User question
Past Conversations: {{past_conversation}}\nUse Past Conversation intelligently to extract task for next question. Question could be related to past conversation  
Question: {{user_question}}
"""


debugger_prompt = """
You an an helpful assistant. Your task is to take a look at the python code and error below and rewrite the python code to make it bug-free.

Your response must be in JSON format.

It must be an object, and it must contain two fields:
* `thought`, what is your thought about the task and python code you are writting
* `python_code`, python code to achieve the task. Always use print to track the results of your work

## Python Code and Error
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


feature_extractor = """
You are helpful assistant that extract the features from the product description.

## Data Features
These are the unique valeus of features for pants and shirts.

{{feature_list}}

## Product Description
{{product_description}}

## Response
Your response must be in JSON format.

It must be an object:
* `feature_name`, the name of the feature
* `feature_value`, the value of the feature
  (....this can repeat N times)
* `product_type`, where it is shirt or pant

## Example
User: I can see the shirt in grey color with no pockets and a solid pattern

Output:
{
  "color": "Grey",
  "pocket style": "No Pocket",
  "pattern": "Solid",
  "product_type": "shirts"
}
"""

shap_prompt = """
You are helpful assistant that interpret the forecasting results and list down the factors contribution to increasing sales and factor contributing to decreasing sales

## Results
It is a dictionary with keys being the features of product and values being the forecasted sales

{{results}}

Go!
"""