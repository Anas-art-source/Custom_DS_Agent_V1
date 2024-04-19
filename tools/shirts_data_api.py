import pandas as pd


df = pd.read_csv('/home/khudi/Desktop/my_own_agent/final_shirts_dataset.csv')
# print(df.head())
def shirts_data_api(start_date:str='2022-01-01', end_date:str='2023-12-31', sku_id: str='all', size:str="all", color:str='all',  material:str='all', pattern:str='all', sleeve_length:str='all', neck_style:str='all', tags:str='all', pocket_style:str='all', is_store_level:bool=False, store:str='all', region:str='all'):
    kwargs = {k: v for k, v in locals().items() if k != 'self' and k != 'kwargs' and v != 'all' and k != 'is_store_level'}
    column_mapping = {
        'sku_id': 'sku id',
        'name': 'name',
        'size': 'size', 
        'material': 'material',
        'pattern': 'pattern',
        'sleeve_length':'sleeve length',
        'neck_style': 'neck style',
        'tags': 'tags',
        'pocket_style': "pocket style",
        'store': 'Store',
        'region': 'Region'
    }

    filters = {}
    for column, value in kwargs.items():
        if value != 'all' and  column not in ['start_date', 'end_date']:
            filters[column_mapping[column]] = value
    filtered_df = df
    for k,v in filters.items():
        filtered_df = filtered_df[filtered_df[k] == v]

    filtered_df = filtered_df[(filtered_df['date'] >= kwargs['start_date']) & (filtered_df['date'] <= kwargs['end_date'])]

    if is_store_level:  
        filtered_df = filtered_df.groupby(['date', "Store"])[['sales', 'item_sold']].sum().reset_index()
        filtered_df.rename(columns={'date': "date", 'sales': 'sales', "Store": 'store', 'item_sold': 'item_sold'}, inplace=True)
        print("The data is at store level. It has three columns: dates, store, and sales. The unique values of stores are: ", filtered_df['store'].unique())
    else:
        filtered_df = filtered_df.groupby(['date'])[['sales', 'item_sold']].sum().reset_index()
        filtered_df.rename(columns={'date': "date", 'sales': 'sales', 'item_sold': 'item_sold'}, inplace=True)
        print("the data has two columns: dates and sales")

    print('First five rows of fetched data: ', filtered_df.head())
    return filtered_df


print(shirts_data_api(is_store_level=True))


