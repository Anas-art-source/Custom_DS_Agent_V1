import pandas as pd


df = pd.read_csv('/home/khudi/Desktop/my_own_agent/Shirts_sales_data.csv')


def shirts_data_api(start_date:str='2022-01-01', end_date:str='2023-12-31', sku_id: str='all', name:str="all", size:str="all", color:str='all',  material:str='all', pattern:str='all', sleeve_length:str='all', neck_style:str='all', tags:str='all', pocket_style:str='all'):
    kwargs = {k: v for k, v in locals().items() if k != 'self' and k != 'kwargs' and v != 'all'}
    column_mapping = {
        'sku_id': 'sku id',
        'name': 'name',
        'size': 'size',
        'color': 'color',
        'material': 'material',
        'pattern': 'pattern',
        'sleeve_length':'sleeve length',
        'neck_style': 'neck style',
        'tags': 'tags',
        'pocket_style': "pocket style"
    }

    filters = {}
    for column, value in kwargs.items():
        if value != 'all' and  column not in ['start_date', 'end_date']:
            filters[column_mapping[column]] = value
    filtered_df = df
    for k,v in filters.items():
        filtered_df = filtered_df[filtered_df[k] == v]

    filtered_df = filtered_df[(filtered_df['date'] >= kwargs['start_date']) & (filtered_df['date'] <= kwargs['end_date'])]

    filtered_df = filtered_df.groupby(['date'])['sales'].sum().reset_index()
    print(filtered_df)
    return filtered_df



