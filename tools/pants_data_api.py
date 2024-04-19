import pandas as pd


df = pd.read_csv('/home/khudi/Desktop/my_own_agent/final_pants_dataset.csv')


# print(df.tail())

def pants_data_api(start_date:str='2022-01-01', end_date:str='2023-12-31', sku_id='all', size='all', pant_type='all', fabric='all', waist='all', front_pockets='all', back_pockets='all', closure='all', belt_loops='all', cuff='all', is_store_level:bool=False, store:str='all', region='all'):
  kwargs = {k: v for k, v in locals().items() if k != 'self' and k != 'kwargs' and v != 'all' and k != 'is_store_level'}
  import pandas as pd
  column_mapping = {
        'sku_id': 'SKU ID',
        'size': 'Size',
        'pant_type': "Pants Type",
        'fabric': "Fabric",
        'waist': "Waist",
        'front_pockets': "Front Pockets",
        'back_pockets': "Back Pockets",
        'closure': "Closure",
        'belt_loops': "Belt Loops",
        'cuff': "Cuff",
        'pattern': "Pattern",
        'store': "Store",
        'region': "Region",

    }


  filters = {}
  for column, value in kwargs.items():
     if value != 'all' and  column not in ['start_date', 'end_date']:
          filters[column_mapping[column]] = value

  filtered_df = df
  for k,v in filters.items():
    filtered_df = filtered_df[filtered_df[k] == v]

  filtered_df = filtered_df[(filtered_df['Date'] >= kwargs['start_date']) & (filtered_df['Date'] <= kwargs['end_date'])]
  if is_store_level:  
      filtered_df = filtered_df.groupby(['Date', "Store"])['Sales'].sum().reset_index()
      filtered_df.rename(columns={'Date': "date", 'Sales': 'sales', "Store": 'store'}, inplace=True)
      print("The data is at store level. It has three columns: dates, store, and sales. The unique values of stores are: ", filtered_df['store'].unique())
  else:
      filtered_df = filtered_df.groupby(['Date'])['Sales'].sum().reset_index()
      filtered_df.rename(columns={'Date': "date", 'Sales': 'sales'}, inplace=True)
      print("the data has two columns: dates and sales")

  filtered_df['item_sold'] = 1000
  print('First five rows of fetched data: ', filtered_df.head())
  return filtered_df




# print(pants_data_api(is_store_level=True))