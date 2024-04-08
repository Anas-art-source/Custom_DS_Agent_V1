import pandas as pd


df = pd.read_csv('/home/khudi/Desktop/my_own_agent/final_pants_dataset.csv')


# print(df.tail())

def pants_data_api(start_date:str='2022-01-01', end_date:str='2023-12-31', sku_id='all', size='all', pant_type='all', fabric='all', waist='all', front_pockets='all', back_pockets='all', closure='all', belt_loops='all', cuff='all', store='all', region='all'):
  kwargs = {k: v for k, v in locals().items() if k != 'self' and k != 'kwargs' and v != 'all'}
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
  print(filters)

  filtered_df = df
  for k,v in filters.items():
    filtered_df = filtered_df[filtered_df[k] == v]

  filtered_df = filtered_df[(filtered_df['Date'] >= kwargs['start_date']) & (filtered_df['Date'] <= kwargs['end_date'])]
  filtered_df = filtered_df.groupby(['Date'])['Sales'].sum().reset_index()
  filtered_df.rename(columns={'Date': "date", 'Sales': 'sales'}, inplace=True)
  print(filtered_df)
  return filtered_df[['date', 'sales']]



# print(pants_data_api())