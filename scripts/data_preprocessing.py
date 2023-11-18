
def preprocess_data(data):

    # Select columns to drop
    columns_to_drop = ['UDI', 'Product ID']

    # Drop columns with no predictive power
    data.drop(columns=columns_to_drop, axis=1, inplace=True)
    
    print('Data Null: \n', data.isnull().sum())

    if data.isnull().sum().sum() == 0:
        print('Data has no null values')
    
    else:
        print('Data has null values')
        data.dropna(axis=0,inplace=True)
    
    # Drop duplicates
    data = data.drop_duplicates()
    
    return data
