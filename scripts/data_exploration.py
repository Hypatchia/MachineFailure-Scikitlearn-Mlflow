
    
def explore_data(data):
    # Check the information of the data 
    print('Information of the data: \n', data.info())
    print('Types of the data: \n', data.dtypes)  
    # Check the shape of the data
    print('Shape of the data: \n', data.shape)

    # Check the first five rows of the data
    print('First five rows of the data: \n', data.head())

    # Check the last five rows of the data
    print('Last five rows of the data: \n', data.tail())

    # Check the summary statistics of the data
    print('Summary statistics of the data: \n', data.describe())

    # Check unique values
    data.nunique()
    print('Unique values: \n', data.nunique())

    return None