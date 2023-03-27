import pandas as pd
from neuroCombat import neuroCombat

def combat(data,covariables,category_name):
    '''
    data: 2 dimensional array. The 1st dimension index samples. The 2nd dimension index variables.
    covariables: dataframe. Must contain a column named 'center'.
    category_name: a list of str.

    return:
    data_combat: 2 dimensional array. The 1st dimension index samples. The 2nd dimension index variables.
    '''
    data=data.T
    data_combat = neuroCombat(dat=data,
        covars=covariables,
        batch_col='center',
        categorical_cols=category_name)["data"]
    data_combat=data_combat.T
    return data_combat