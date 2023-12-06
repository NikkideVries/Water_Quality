# improts: 
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

#--------------------------Acquire Functions----------------------#
# acquire the data: 
def acquire_water():
    '''
    This function will accquire the water potability data set using the water_potability.csv
    '''
    df = pd.read_csv('water_potability.csv')
    return df



#---------------------Prepare Functions--------------------------#
# prepare the data:
def prep_water(df): 
    '''
    This function should take in the water potability df, it will: 
    - Drop null columns
    - Rename the columns to have all lowercase
    '''
    df = df.dropna()
    df.columns = df.columns.str.lower()
    df = df.round(2)
    return df



#----------------------Split Function----------------------------#
def split_water(df):
    '''
    This function will split my data
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify = df.potability)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify = train_validate.potability)
    
    return train, validate, test