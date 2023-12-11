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





#------------------Create bins---------------------------------#
def safe_water(df):
    '''
    This function will create bins for what is recomended as safe
    '''
    df['ph_range'] = pd.cut(df.ph, [0,6.5,8.5,14], \
                           labels = ['acidic', 'safe', 'basic'])
    df['hardness_range'] = pd.cut(df.hardness, [0,17.1,60,120,180,372], \
                                 labels = ['soft', 'slightly_hard','moderately_hard','hard','very_hard'])
    df['solids_range'] = pd.cut(df.solids, [0,300,600,900,1200,57000], \
                               labels = ['excellent','good','fair','poor','unacceptable'])
    df['chloramines_range'] = pd.cut(df.chloramines, [0,4,14], \
                                labels = ['safe','high'])
    df['sulfate_range'] = pd.cut(df.sulfate, [0,250,482 ], \
                      labels = ['safe', 'high'])
    df['conductivity_rage'] = pd.cut(df.conductivity, [0,400,1000], \
                                labels = ['safe', 'high'])
    df['organic_car_range'] = pd.cut(df.organic_carbon, [0,3,30], \
                                labels = ['safe','high'])
    df['trihalomethanes_range'] = pd.cut(df.trihalomethanes, [0,80,125], \
                                    labels = ['safe', 'high'])
    df['turbidity_range'] = pd.cut(df.turbidity, [0,5.0, 7], \
                              labels = ['safe','high'])
    return df


def water_range(df):
    '''
    This function will create bins for what are the ranges for the data
    '''
    df['ph_level'] = pd.cut(df.ph, [0,4,7,14], labels = ['high','medium','low'])
    df['hardness_level'] = pd.cut(df.hardness, [73.49,150,250,317.34], labels = ['high','medium','low'])
    df['solids_level'] = pd.cut(df.solids, [320.94,10000,20000,56488.67], labels = ['high','medium','low'])
    df['chloramines_level'] = pd.cut(df.chloramines, [1.39,4,8,13.13], labels = ['high','medium','low'])
    df['sulfate_level'] = pd.cut(df.sulfate, [129.0,250,350,481.03], labels = ['high','medium','low'])
    df['conductivity_level'] = pd.cut(df.conductivity, [201.62, 400,600,753.34], labels = ['high','medium','low'])
    df['oraganic_level'] = pd.cut(df.organic_carbon, [2.20,10,20,27.01],labels = ['high','medium','low'])
    df['trihalomethanes_level'] = pd.cut(df.trihalomethanes, [8.58,50,80,124.00], labels = ['high','medium','low'])
    df['turbidity_level'] = pd.cut(df.turbidity, [1.45,3,5,6.49], labels = ['high','medium','low']) 
    
    return df

#-----------------------Dummies-----------------------#
# need to create dummies; 
def create_dummies(df): 
    dummies_list = []
    cat_cols = ['ph_range',
                 'hardness_range',
                 'solids_range',
                 'chloramines_range',
                 'sulfate_range',
                 'conductivity_rage',
                 'organic_car_range',
                 'trihalomethanes_range',
                 'turbidity_range']
    for col in cat_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        dummies_list.append(dummies)
    df = pd.concat([df] + dummies_list, axis = 1)
    return df

# need to drop non dummy columns
def drop_columns(train, val, test):
    
    dropcols = ['ph_range',
 'hardness_range',
 'solids_range',
 'chloramines_range',
 'sulfate_range',
 'conductivity_rage',
 'organic_car_range',
 'trihalomethanes_range',
 'turbidity_range']
    
    train.drop(columns = dropcols, inplace = True)
    val.drop(columns = dropcols, inplace=True)
    test.drop(columns = dropcols, inplace=True)
    return train, val, test