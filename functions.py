# functions for the lab

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler


# This function obtains the trained values of the num_tr, cat_enc, minmax_tr
def Model_train(df):
    X_train = df.copy()
    
    # Numerical categorical
    numerical_train, categorical_train= cat_num_split(X_train)

    # Numerica Transformation
    num_tr = num_get_tr_power(numerical_train)
    numerical_train_trans = num_transform(numerical_train, num_tr)
    
    # Group some categories
    cat_nom_train_group = cat_group_labels(categorical_train)

    # For the OneHotEncoder:
    cat_ohe_train =  cat_nom_train_group[['state','marital_status','policy_type','policy','renew_offer_type','sales_channel','vehicle_class']]

    cat_enc = cat_get_enc(cat_ohe_train)
    categorical_train_enc = cat_transform_hot(cat_ohe_train,cat_enc)

    # For the Ordinal values:
    cat_nom_train = cat_nom_train_group[['coverage','employmentstatus','location_code','vehicle_size','education']]

    categorical_train_enc2 = cat_transform_ordinal(cat_nom_train)

    categorical_time_train = day_week_month(categorical_train)

    # Concatenate everything back
    X_tot_train = pd.concat([numerical_train_trans,categorical_train_enc,categorical_train_enc2,categorical_time_train], axis=1) 

    # Do a minmax scaler on everything
    minmax_tr = num_get_tr_power(X_tot_train)
    X_tot_train_trans = num_transform(X_tot_train, minmax_tr)
    
    return X_tot_train_trans, num_tr, cat_enc, minmax_tr 


# This function transforms the data from a dataframe into the X-y using the trained transformers
def Model_transform(df,num_tr,cat_enc,minmax_tr):
    X = df.copy()
    
    # Numerical categorical
    numerical, categorical= cat_num_split(X)
    
    # Numerica Transformation
    numerical_trans = num_transform(numerical, num_tr)
    
    # Group some categories
    cat_nom_group = cat_group_labels(categorical)
    
    # For the OneHotEncoder:
    cat_ohe = cat_nom_group[['state','marital_status','policy_type','policy','renew_offer_type','sales_channel','vehicle_class']]
    categorical_enc = cat_transform_hot(cat_ohe,cat_enc)
    
    # For the Ordinal values:
    cat_nom = cat_nom_group[['coverage','employmentstatus','location_code','vehicle_size','education']]
    categorical_enc2 = cat_transform_ordinal(cat_nom)
    
    # For the time variable
    categorical_time = day_week_month(categorical)
    
    # Concatenate everything back
    X_tot = pd.concat([numerical_trans,categorical_enc,categorical_enc2,categorical_time], axis=1) 

    # Do a minmax scaler on everything
    X_tot_trans = num_transform(X_tot, minmax_tr)
    
    return X_tot_trans
    
# -----------------------------------------------------------------------------------

def Remove_outlayers(df, col, coef):
    # This function returns the index of the outlayers removed on specific column
    data = df.copy()
    
    scaler = PowerTransformer()
    scaler.fit(data[[col]])
    X_normalized_np = scaler.transform(data[[col]])
    normalized_data = pd.DataFrame(X_normalized_np, columns=data[[col]].columns)   
    
    iqr = np.nanpercentile(normalized_data[col],75) - np.nanpercentile(normalized_data[col],25)
    upper_limit = np.nanpercentile(normalized_data[col],75) + coef*iqr

    extraordinary_points = normalized_data[normalized_data[col] > upper_limit]
    print("Points removed:", len(extraordinary_points)) # This checks the number of points that will be removed
    print("Percentage of points removed is:",round(len(extraordinary_points)/len(data)*100,2),"%")  

    X_outliers_dropped = normalized_data[normalized_data[col] <= upper_limit].copy()
    
    return df.iloc[X_outliers_dropped.index]

# Splits -------------------------------------------------------------------------

# X,y split
def tr_ts_split(X,y):  
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)
    X_train.reset_index(inplace=True,drop=True)
    X_test.reset_index(inplace=True,drop=True)
    y_train.reset_index(inplace=True,drop=True)
    y_test.reset_index(inplace=True,drop=True)
    return X_train, X_test, y_train, y_test

# X,y split
def xy_split(df):
    X = df.drop(columns=['total_claim_amount'])
    y = df[['total_claim_amount']]
    return X,y

# Categorical / numerical split
def cat_num_split(df):
    numerical= df.select_dtypes(include=[np.number])
    categorical = df.select_dtypes(exclude=[np.number])
    return numerical, categorical

# Numerical transformations --------------------------------------------------------
def num_get_tr_minmax(df_train):
    return MinMaxScaler().fit(df_train)

def num_get_tr_power(df_train):
    return PowerTransformer().fit(df_train)

def num_get_tr_scale(df_train):
    return StandardScaler().fit(df_train)

def num_transform(df, enc):
    num_test = enc.transform(df)
    return pd.DataFrame(num_test,columns = df.columns)

# Categorical transformations -------------------------------------------------------
def cat_get_enc(df_train):
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df_train)
    return enc

def cat_transform_hot(df,enc):
    onehotlabels = enc.transform(df).toarray()
    df_enc = pd.DataFrame(onehotlabels,columns = enc.get_feature_names_out())
    return df_enc

def cat_transform_ordinal(df):
    df_enc2 = df.copy()
    mapping = {'Basic' : 0, "Extended" : 1, "Premium" : 2}
    df_enc2 = df_enc2.replace({'coverage': mapping})
 
    mapping = {'Other':0,'Employed':1}
    df_enc2 = df_enc2.replace({'employmentstatus': mapping})
    
    mapping = {'Urban':0,'Rural':1,'Suburban':2}
    df_enc2 = df_enc2.replace({'location_code': mapping})
    
    mapping = {'High School or Below':0,'High Education':1}
    df_enc2 = df_enc2.replace({'education': mapping})

    mapping = {'Small':0,'Medsize':1,'Large':2}
    df_enc2 = df_enc2.replace({'vehicle_size': mapping})
    
    
    df_enc2['education'] = pd.to_numeric(df_enc2['education'])
    df_enc2['vehicle_size'] = pd.to_numeric(df_enc2['vehicle_size'])
    df_enc2['employmentstatus'] = pd.to_numeric(df_enc2['employmentstatus'])
    
    return df_enc2

def cat_group_labels(df):
    def Luxury_car(x):
        if ('Luxury' in x) or ('Sports' in x):
            x = "Luxury"
        return x

    def High_education(x):
        if ('Master' in x) or ('Doctor' in x) or ('College' in x) or ('Bachelor' in x) :
            x = "High Education"
        return x

    def Employment(x):
        if ('Medical' in x) or ('Disabled' in x) or ('Retired' in x) or ('Unemployed' in x):
            x = "Other"
        return x

    df2 = df.copy()
    df2['vehicle_class'] = df2['vehicle_class'].apply(Luxury_car)
    df2['education'] = df2['education'].apply(High_education)
    df2['employmentstatus'] = df2['employmentstatus'].apply(Employment)

    return df2

# Time functions -----------------------------------------------------------------------------------------
def day_week_month(df):
    categorical_time_train = df.copy()
    from datetime import datetime
    categorical_time_train = categorical_time_train[['effective_to_date']].copy()
    categorical_time_train['effective_to_date'] = categorical_time_train['effective_to_date'].apply(lambda x: datetime.strptime(x, '%m/%d/%y'))
    categorical_time_train['etd_day'] = categorical_time_train['effective_to_date'].apply(lambda x: x.weekday())
    categorical_time_train['etd_week'] = categorical_time_train['effective_to_date'].apply(lambda x: int(x.strftime("%w")))
    categorical_time_train['etd_month'] = categorical_time_train['effective_to_date'].apply(lambda x: int(x.strftime("%m")))
    categorical_time_train = categorical_time_train.drop(columns=['effective_to_date'])
    return categorical_time_train
