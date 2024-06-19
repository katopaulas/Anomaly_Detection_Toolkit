import pandas as pd
import numpy as np
import os 

DATA_PATH = './data'

def extract_files_to_df():
    APP_REC_FILE = os.path.join(DATA_PATH,'application_record.csv')
    CR_REC_FILE = os.path.join(DATA_PATH,'credit_record.csv') 
    appdf = pd.read_csv(APP_REC_FILE)
    creditdf = pd.read_csv(CR_REC_FILE)   
    # DATA EXPLORATION

    # appdf.head(10)
    # appdf.shape
    # appdf.keys()
    # appdf.info()
    # appdf.describe().T
    # appdf.nunique()
    # appdf.isnull().sum()
    ##

    # creditdf.head(10)
    # creditdf.shape
    # creditdf.keys()
    # creditdf.info()
    # creditdf.describe().T
    # creditdf.nunique()
    # creditdf.isnull().sum()
    ##
    return appdf, creditdf


def prepare_data(trainable_df,creditdf):

    ############################################ DATA
    # Data preparation
    # Transform DAYS_BIRTH column to AGE column
    trainable_df['AGE'] = trainable_df['DAYS_BIRTH'].abs() // 365.25
    trainable_df.drop('DAYS_BIRTH',axis=1,inplace=True)

    # Converting employed days of unemployed people to 0
    trainable_df.loc[(trainable_df['DAYS_EMPLOYED'] > 0), 'DAYS_EMPLOYED'] = 0

    # Insert 'EMPLOYED_YEARS' column and drop 'DAYS_EMPLOYED' column
    trainable_df['EMPLOYED_YEARS'] = trainable_df['DAYS_EMPLOYED'].abs() // 365.25
    trainable_df.drop('DAYS_EMPLOYED',axis=1,inplace=True)


    #Drop duplicate IDs and not usefull flag_mobil (unique value is only 1)
    trainable_df.drop_duplicates(subset = ['ID'], inplace=True)
    trainable_df.drop(['FLAG_MOBIL'], axis = 1, inplace=True)

    trainable_df['OCCUPATION_TYPE'] = trainable_df['OCCUPATION_TYPE'].fillna(value = 'Occupation Not Identified')
    trainable_df.loc[(trainable_df['NAME_INCOME_TYPE'] == 'Pensioner') & (trainable_df['OCCUPATION_TYPE'] == 'Occupation Not Identified'), 'OCCUPATION_TYPE'] = 'Retired'

    creditdf[['ID', 'MONTHS_BALANCE']].drop_duplicates()

    convert_to = {'C' : 'Good_Debt', 'X' : 'Good_Debt', '0' : 'Good_Debt', '1' : 'Neutral_Debt', '2' : 'Neutral_Debt', '3' : 'Bad_Debt', '4' : 'Bad_Debt', '5' : 'Bad_Debt'}
    creditdf.replace({'STATUS' : convert_to}, inplace = True)
    creditdf = creditdf.value_counts(subset = ['ID', 'STATUS']).unstack(fill_value = 0)

    creditdf.loc[(creditdf['Good_Debt'] > creditdf['Neutral_Debt']), 'CREDIT_CARD_APPROVAL_STATUS'] = 1
    creditdf.loc[(creditdf['Good_Debt'] > creditdf['Bad_Debt']), 'CREDIT_CARD_APPROVAL_STATUS'] = 1
    creditdf.loc[(creditdf['Neutral_Debt'] > creditdf['Bad_Debt']), 'CREDIT_CARD_APPROVAL_STATUS'] = 1
    creditdf.loc[(creditdf['Neutral_Debt'] > creditdf['Good_Debt']), 'CREDIT_CARD_APPROVAL_STATUS'] = 0
    creditdf.loc[(creditdf['Bad_Debt'] > creditdf['Good_Debt']), 'CREDIT_CARD_APPROVAL_STATUS'] = 0
    creditdf.loc[(creditdf['Bad_Debt'] > creditdf['Neutral_Debt']), 'CREDIT_CARD_APPROVAL_STATUS'] = 0
    creditdf['CREDIT_CARD_APPROVAL_STATUS'] = creditdf['CREDIT_CARD_APPROVAL_STATUS'].astype('int')
    creditdf.drop(['Bad_Debt', 'Good_Debt', 'Neutral_Debt'], axis = 1, inplace = True)
    creditdf['CREDIT_CARD_APPROVAL_STATUS'].value_counts()

    encoded = pd.get_dummies(trainable_df[['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE']], prefix=['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE'], drop_first = True)
    trainable_df = trainable_df.join(encoded)
    #Drop the non-encoded columns
    trainable_df.drop(['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE'], axis = 1, inplace = True)

    merged_df = trainable_df.reset_index().merge(creditdf, on = 'ID').set_index('index')
    merged_df.to_csv(os.path.join(DATA_PATH,'final_dataset.csv'))
    return merged_df

def get_data():
    if os.path.isfile(os.path.join(DATA_PATH,'final_dataset.csv')):
        return pd.read_csv(os.path.join(DATA_PATH,'final_dataset.csv'))
    trainable_df,creditdf = extract_files_to_df()
    trainable_df = prepare_data(trainable_df,creditdf)
    return trainable_df


############################################ DATA