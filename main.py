mport os
import pandas as pd
from ml import ML




# Path to .CSV files: Read only
DATA_PATH = 'csv/'
# Path to your generated .csv files: Save your .csv files here
DELIVERABLE_PATH = 'out/'
# Path to models output: Save your models here
MODEL_PATH = 'models/'

import warnings
warnings.filterwarnings("ignore")

##### MAIN SCRIPT

        # DATA EXPLORATION 


if __name__ == '__main__':

    # DATA EXPLORATION PHASE
    ####

    # Load all files from the data path
    # MAIN_PATH = os.getcwd()
    # files_main = os.path.join( MAIN_PATH,DATA_PATH,"main.csv")

    # df_main = pd.read_csv(files_main)

    ### reduce data for prototyping
    #dfm_n = df_main.head(5000)
    #dfm_n.to_csv('csv_cropped/main.csv',index=False)

    #df_main = pd.read_csv('csv_cropped/main.csv')



    ### Check nans/ counts / column names
    #count_nans_per_cols_m = df_main.isna().sum()
    #print(count_nans_per_cols_m)

    ### Statistics per columns
    #print(df_main.describe())
    #df_animals.corr().to_csv('corr_main.csv')
    
    ### Gather distributions of sightings
    # df_ufoTrue = df_main.loc[df_main['target'] == True] ### TODO : global or input

    # #df.to_csv('csv_cropped/ufo_main.csv',index=False)


    # Actual prepro, prepare data # Trainable_df ~O(N)
    #DF_with_Y = trainable_df(df_main) # TODO TRANSLATE TO MODERN
    #DF_with_Y = pd.read_csv('prepro_all.csv')
    #DF_with_Y = DF_with_Y.drop('Object_animated_n',1).fillna(0)
    #DF_with_Y['No_Farmers'] = df_main['No_Farmers'].values
    #DF_with_Y['Y'] = df_main['Is_UFO'].astype(int)
    #DF_with_Y['is_Raining'] = df_main['is_Raining'].astype(int)
    #DF_with_Y.corr().to_csv('corr_prepro.csv')