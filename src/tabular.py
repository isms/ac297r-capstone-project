import pandas as pd
import numpy as np
import seaborn as sns

def clean_categorical(df, clm, threshold):
    value_counts = df[clm].value_counts()
    keep = value_counts[value_counts > threshold].index
    df[clm] = df[clm].apply(lambda x: x if x in keep else 'other')

    one_hot = pd.get_dummies(df[clm])
    df = df.join(one_hot)
    df = df.drop([clm, 'other'],axis = 1)
    return df

def clean(df):
    # drop all the useless data
    good_columns = ['BLDG_VAL', 'LAND_VAL', 'OTHER_VAL',
           'LOT_SIZE', 'LS_DATE', 'LS_PRICE', 'USE_CODE',
           'ZONE', 'YEAR_BUILT', 'BLD_AREA',
           'UNITS', 'RES_AREA', 'STYLE', 'STORIES', 'NUM_ROOMS', 'LOT_UNITS', 'MBL']
    df = df[good_columns]

    # convert fake missing values (0s and 1s) to NaN
    zero_nan_columns = ['BLDG_VAL', 'LAND_VAL', 'OTHER_VAL', 'LS_PRICE', 'LOT_SIZE', 'STORIES', 'NUM_ROOMS','YEAR_BUILT','BLD_AREA','UNITS','RES_AREA','LOT_UNITS']
    df[zero_nan_columns] = df[zero_nan_columns].replace(0, np.nan)

    one_nan_columns = ['LS_PRICE']
    df[one_nan_columns] = df[one_nan_columns].replace(1, np.nan)

    # add is missing columns (columns chosen based on correlations of missing values)
    is_missing_column = ['BLD_AREA','BLDG_VAL','LOT_SIZE']
    is_missing_column_names = [clm+'_MISSING' for clm in is_missing_column]
    df[is_missing_column_names] = df[is_missing_column].isna().astype('int32')

    # convert last sale date to just year
    df['LS_YEAR'] = df['LS_DATE'].apply(lambda x: x.year)
    df = df.drop('LS_DATE', axis = 1)

    #clean use codes
    df = clean_categorical(df, 'USE_CODE', 400)

    #clean ZONE
    df = clean_categorical(df, 'ZONE', 100)

    # clean STYLE
    style_mapping = {
     'MULTIFAMILY': ['Condominium', 'Apartments', 'Conventional-Apts','Mansard-Apts','Mid rise','Low rise','High Rise Apt','Mid Rise Apartments','Victorian-Apts'],
     '3_FLOORS':['3-Decker', 'Three decker','3-Decker-Apts'],
     'TRIPLEX':['3 fam Conv'],
     '2_FLOOR':['2-Decker', 'Two decker','2-Decker-Apts'],
     'DUPLEX':['Family Duplex', '2 Fam Conv', 'Duplex', 'Two Family','Two Family-Apts','Family Duplex-Apts'],
     'CONVENTIONAL':['Conventional', 'Fam Conv', 'Mansard'],
     'OTHER':['Stores/Apt Com'],
     'ROW_HOME':['Row End', 'Row Mid','Row End-Apts', 'Row Mid-Apts','Row Middle'],
     'TOWNHOUSE':['Townhouse end','Townhouse middle','Townhouse'],
     'VICTORIAN':['Victorian'],
     'COTTAGE':['Cottage Bungalow','Cottage']
    }

    style_mapping_long = {}
    for style_category in style_mapping:
        for style in style_mapping[style_category]:
            style_mapping_long[style] = style_category

    df['STYLE'] = df['STYLE'].apply(lambda x: style_mapping_long[x])

    # impute missing values with mean
    df = df.fillna(df.mean())
    df.head()

    # save
    df.to_csv('./data/residence_addresses_googlestreetview_clean.csv')

if __name__ == '__main__':
    #load assessor data from excel spreadsheet
    df = pd.read_excel('./data/residence_addresses_googlestreetview.xlsx',
                       sheet_name='residential by unit')
    clean(df)
