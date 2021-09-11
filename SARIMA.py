import pandas as pd #DataFrame functions
from numpy import array # Array
import numpy as np
from sklearn.impute import KNNImputer #Imputaion
from statsmodels.tsa.statespace.sarimax import SARIMAX #SARIMAX
import pickle #To create pickle file

data = pd.read_csv('Beds_Occupied.csv')
data['collection_date'] = pd.to_datetime(data['collection_date'], format='%d-%m-%Y')
def avail(beds):
    available_beds = 900-beds
    return available_beds

data['availability']=data['Total Inpatient Beds'].apply(avail)
data = data.drop('Total Inpatient Beds', axis = 1)
range_dates = pd.date_range(start=data.collection_date.min(), end=data.collection_date.max())
missing_dates = range_dates.difference(data['collection_date'])
data = data.set_index('collection_date').reindex(range_dates).rename_axis('date').reset_index()
imputer = KNNImputer(n_neighbors=3)
df = imputer.fit_transform(data[['availability']])
df = np.round(df,0)
avail_df = pd.DataFrame(df, columns=['Availability']) #Array ouput of imputer is converted in to a DataFrame
data = data.assign(availability=avail_df['Availability']) #Then replacing the imputed column to our original dataframe
data = data.set_index('date')
SARIMA_model = SARIMAX(data.availability, order=(13,0,9), seasonal_order=(0,0,4,12)).fit()
pickle.dump(SARIMA_model,open('forecast_model.pkl','wb'))