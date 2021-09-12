from flask import Flask,render_template,request
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import plotly

import numpy as np
from sklearn.impute import KNNImputer #Imputaion

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

app = Flask(__name__)
model = pickle.load(open('forecast_model.pkl','rb'))

@app.route('/')
def input():
    return render_template('input.html')

@app.route('/output', methods=['POST'])
def output():
    x = int(request.form['days'])
    available_beds = model.forecast(x)
    df = available_beds.to_frame('beds').reset_index().rename(columns={'index':'dates'})
    dates = []
    beds = []
    for i in range(0,len(df)):
        d = str(df.dates[i])[:10]
        dates.append(d)
        b = int(df.beds[i])
        beds.append(b)
    dic = {dates[i]: beds[i] for i in range(len(dates))}
    fig = go.Figure(data=[go.Scatter(x=df.dates, y=df.beds,)])
    graphJSON = json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('output.html', dic=dic, ndays=x, graphJSON=graphJSON)

    
if __name__ == '__main__':
    app.run(debug=True)

