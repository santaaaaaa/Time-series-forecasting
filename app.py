from flask import Flask,render_template,request
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import plotly

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

