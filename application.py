from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app=application

## import ridge regressor and standard scaker pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
Standard_Scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        input_df = pd.DataFrame([{
        'Temperature': Temperature,
        'RH': RH,
        'Ws': Ws,
        'Rain': Rain,
        'FFMC': FFMC,
        'DMC': DMC,
        'ISI': ISI,
        'Classes': Classes,
        'Region': Region
        }])

    
        new_data_scaled = Standard_Scaler.transform(input_df)
        
        result=ridge_model.predict(new_data_scaled)

        return render_template("home.html",results=result[0])


    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)