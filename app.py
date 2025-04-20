from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import plotly.express as px

app = Flask(__name__)

# Load models and encoders
with open("crop_model.pkl", "rb") as f:
    crop_model = pickle.load(f)

with open("fertilizer_model.pkl", "rb") as f:
    fert_model = pickle.load(f)

with open("soil_encoder.pkl", "rb") as f:
    le_soil = pickle.load(f)

with open("crop_encoder.pkl", "rb") as f:
    le_crop = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        # Crop Inputs
        crop_inputs = {
            'nitrogen': request.form.get('nitrogen', ''),
            'phosphorus': request.form.get('phosphorus', ''),
            'potassium': request.form.get('potassium', ''),
            'temperature': request.form.get('temperature', ''),
            'humidity': request.form.get('humidity', ''),
            'ph': request.form.get('ph', ''),
            'rainfall': request.form.get('rainfall', '')
        }

        crop_array = np.array([[int(crop_inputs['nitrogen']),
                                int(crop_inputs['phosphorus']),
                                int(crop_inputs['potassium']),
                                float(crop_inputs['temperature']),
                                float(crop_inputs['humidity']),
                                float(crop_inputs['ph']),
                                float(crop_inputs['rainfall'])]])

        crop_result = crop_model.predict(crop_array)[0]

        return render_template('index.html', crop_result=crop_result, crop_inputs=crop_inputs)

    except Exception as e:
        return render_template('index.html', error=str(e))


@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        # Fertilizer Inputs
        fert_inputs = {
            'temp_f': request.form.get('temp_f', ''),
            'humidity_f': request.form.get('humidity_f', ''),
            'moisture': request.form.get('moisture', ''),
            'soil_type': request.form.get('soil_type', ''),
            'crop_type': request.form.get('crop_type', ''),
            'N_f': request.form.get('N_f', ''),
            'P_f': request.form.get('P_f', ''),
            'K_f': request.form.get('K_f', '')
        }

        soil_encoded = le_soil.transform([fert_inputs['soil_type']])[0]
        crop_encoded = le_crop.transform([fert_inputs['crop_type']])[0]

        fert_array = np.array([[float(fert_inputs['temp_f']),
                                float(fert_inputs['humidity_f']),
                                float(fert_inputs['moisture']),
                                soil_encoded,
                                crop_encoded,
                                int(fert_inputs['N_f']),
                                int(fert_inputs['P_f']),
                                int(fert_inputs['K_f'])]])

        fert_result = fert_model.predict(fert_array)[0]

        return render_template('index.html', fert_result=fert_result, fert_inputs=fert_inputs)

    except Exception as e:
        return render_template('index.html', error=str(e))
 

@app.route('/dashboard')
def dashboard():
    crop_df = pd.read_csv('Dataset/Crop_recommendation.csv')
    fert_df = pd.read_csv('Dataset/Fertilizer Prediction.csv')

    crop_count = crop_df['label'].value_counts().reset_index()
    crop_count.columns = ['Crop', 'Count']
    crop_fig = px.bar(crop_count, x='Crop', y='Count', title='Most Common Crops')

    nutrients_avg = crop_df[['N', 'P', 'K']].mean().reset_index()
    nutrients_avg.columns = ['Nutrient', 'Average Value']
    nutrients_fig = px.pie(nutrients_avg, names='Nutrient', values='Average Value', title='Average Soil Nutrients')

    crop_graph = crop_fig.to_html(full_html=False)
    nutrients_graph = nutrients_fig.to_html(full_html=False)

    return render_template('dashboard.html', crop_graph=crop_graph, nutrients_graph=nutrients_graph)

if __name__ == '__main__':
    app.run(debug=True)

