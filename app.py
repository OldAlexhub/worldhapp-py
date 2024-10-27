from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pymongo
import os
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from prophet import Prophet
from datetime import datetime
from flask_cors import CORS
import tempfile

# Load environment variables.
os.environ['TMPDIR'] = '/dev/shm'
load_dotenv('.env')
today = datetime.today()
# Database connection
mongo_url = os.getenv('MONGO_URL')
client = pymongo.MongoClient(mongo_url)
db = client['test']
collection = db['world']

# Load and preprocess data
data = pd.DataFrame(list(collection.find()))
data_cleaned = data.drop(['_id', 'Country_name', 'year', 'Positive_affect', 'Negative_affect'], axis=1)
data_cleaned['outcome'] = data_cleaned['Life_Ladder'] + data_cleaned['Log_GDP_per_capita'] + \
    data_cleaned['Social_support'] + data_cleaned['Healthy_life_expectancy_at_birth'] + \
    data_cleaned['Freedom_to_make_life_choices'] + data_cleaned['Generosity'] + \
    data_cleaned['Perceptions_of_corruption']

# Quantile-based classification for outcome
def classification(data_scaled):
    quantile_labels = ['Unhappy', 'Moderately Happy', 'Very Happy']
    data_scaled['outcome'] = pd.qcut(
        data_scaled['outcome'],
        q=[0, 0.25, 0.75, 1],  
        labels=quantile_labels
    )
    return data_scaled

data_cleaned = classification(data_cleaned)

# Initialize the scaler and model
scaler = StandardScaler()
features = data_cleaned.drop('outcome', axis=1)
data_scaled = scaler.fit_transform(features)
data_scaled = pd.DataFrame(data_scaled, columns=features.columns)
data_scaled['outcome'] = data_cleaned['outcome']

X = data_scaled.drop('outcome', axis=1)
y = data_scaled['outcome']
model = RandomForestClassifier(class_weight="balanced")
model.fit(X, y)

prophetModel = Prophet()

# Flask app setup
app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
def home():
    return 'Hello World'

@app.route("/status", methods=['POST'])
def get_status():
    data = request.get_json()

    # Prepare user input
    user_input = pd.DataFrame([{
        "Life_Ladder": float(data.get("life_Ladder")),
        "Log_GDP_per_capita": float(data.get("Log_GDP_per_capita")),
        "Social_support": float(data.get("Social_support")) / 100,
        "Healthy_life_expectancy_at_birth": float(data.get("Healthy_life_expectancy_at_birth")) / 100,
        "Freedom_to_make_life_choices": float(data.get("Freedom_to_make_life_choices")) / 100,
        "Generosity": float(data.get("Generosity")) / 100,
        "Perceptions_of_corruption": float(data.get("Perceptions_of_corruption")) / 100
    }])

    # Scale and predict
    user_input_scaled = scaler.transform(user_input)
    y_pred = model.predict(user_input_scaled)
    
    # Generate response message
    prediction = y_pred[0]
    messages = {
        "Very Happy": "🌞 This environment is classified as 'Very Happy,' indicating a robust societal foundation. Research shows that environments with high happiness often have strong economic support, high levels of education, and a significant focus on public health and social programs. Such environments are known to have greater resilience to crises, which contributes to stability in times of global economic uncertainty. Keep fostering these values—it not only benefits current generations but also sets up future generations for success. Fun fact: Iceland, which consistently ranks high in happiness, attributes its stability to strong social cohesion and public welfare systems.",
        
        "Moderately Happy": "😊 Your environment is rated 'Moderately Happy,' suggesting a relatively balanced but not ideal living condition. While there may be reasonable levels of freedom and economic support, areas like income equality, health care access, or government transparency could likely use some improvement. Studies indicate that nations focusing on gradual improvements in these areas see positive shifts in overall well-being. Consider advocating for economic reforms, mental health programs, or community-building initiatives. Finland, often cited as a happiness leader, consistently invests in social equality and accessible healthcare, yielding long-term benefits for societal satisfaction.",
        
        "Unhappy": "🌧 The environment is classified as 'Unhappy.' This rating often correlates with issues like limited access to basic services, economic inequality, or high levels of corruption, which can dampen social trust and the overall quality of life. However, studies show that even modest efforts to improve transparency, healthcare, and education can significantly impact collective well-being. A focus on community engagement, economic opportunities, and fair governance can create positive momentum. For instance, Bhutan, despite economic challenges, emphasizes 'Gross National Happiness,' focusing on holistic well-being, mental health, and cultural values to uplift its communities. Change is possible, but it starts with small, actionable steps in everyday governance and social practices."
    }

    message = messages.get(prediction, "Status unknown.")
    
    # Return JSON Response
    return jsonify({"prediction": prediction, "message": message})

@app.route("/predict", methods=['POST'])
def get_predictions():
    datas = request.get_json()
    user_country = datas.get("country")
    
    # Use the original 'data' DataFrame to filter by 'Country_name'
    if user_country not in data['Country_name'].values:
        return jsonify({"error": "Country not found in the dataset."}), 404

    prophetData = data[data['Country_name'] == user_country]

    def prepare_prophet_data(data, target_column):
        prophet_data = data[['year', target_column]].rename(columns={'year': 'ds', target_column: 'y'})
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'], format='%Y')
        return prophet_data

    def prophet_pipeline(data, periods=5, freq='Y'):
        model = Prophet()
        model.fit(data)
        future = model.make_future_dataframe(periods=periods, freq=freq)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat']]

    # Prepare data for each variable
    forecast1 = prophet_pipeline(prepare_prophet_data(prophetData, 'Life_Ladder'))
    forecast2 = prophet_pipeline(prepare_prophet_data(prophetData, 'Log_GDP_per_capita'))
    forecast3 = prophet_pipeline(prepare_prophet_data(prophetData, 'Social_support'))
    forecast4 = prophet_pipeline(prepare_prophet_data(prophetData, 'Healthy_life_expectancy_at_birth'))
    forecast5 = prophet_pipeline(prepare_prophet_data(prophetData, 'Freedom_to_make_life_choices'))
    forecast6 = prophet_pipeline(prepare_prophet_data(prophetData, 'Generosity'))
    forecast7 = prophet_pipeline(prepare_prophet_data(prophetData, 'Perceptions_of_corruption'))

    # Merge all forecast data
    forecast1.name = 'Life_Ladder'
    forecast2.name = 'Log_GDP_per_capita'
    forecast3.name = 'Social_support'
    forecast4.name = 'Healthy_life_expectancy_at_birth'
    forecast5.name = 'Freedom_to_make_life_choices'
    forecast6.name = 'Generosity'
    forecast7.name = 'Perceptions_of_corruption'

    def merge_forecasts(*datasets):
        merged_data = datasets[0][['ds', 'yhat']].rename(columns={'yhat': datasets[0].name})
        for dataset in datasets[1:]:
            merged_data = pd.merge(
                merged_data,
                dataset[['ds', 'yhat']].rename(columns={'yhat': dataset.name}),
                on='ds',
                how='outer'
            )
        return merged_data

    all_forecasts = merge_forecasts(forecast1, forecast2, forecast3, forecast4, forecast5, forecast6, forecast7)

    all_forecasts = all_forecasts[all_forecasts['ds'] > today]
    # Convert merged forecasts to JSON format and return
    result = all_forecasts.to_dict(orient='records')
    return jsonify({"country": user_country, "forecasts": result})

