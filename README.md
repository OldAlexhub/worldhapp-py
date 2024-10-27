# World Happiness Prediction API

The World Happiness Prediction API is a machine learning-based application designed to analyze and predict happiness indicators for different countries based on socio-economic data. This project employs a combination of supervised learning models and time-series forecasting to provide a robust set of insights into the global happiness landscape. The system supports country-level happiness classifications, with forecasted trends in key happiness indicators over time, which makes it suitable for applications in sociology, economics, and policy-making.

# Project Overview

The World Happiness Prediction API aims to help researchers, policymakers, and developers analyze happiness levels across countries and track changes in key happiness indicators over time. Happiness is measured using both qualitative classification and quantitative forecasting, leveraging data related to social support, GDP per capita, life expectancy, and more.

## Key objectives include:

* **Quantifying Happiness:** Providing an approximate "happiness score" for each country.
* **Tracking Trends:** Identifying how happiness factors change over time.
* **Supporting Interventions:** Offering insights to support policies and social initiatives.

# Features

* **Country-based Happiness Classification:** Uses a RandomForestClassifier to classify countries as "Very Happy," "Moderately Happy," or "Unhappy" based on recent data.
* **Happiness Forecasting:** Employs Facebook's Prophet model to predict future trends in happiness-related indicators like social support and freedom.
* **Detailed Insights:** Provides actionable feedback tailored to each country’s classification.
* **RESTful API:** Offers a robust interface to interact with the data, accessible via standard HTTP requests, making it suitable for integration into web applications and dashboards.


# Architecture

The application is built using Python and Flask, with MongoDB as the primary database for data storage. Core components include:

* **MongoDB Database:** Stores country-level happiness data.
* **Data Processing:** Uses pandas for data handling and cleaning.
* **Classification Model:** Utilizes RandomForestClassifier for happiness classification.
* **Time-Series Forecasting:** Uses Prophet to forecast happiness indicators.
* **API:** Built with Flask and Flask-CORS, supporting JSON-based requests and responses.

