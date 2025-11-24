# Seismic Analysis & Forecasting Dashboard

This project is an interactive Data Science application developed with **Python Dash**. It allows users to explore global seismic activity, identify high-risk zones using Machine Learning, and visualize detailed statistics by country.

## 1. Features

* **Interactive Map (Drill-down)**: Global choropleth map (logarithmic scale). Click on any country to instantly display specific statistics.
* **General Analysis**: Study of frequencies, correlation between depth and magnitude, and financial impact.
* **Spatial Clustering (K-Means)**: Automatic grouping of epicenters to visualize active tectonic faults without prior geographical bias.
* **Temporal Forecasting**: Linear Regression model to estimate the trend of earthquake frequency over the next 10 years.

## 2. Technologies

* **Python 3.x**
* **Dash & Plotly** (Frontend and Visualization)
* **Pandas** (Data Manipulation)
* **Scikit-Learn** (Machine Learning Models)

## 3. File Structure

* `dashboard.py`: The main script that launches the Web server and manages the interface.
* `utils.py`: Contains business logic, cleaning functions, Plotly charts, and ML algorithms.
* `earthquakes.csv`: The source dataset (USGS).
* `Jego_Earthquakes.ipynb`: Jupyter Notebook for initial exploration and research.

## 4. Installation and Launch

A.  **Install the required libraries**:
    ```bash
    pip install dash pandas plotly scikit-learn
    ```

B.  **Launch the application**:
    Ensure you are in the project folder, then run:
    ```bash
    python dashboard.py
    ```

3.  **Open the Dashboard**:
    Go to `http://127.0.0.1:8051/` in your browser.

## 5. Authors

* **Jego Thomas**
