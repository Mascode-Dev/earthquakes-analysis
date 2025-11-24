import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Importing functions from our utils.py file
# Make sure utils.py is in the same folder as this dashboard.py
from utils import (
    magnitude_vs_damage_cost,
    earthquake_frequency_per_year,
    spatial_clustering_kmeans,
    forecast_earthquake_frequency,
    focal_depth_vs_magnitude,
    get_country_map
)

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================

try:
    # Loading the CSV
    df = pd.read_csv('earthquakes.csv')
    
    # Small preventive cleaning step to ensure the 'Year' column exists
    # (Based on the logic seen in your notebook)
    if 'Year' not in df.columns and 'Date' in df.columns:
        # We try to convert the date to extract the year
        df['Date_Parsed'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year'] = df['Date_Parsed'].dt.year

except FileNotFoundError:
    print("ERROR: The file 'earthquakes.csv' is not found.")
    # Creating an empty dataframe to prevent the script from crashing immediately
    df = pd.DataFrame()

# =============================================================================
# 2. GENERATION OF GRAPHS
# =============================================================================
# We call the functions from utils.py that return Plotly 'fig' objects

if not df.empty:
    # Tab 1: General Analyses
    fig_freq = earthquake_frequency_per_year(df)
    fig_cost = magnitude_vs_damage_cost(df)
    fig_depth = focal_depth_vs_magnitude(df)

    # Tab 2: Advanced Analyses (ML & Forecasting)
    fig_forecast = forecast_earthquake_frequency(df, start_year=1950, future_years=10)
    fig_cluster = spatial_clustering_kmeans(df, n_clusters=5)
else:
    # Fallback if no data
    fig_freq = fig_cost = fig_depth = fig_forecast = fig_cluster = px.scatter(title="No data available")

# =============================================================================
# 3. DASHBOARD LAYOUT
# =============================================================================

app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server
fig_choropleth = get_country_map(df)
app.layout = html.Div([
    
    # --- Header ---
    html.Div([
        html.H1("Seismic Analysis Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
        html.H4("Data Science Project - Analysis & Predictions", style={'textAlign': 'center', 'color': '#7f8c8d'}),
        html.P("Team Members: Jego, Thomas (Team Leader)", style={'textAlign': 'center', 'fontWeight': 'bold'}),
    ], style={'padding': '20px', 'backgroundColor': '#f4f6f7', 'marginBottom': '20px'}),

    # --- Content with Tabs ---
    dcc.Tabs([
        
        # --- TAB 1: Overview ---
        dcc.Tab(label='Overview & Correlations', children=[
            html.Div([
                
                # Line 1: Frequency
                html.Div([
                    dcc.Graph(figure=fig_freq)
                ], style={'width': '98%', 'display': 'inline-block', 'padding': '10px'}),
                
                html.Hr(),

                # Line 2: Correlations (Side by Side)
                html.Div([
                    # Cost vs Mag Chart
                    html.Div([
                        dcc.Graph(figure=fig_cost)
                    ], style={'width': '48%', 'display': 'inline-block'}),

                    # Depth vs Mag Chart
                    html.Div([
                        dcc.Graph(figure=fig_depth)
                    ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
                ]),

                html.Hr(),

                # Zone de la carte
                dcc.Graph(
                    id='world-map',  # ID important pour le callback
                    figure=fig_choropleth
                ),

                html.Hr(),

                # Zone d'affichage des détails (vide au début)
                html.Div(id='country-details-container', style={'padding': '20px', 'backgroundColor': '#f9f9f9'})
            ])
        ]),

        # --- TAB 2 : Machine Learning ---
        dcc.Tab(label='Machine Learning & Forecasting', children=[
            html.Div([
                html.H3("Advanced Modeling", style={'textAlign': 'center', 'marginTop': '20px'}),
                
                # Forecasting
                html.Div([
                    dcc.Graph(figure=fig_forecast),
                    html.P("This graph uses Linear Regression on historical data (1950-2020) to predict the trend for the next 10 years.", 
                           style={'textAlign': 'center', 'fontStyle': 'italic', 'color': '#555'})
                ], style={'padding': '20px', 'borderBottom': '1px solid #ddd'}),
                
                # Clustering
                html.Div([
                    dcc.Graph(figure=fig_cluster),
                    html.P("The K-Means algorithm groups earthquakes by geographic proximity to automatically identify active seismic zones.", 
                           style={'textAlign': 'center', 'fontStyle': 'italic', 'color': '#555'})
                ], style={'padding': '20px'})
            ])
        ])
    ])
], style={'fontFamily': 'Arial, sans-serif'})


# --- Callback / (When a country is clicked on the map) ---
@app.callback(
    Output('country-details-container', 'children'),
    Input('world-map', 'clickData')
)
def display_country_data(clickData):
    if clickData is None:
        return html.H3("Select a country to see details.")

    # Get country name from clickData
    country_name = clickData['points'][0]['location']
    
    # Filter data for the selected country
    df_country = df[df['Country'] == country_name]
    
    # Compute stats (KPIs)
    count = len(df_country)
    avg_mag = df_country['Mag'].mean()
    max_mag = df_country['Mag'].max()
    
    # Country-specific chart
    fig_country_trend = px.histogram(df_country, x='Year', title=f"Earthquakes history for : {country_name}")

    # Return the HTML content
    return html.Div([
        html.H2(f"Details for : {country_name}", style={'color': 'darkblue'}),
        
        # KPIs
        html.Div([
            html.Div([html.H4("Number of Earthquakes"), html.P(str(count), style={'fontSize': '24px'})], className='three columns'),
            html.Div([html.H4("Average Magnitude"), html.P(f"{avg_mag:.2f}/10", style={'fontSize': '24px'})], className='three columns'),
            html.Div([html.H4("Largest Earthquake"), html.P(f"{max_mag:.2f}/10", style={'fontSize': '24px'})], className='three columns'),
            html.Div([html.H4("Average Deaths / earthquake"), html.P(f"{df_country['Deaths'].mean():.2f}", style={'fontSize': '24px'})], className='three columns'),
            html.Div([html.H4("Average Damage Cost / earthquake"), html.P(f"{df_country['Damage ($Mil)'].mean():.2f} M$", style={'fontSize': '24px'})], className='three columns')
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'textAlign': 'center'}),
        
        html.Hr(),
        
        # Country-specific chart
        dcc.Graph(figure=fig_country_trend)
    ])

# =============================================================================
# 4. SERVER LAUNCH
# =============================================================================

if __name__ == '__main__':
    print("Lancement du serveur Dash...")
    app.run(debug=True, port=8051)