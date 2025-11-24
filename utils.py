import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import kagglehub
from kagglehub import KaggleDatasetAdapter

# ---------------------------------------------------------
# Fonction 1 : Relation Magnitude vs Coût des dommages
# ---------------------------------------------------------
def magnitude_vs_damage_cost(df):
    """
    Creates a scatter plot showing the relationship between magnitude and damage cost.
    Returns a Plotly figure.
    """
    df = df.copy()
    clean_df = df[['Mag', 'Damage ($Mil)']].dropna()

    fig = px.scatter(
        clean_df, 
        x='Mag', 
        y='Damage ($Mil)', 
        title='Magnitude vs Damage Cost ($Mil)',
        opacity=0.6,
        labels={'Mag': 'Magnitude', 'Damage ($Mil)': 'Damage Cost ($M)'}
    )
    fig.update_layout(template="plotly_white")
    return fig

# ---------------------------------------------------------
# Function 2: Frequency of Earthquakes by Year Interval
# ---------------------------------------------------------
def earthquake_frequency_per_year(df):
    """
    Discretizes years into intervals and displays the frequency of earthquakes.
    Returns a Plotly figure.
    """
    df = df.copy()
    frequency_df = df['Year'].astype(int).value_counts().sort_index()

    min_year = int(frequency_df.index.min())
    max_year = int(frequency_df.index.max())
    
    bins = range(min_year, max_year + 5, 200)
    frequency_df_binned = frequency_df.groupby(pd.cut(frequency_df.index, bins), observed=False).sum()

    x_values = frequency_df_binned.index.astype(str)
    y_values = frequency_df_binned.values

    fig = px.line(
        x=x_values, 
        y=y_values, 
        markers=True,
        title='Frequency of Earthquakes by Interval (Discretization)',
        labels={'x': 'Year Interval', 'y': 'Number of Earthquakes'}
    )
    fig.update_layout(xaxis_tickangle=-45, template="plotly_white")
    return fig

# ---------------------------------------------------------
# Function 3: Correlation Focal Depth vs Magnitude (ADDED)
# ---------------------------------------------------------
def focal_depth_vs_magnitude(df):
    """
    Displays the correlation between focal depth and magnitude.
    Returns a Plotly figure.
    """
    df = df.copy()
    clean_df = df[['Focal Depth (km)', 'Mag']].dropna()

    fig = px.scatter(
        clean_df,
        x='Focal Depth (km)',
        y='Mag',
        title='Correlation : Focal Depth vs Magnitude',
        opacity=0.5,
        color_discrete_sequence=['orange'],
        labels={'Focal Depth (km)': 'Focal Depth (km)', 'Mag': 'Magnitude'}
    )
    fig.update_layout(template="plotly_white")
    return fig

# ---------------------------------------------------------
# Function 4: Spatial Clustering (K-Means)
# ---------------------------------------------------------
def spatial_clustering_kmeans(df, n_clusters=5):
    """
    Applies K-Means on geographic coordinates to identify seismic zones.
    Returns a Plotly geographic map.
    """
    geo_data = df[['Latitude', 'Longitude']].dropna()

    scaler = MinMaxScaler()
    geo_data_scaled = scaler.fit_transform(geo_data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    geo_data['Cluster'] = kmeans.fit_predict(geo_data_scaled)

    fig = px.scatter_geo(
        geo_data, 
        lat='Latitude', 
        lon='Longitude', 
        color=geo_data['Cluster'].astype(str),
        title=f'Clustering Spatial des Séismes (K-Means, k={n_clusters})',
        projection="natural earth",
        opacity=0.7,
        labels={'Cluster': 'ID Cluster'}
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig

# ---------------------------------------------------------
# Function 5: Temporal Forecasting (Linear Regression)
# ---------------------------------------------------------
def forecast_earthquake_frequency(df, start_year=1950, future_years=10):
    """
    Analyze the historical trend and forecast the future number of earthquakes.
    Returns a Plotly figure combining historical data, trend, and forecast.
    """
    df_filtered = df[(df['Year'] >= start_year) & (df['Year'] < 2021)].copy()
    yearly_counts = df_filtered['Year'].value_counts().sort_index()
    
    X = yearly_counts.index.values.reshape(-1, 1)
    y = yearly_counts.values

    model = LinearRegression()
    model.fit(X, y)

    last_year = int(X.max())
    future_X = np.arange(last_year + 1, last_year + future_years + 1).reshape(-1, 1)
    future_pred = model.predict(future_X)
    trend_pred = model.predict(X)

    fig = go.Figure()

    # Données historiques
    fig.add_trace(go.Scatter(
        x=X.flatten(), y=y, 
        mode='markers', 
        name='Historical Data',
        marker=dict(color='blue', opacity=0.5)
    ))

    # Trend Line
    fig.add_trace(go.Scatter(
        x=X.flatten(), y=trend_pred, 
        mode='lines', 
        name='Trend',
        line=dict(color='green', width=2)
    ))

    # Prévision
    fig.add_trace(go.Scatter(
        x=future_X.flatten(), y=future_pred, 
        mode='lines+markers', 
        name=f'Forecast (+{future_years} years)',
        line=dict(color='red', dash='dash', width=2)
    ))

    fig.update_layout(
        title='Forecast of Earthquake Frequency (Linear Regression)',
        xaxis_title='Year',
        yaxis_title='Number of Recorded Earthquakes',
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig

# ---------------------------------------------------------
# Function 6: Map of Earthquakes by Country
# ---------------------------------------------------------
def get_country_map(df):

    #1nd Case : CITY, COUNTRY -> COUNTRY
    df['Country'] = df['Location Name'].astype(str).str.split(',').str[-1].str.strip().str.upper() if ":" not in df['Location Name'] else df['Location Name'].astype(str).str.upper()
    
    # 2nd Case : LOCATION NAME: COUNTRY -> COUNTRY
    df['Country'] = df['Country'].astype(str).str.split(':').str[0].str.strip()

    # 3rd Case : COUNTRY1; COUTNRY2 -> COUNTRY1
    df['Country'] = df['Country'].astype(str).str.split(';').str[0].str.strip()

    # Importing external dataset to map states to countries if needed
    df_state = pd.read_csv("state_names.csv")
    list_state = [i.upper() for i in df_state['State'].tolist()]

    # If the 'Country' column contains a state name, replace it with 'United States'
    df['Country'] = df['Country'].apply(lambda x: 'USA' if x in list_state else x)
    df['Country'] = df['Country'].replace({ # Manual corrections
        'GULF OF MEXICO': 'USA'
    })
    # Aggregate data for the map (1 row per country)
    country_stats = df.groupby('Country').agg({
        'Mag': 'mean',      # Average Magnitude
        'Focal Depth (km)': 'mean',          # Average Focal Depth
        'Year': 'count',           # Earthquake count
        'Deaths': 'mean',          # Average Deaths
        'Damage ($Mil)': 'mean'    # Average Damage Cost ($Mil)
    }).reset_index()
    country_stats.rename(columns={'Year': 'Count'}, inplace=True)

    fig = px.choropleth(
        country_stats,
        locations="Country",
        locationmode='country names', # Plotly will try to match English names
        color="Count", # Color depends on the number of earthquakes
        hover_name="Country",
        title="Interactive Map : Select a country to see details",
        color_continuous_scale=px.colors.sequential.OrRd,
        range_color=[0, 600]
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, clickmode='event+select')
    return fig