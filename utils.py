import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# Fonction 1 : Relation Magnitude vs Coût des dommages
# ---------------------------------------------------------
def magnitude_vs_damage_cost(df):
    """
    Crée un nuage de points montrant la relation entre la magnitude et le coût des dommages.
    Retourne une figure Plotly.
    """
    df = df.copy()
    clean_df = df[['Mag', 'Damage ($Mil)']].dropna()

    fig = px.scatter(
        clean_df, 
        x='Mag', 
        y='Damage ($Mil)', 
        title='Magnitude vs Coût des Dommages ($Mil)',
        opacity=0.6,
        labels={'Mag': 'Magnitude', 'Damage ($Mil)': 'Coût ($M)'}
    )
    fig.update_layout(template="plotly_white")
    return fig

# ---------------------------------------------------------
# Fonction 2 : Fréquence des séismes par intervalle d'années
# ---------------------------------------------------------
def earthquake_frequency_per_year(df):
    """
    Discrétise les années en intervalles et affiche la fréquence des séismes.
    Retourne une figure Plotly.
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
        title='Fréquence des séismes par intervalle (Discrétisation)',
        labels={'x': 'Intervalle d\'années', 'y': 'Nombre de séismes'}
    )
    fig.update_layout(xaxis_tickangle=-45, template="plotly_white")
    return fig

# ---------------------------------------------------------
# Fonction 3 : Corrélation Profondeur Focale vs Magnitude (AJOUTÉE)
# ---------------------------------------------------------
def focal_depth_vs_magnitude(df):
    """
    Affiche la corrélation entre la profondeur focale et la magnitude.
    Retourne une figure Plotly.
    """
    df = df.copy()
    clean_df = df[['Focal Depth (km)', 'Mag']].dropna()

    fig = px.scatter(
        clean_df,
        x='Focal Depth (km)',
        y='Mag',
        title='Corrélation : Profondeur Focale vs Magnitude',
        opacity=0.5,
        color_discrete_sequence=['orange'], # Couleur orange comme demandé
        labels={'Focal Depth (km)': 'Profondeur Focale (km)', 'Mag': 'Magnitude'}
    )
    fig.update_layout(template="plotly_white")
    return fig

# ---------------------------------------------------------
# Fonction 4 : Clustering Spatial (K-Means)
# ---------------------------------------------------------
def spatial_clustering_kmeans(df, n_clusters=5):
    """
    Applique K-Means sur les coordonnées géographiques pour identifier des zones sismiques.
    Retourne une carte géographique Plotly.
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
# Fonction 5 : Prévision Temporelle (Régression Linéaire)
# ---------------------------------------------------------
def forecast_earthquake_frequency(df, start_year=1950, future_years=10):
    """
    Analyse la tendance historique et prévoit le nombre de séismes futurs.
    Retourne une figure Plotly combinant historique, tendance et prévision.
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
        name='Données Historiques',
        marker=dict(color='blue', opacity=0.5)
    ))

    # Ligne de tendance
    fig.add_trace(go.Scatter(
        x=X.flatten(), y=trend_pred, 
        mode='lines', 
        name='Tendance',
        line=dict(color='green', width=2)
    ))

    # Prévision
    fig.add_trace(go.Scatter(
        x=future_X.flatten(), y=future_pred, 
        mode='lines+markers', 
        name=f'Prévision (+{future_years} ans)',
        line=dict(color='red', dash='dash', width=2)
    ))

    fig.update_layout(
        title='Prévision de la fréquence des séismes (Régression Linéaire)',
        xaxis_title='Année',
        yaxis_title='Nombre de séismes enregistrés',
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig