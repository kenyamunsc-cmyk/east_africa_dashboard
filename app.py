import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import requests
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

st.set_page_config(page_title="Eastern Africa Climate–Health AI Dashboard", layout="wide")

# --------------------- NASA POWER API ---------------------
def fetch_nasa_power(lat, lon, start_date, end_date):
    base = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "latitude": lat,
        "longitude": lon,
        "parameters": "T2M,PRECTOT",
        "format": "JSON",
        "user": "dashboard_user"
    }
    r = requests.get(base, params=params)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    df = pd.DataFrame(data).T.reset_index()
    df.rename(columns={"index": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df

# --------------------- WHO GHO API ---------------------
def fetch_who_gho(indicator="WHS4_159", iso="KEN"):
    url = f"https://ghoapi.azureedge.net/api/{indicator}?$filter=SpatialDim eq '{iso}'"
    r = requests.get(url)
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("value", []))
    if "TimeDim" in df:
        df["date"] = pd.to_datetime(df["TimeDim"], format="%Y")
        df.rename(columns={"NumericValue": "disease_cases"}, inplace=True)
        df = df[["date", "disease_cases"]]
    return df

# --------------------- LOAD GEOJSON ---------------------
@st.cache_data
def load_geo():
    return gpd.read_file("east_africa_admin1.geojson")

gdf = load_geo()

# --------------------- USER INPUT ---------------------
regions = gdf["NAME_1"].unique().tolist()
selected_region = st.sidebar.selectbox("Select Region", regions)

coords = gdf[gdf["NAME_1"] == selected_region].centroid.iloc[0]
lat, lon = coords.y, coords.x

start_date = datetime.now() - timedelta(days=60)
end_date = datetime.now()

# --------------------- FETCH DATA ---------------------
st.info(f"Fetching NASA POWER climate data for {selected_region}…")
climate_df = fetch_nasa_power(lat, lon, start_date, end_date)

st.info(f"Fetching WHO GHO health data for region country…")
health_df = fetch_who_gho()

# --------------------- MERGE & CLEAN ---------------------
df = climate_df.merge(health_df, on="date", how="left")
df.fillna(method="ffill", inplace=True)
df.fillna(method="bfill", inplace=True)

# --------------------- PROPHET FORECAST ---------------------
forecast_input = df[["date", "disease_cases"]].rename(columns={"date": "ds", "disease_cases": "y"})
model = Prophet()
model.fit(forecast_input)
future = model.make_future_dataframe(periods=14)
forecast = model.predict(future)

# --------------------- RISK INDEX ---------------------
scaler = MinMaxScaler()
df_norm = scaler.fit_transform(df[["T2M", "PRECTOT", "disease_cases"]])
df[["temp_n", "rain_n", "cases_n"]] = df_norm
df["risk_index"] = df[["temp_n", "rain_n", "cases_n"]].mean(axis=1)

# --------------------- DASHBOARD ---------------------
st.title("🌍 Eastern Africa Climate–Health–AI Dashboard")
st.subheader(f"Region: {selected_region}")

col1, col2, col3 = st.columns(3)
col1.metric("Avg Temp (°C)", round(df["T2M"].mean(), 2))
col2.metric("Avg Rainfall (mm)", round(df["PRECTOT"].mean(), 2))
col3.metric("Avg Disease Cases", round(df["disease_cases"].mean(), 2))

# Time series
st.header("📈 Climate & Health Trends")
fig_trend = px.line(df, x="date", y=["T2M", "PRECTOT", "disease_cases"], title="Climate vs Health")
st.plotly_chart(fig_trend, use_container_width=True)

# Forecast plot
st.header("🤖 Prophet Disease Forecast (14 days)")
fig_f = px.line(forecast, x="ds", y=["yhat", "yhat_lower", "yhat_upper"], title="Forecast")
st.plotly_chart(fig_f, use_container_width=True)

# Choropleth map
st.header("🗺️ Regional Climate–Health Risk Index")
map_df = gdf.merge(df.tail(1), left_on="NAME_1", right_on="NAME_1", how="left")
fig_map = px.choropleth_mapbox(
    map_df,
    geojson=map_df.geometry.__geo_interface__,
    locations=map_df.index,
    color="risk_index",
    hover_name="NAME_1",
    mapbox_style="open-street-map",
    center={"lat": 0.5, "lon": 37},
    zoom=4,
)
st.plotly_chart(fig_map, use_container_width=True)

# Download button
st.subheader("📥 Download Data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, "east_africa_climate_health.csv")

st.success("Dashboard loaded successfully!")
