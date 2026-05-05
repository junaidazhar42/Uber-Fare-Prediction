import streamlit as st
import numpy as np
import joblib
import math
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Fare Estimator",
    page_icon="🚖",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0d0d;
    color: #f0ece4;
}
.hero {
    text-align: center;
    padding: 2.5rem 0 1.2rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0;
    color: #f0ece4;
}
.hero h1 span { color: #f5c542; }
.hero p {
    color: #666;
    font-size: 0.85rem;
    margin-top: 0.4rem;
    font-family: 'DM Mono', monospace;
}
.card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 14px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #f5c542;
    font-family: 'DM Mono', monospace;
    margin-bottom: 1rem;
}
.result-box {
    background: #f5c542;
    color: #0d0d0d;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 1.2rem;
}
.result-box .label {
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
    opacity: 0.55;
}
.result-box .fare {
    font-size: 3.8rem;
    font-weight: 800;
    line-height: 1.1;
}
.result-box .meta {
    font-size: 0.8rem;
    opacity: 0.6;
    margin-top: 0.4rem;
    font-family: 'DM Mono', monospace;
}
.resolved {
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    color: #666;
    margin-top: 0.3rem;
    margin-bottom: 0.3rem;
}
.divider {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 1.4rem 0;
}
div[data-testid="stTextInput"] input {
    background: #1e1e1e !important;
    border: 1px solid #2a2a2a !important;
    color: #f0ece4 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
}
div[data-testid="stNumberInput"] input {
    background: #1e1e1e !important;
    border: 1px solid #2a2a2a !important;
    color: #f0ece4 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
}
div[data-testid="stButton"] > button {
    background: #f5c542 !important;
    color: #0d0d0d !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }
label {
    color: #888 !important;
    font-size: 0.78rem !important;
    font-family: 'DM Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def geocode(location: str):
    """
    Resolve a place name to (lat, lon, display_name) biased toward NYC.
    Returns None if not found.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{location}, New York City",
        "format": "json",
        "limit": 1,
        "countrycodes": "us",
        "viewbox": "-74.259,40.477,-73.700,40.917",  # NYC bounding box
        "bounded": 0,  # soft bias, not hard restriction
    }
    headers = {"User-Agent": "NYC-Fare-Estimator/1.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=5)
        results = r.json()
        if results:
            res = results[0]
            return float(res["lat"]), float(res["lon"]), res["display_name"]
    except Exception:
        pass
    return None


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * math.asin(math.sqrt(a)) * 6371.0


def is_rush_hour(weekday: int, hour: int) -> int:
    return int(weekday < 5 and (7 <= hour <= 9 or 16 <= hour <= 19))


def build_map(pickup: tuple, dropoff: tuple) -> folium.Map:
    """Draw a folium map with two markers and a dashed line between them."""
    mid_lat = (pickup[0] + dropoff[0]) / 2
    mid_lon = (pickup[1] + dropoff[1]) / 2

    m = folium.Map(
        location=[mid_lat, mid_lon],
        zoom_start=12,
        tiles="CartoDB dark_matter",
    )

    folium.Marker(
        location=[pickup[0], pickup[1]],
        tooltip="📍 Pickup",
        icon=folium.Icon(color="green", icon="circle", prefix="fa"),
    ).add_to(m)

    folium.Marker(
        location=[dropoff[0], dropoff[1]],
        tooltip="🏁 Dropoff",
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(m)

    folium.PolyLine(
        locations=[pickup, dropoff],
        color="#f5c542",
        weight=3,
        opacity=0.85,
        dash_array="6 4",
    ).add_to(m)

    return m


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


# ── Load model ─────────────────────────────────────────────────────────────────
try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>NYC <span>Fare</span> Estimator</h1>
  <p>type any address or landmark · we handle the rest</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.warning("⚠️ `model.pkl` not found — run `joblib.dump(best_rf, 'model.pkl')` in your notebook first.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Locations card ─────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">📍 Locations</div>', unsafe_allow_html=True)
pickup_input  = st.text_input("Pickup",  placeholder="e.g. Times Square, Manhattan")
dropoff_input = st.text_input("Dropoff", placeholder="e.g. JFK Airport, Queens")
st.markdown('</div>', unsafe_allow_html=True)

# ── Trip details card ──────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🕐 Trip Details</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
pickup_time     = col1.time_input("Pickup time", value=datetime.now().time())
pickup_date     = col2.date_input("Pickup date", value=datetime.today())
passenger_count = col3.number_input("Passengers", min_value=1, max_value=7, value=1)
st.markdown('</div>', unsafe_allow_html=True)

# ── Estimate button ────────────────────────────────────────────────────────────
if st.button("Estimate Fare →"):

    if not pickup_input or not dropoff_input:
        st.error("Please enter both a pickup and a dropoff location.")
    else:
        with st.spinner("Resolving locations..."):
            pickup_geo  = geocode(pickup_input)
            dropoff_geo = geocode(dropoff_input)

        if pickup_geo is None:
            st.error(f"Could not find **{pickup_input}**. Try a more specific address.")
        elif dropoff_geo is None:
            st.error(f"Could not find **{dropoff_input}**. Try a more specific address.")
        else:
            p_lat, p_lon, p_name = pickup_geo
            d_lat, d_lon, d_name = dropoff_geo

            distance_km = haversine_distance(p_lat, p_lon, d_lat, d_lon)

            if distance_km < 0.1:
                st.error("Pickup and dropoff are less than 100 m apart — please check your inputs.")
            else:
                hour    = pickup_time.hour
                weekday = pickup_date.weekday()
                rush    = is_rush_hour(weekday, hour)

                # Map
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">🗺️ Route</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="resolved">📍 {p_name[:90]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="resolved">🏁 {d_name[:90]}</div>', unsafe_allow_html=True)
                route_map = build_map((p_lat, p_lon), (d_lat, d_lon))
                st_folium(route_map, width=680, height=340, returned_objects=[])
                st.markdown('</div>', unsafe_allow_html=True)

                # Fare result
                if model_loaded:
                    features = np.array([[distance_km, hour, weekday, rush, passenger_count]])
                    fare = max(model.predict(features)[0], 2.50)

                    day_names  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    rush_label = "🔴 Rush hour" if rush else "🟢 Off-peak"

                    st.markdown(f"""
                    <div class="result-box">
                        <div class="label">Estimated Fare</div>
                        <div class="fare">${fare:.2f}</div>
                        <div class="meta">
                            {distance_km:.2f} km &nbsp;·&nbsp;
                            {day_names[weekday]} {pickup_time.strftime('%H:%M')} &nbsp;·&nbsp;
                            {rush_label} &nbsp;·&nbsp;
                            {passenger_count} passenger{'s' if passenger_count > 1 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(f"Computed distance: **{distance_km:.2f} km**. Load `model.pkl` to get a fare estimate.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#333;font-size:0.72rem;font-family:\'DM Mono\',monospace;">'
    'Geocoding via OpenStreetMap Nominatim · Model: Random Forest · NYC'
    '</p>',
    unsafe_allow_html=True
)
