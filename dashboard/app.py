"""
Pune Traffic & Congestion — Unified Interactive Dashboard

Five pages, one app:
    1. Real-Time Monitoring     - current speed, travel time, incidents, weather
    2. Traffic Prediction       - future speed / travel time / congestion (ML)
    3. Risk & Incident Forecast - incident probability, closure risk, risk score
    4. Traffic Analytics        - historical trends, hotspots, weather impact
    5. Model Performance        - MAE, RMSE, Accuracy, Precision, Recall, F1, features

Reads live/history straight from Postgres (traffic_snapshots, live_predictions,
locations) and reads spark_jobs/train_model.py's metrics.json for the
Model Performance page. No page requires all upstream services to be running -
each one degrades gracefully with a clear "here's what to start" message
instead of crashing.

Run (see docker-compose.yml `dashboard` service):
    streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0
"""

import json
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
from streamlit_autorefresh import st_autorefresh

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", "postgresql://traffic_user:traffic_pass@localhost:5432/traffic_db"
)
REFRESH_SECONDS = int(os.environ.get("DASHBOARD_REFRESH_SECONDS", "15"))
METRICS_PATH = os.environ.get("METRICS_PATH", "data/models/congestion_rf/metrics.json")

COLORS = {
    "low": "#22c55e",
    "moderate": "#f59e0b",
    "high": "#ef4444",
    "bg": "#0f172a",
    "card": "#1e293b",
    "accent": "#38bdf8",
}
LABEL_ORDER = ["low", "moderate", "high"]
# Rough expected speed_ratio per predicted class - used only to translate a
# classification output into an approximate future speed / travel time for
# the Traffic Prediction page. These are display-layer estimates, not a
# regression model - flagged as such in the UI.
EXPECTED_RATIO = {"low": 0.9, "moderate": 0.6, "high": 0.35}

st.set_page_config(
    page_title="Pune Traffic Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
.stApp { background: radial-gradient(circle at top left, #111a2e 0%, #0b1120 55%, #070b16 100%); }
section[data-testid="stSidebar"] { background: #0b1120; border-right: 1px solid #1e293b; }
h1, h2, h3 { color: #e2e8f0 !important; }
p, span, label, .stMarkdown { color: #cbd5e1; }

.hero {
    padding: 22px 28px; border-radius: 16px; margin-bottom: 18px;
    background: linear-gradient(120deg, #0ea5e9 0%, #6366f1 55%, #a855f7 100%);
    box-shadow: 0 8px 30px rgba(56,189,248,0.25);
}
.hero h1 { color: white !important; margin: 0; font-size: 28px; }
.hero p { color: #e0f2fe; margin: 4px 0 0 0; font-size: 14px; }

.metric-card {
    background: linear-gradient(145deg, #1e293b, #16213a);
    border: 1px solid #2d3b55; border-radius: 14px; padding: 16px 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25); height: 100%;
}
.metric-card .label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-card .value { font-size: 26px; font-weight: 700; color: #f1f5f9; margin-top: 4px; }
.metric-card .sub { font-size: 12px; color: #64748b; margin-top: 2px; }

.badge { display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; }
.badge-low { background: rgba(34,197,94,0.15); color:#22c55e; border:1px solid rgba(34,197,94,0.4); }
.badge-moderate { background: rgba(245,158,11,0.15); color:#f59e0b; border:1px solid rgba(245,158,11,0.4); }
.badge-high { background: rgba(239,68,68,0.15); color:#ef4444; border:1px solid rgba(239,68,68,0.4); }

.loc-card {
    background: #141d33; border: 1px solid #263252; border-radius: 14px;
    padding: 14px 16px; margin-bottom: 10px;
}
.loc-card .name { font-weight: 700; font-size: 15px; color: #f1f5f9; }
hr { border-color: #1e293b; }
[data-testid="stMetricValue"] { color: #f1f5f9; }
</style>
""",
    unsafe_allow_html=True,
)


def hero(title: str, subtitle: str):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(col, label, value, sub=""):
    col.markdown(
        f"""<div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _lerp_color(c1: str, c2: str, t: float) -> str:
    c1, c2 = c1.lstrip("#"), c2.lstrip("#")
    r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
    r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)
    r, g, b = round(r1 + (r2 - r1) * t), round(g1 + (g2 - g1) * t), round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def ratio_to_color(ratio) -> str:
    """
    speed_ratio -> hex color, matching label_generation.py's thresholds
    (free_flow >= 0.7, moderate >= 0.4, else congested) with a smooth
    gradient between bands instead of a hard cutoff.
    """
    if ratio is None or pd.isna(ratio):
        return "#64748b"  # unknown - neutral gray
    ratio = max(0.0, min(1.0, float(ratio)))
    if ratio >= 0.7:
        return _lerp_color(COLORS["moderate"], COLORS["low"], (ratio - 0.7) / 0.3)
    if ratio >= 0.4:
        return _lerp_color(COLORS["high"], COLORS["moderate"], (ratio - 0.4) / 0.3)
    return _lerp_color("#7f1d1d", COLORS["high"], ratio / 0.4)


def badge(label: str) -> str:
    label = (label or "unknown").lower()
    cls = {"low": "badge-low", "moderate": "badge-moderate", "high": "badge-high"}.get(
        label, "badge-moderate"
    )
    return f'<span class="badge {cls}">{label.upper()}</span>'


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------
@st.cache_resource
def get_engine():
    return create_engine(POSTGRES_DSN, pool_pre_ping=True)


def run_query(sql: str, **params) -> pd.DataFrame:
    try:
        with get_engine().connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        st.session_state["_db_error"] = str(exc)
        return pd.DataFrame()


@st.cache_data(ttl=REFRESH_SECONDS)
def load_latest_snapshots() -> pd.DataFrame:
    return run_query(
        """
        SELECT DISTINCT ON (s.location_id)
            s.location_id, l.name AS location_name, l.lat, l.lon,
            s.collected_at, s.current_speed, s.free_flow_speed,
            s.current_travel_time, s.free_flow_travel_time, s.confidence,
            s.road_closure, s.segment_geometry, s.incident_count, s.incident_severity_max,
            s.temperature_c, s.humidity_pct, s.wind_speed_ms, s.rain_1h_mm,
            s.weather_main, s.weather_description, s.visibility_m
        FROM traffic_snapshots s
        JOIN locations l ON l.location_id = s.location_id
        ORDER BY s.location_id, s.collected_at DESC
        """
    )


@st.cache_data(ttl=REFRESH_SECONDS)
def load_recent_snapshots(hours: int) -> pd.DataFrame:
    return run_query(
        """
        SELECT s.location_id, l.name AS location_name, s.collected_at,
               s.current_speed, s.free_flow_speed, s.current_travel_time,
               s.free_flow_travel_time, s.incident_count, s.incident_severity_max,
               s.incident_severity_avg,
               s.road_closure, s.temperature_c, s.rain_1h_mm, s.visibility_m,
               s.weather_main
        FROM traffic_snapshots s
        JOIN locations l ON l.location_id = s.location_id
        WHERE s.collected_at >= now() - (:hrs || ' hours')::interval
        ORDER BY s.collected_at
        """,
        hrs=hours,
    )


@st.cache_data(ttl=REFRESH_SECONDS)
def load_latest_predictions() -> pd.DataFrame:
    return run_query(
        """
        SELECT DISTINCT ON (p.location_id)
            p.location_id, l.name AS location_name, l.lat, l.lon,
            p.collected_at, p.predicted_congestion_label,
            p.prob_low, p.prob_moderate, p.prob_high, p.speed_ratio, p.delay
        FROM live_predictions p
        JOIN locations l ON l.location_id = p.location_id
        ORDER BY p.location_id, p.collected_at DESC
        """
    )


@st.cache_data(ttl=REFRESH_SECONDS)
def load_recent_predictions(hours: int) -> pd.DataFrame:
    return run_query(
        """
        SELECT p.location_id, l.name AS location_name, p.collected_at,
               p.predicted_congestion_label, p.prob_low, p.prob_moderate,
               p.prob_high, p.speed_ratio, p.delay
        FROM live_predictions p
        JOIN locations l ON l.location_id = p.location_id
        WHERE p.collected_at >= now() - (:hrs || ' hours')::interval
        ORDER BY p.collected_at
        """,
        hrs=hours,
    )


@st.cache_data(ttl=60)
def load_metrics():
    candidates = [
        METRICS_PATH,
        os.path.join("..", METRICS_PATH),
        "/app/" + METRICS_PATH,
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def build_road_map(df: pd.DataFrame, height: int = 430) -> go.Figure:
    """
    One Scattermapbox line trace per monitored road, colored by live
    speed_ratio - this is the actual TomTom segment polyline, not a
    straight line between two points. Falls back to a marker-only dot for
    any row where segment_geometry hasn't been captured yet (e.g. rows
    collected before this feature existed, or a transient API miss).
    """
    fig = go.Figure()
    has_geometry = False

    for _, row in df.iterrows():
        ratio = row.get("speed_ratio")
        color = ratio_to_color(ratio)
        geom = row.get("segment_geometry")
        hover = (
            f"<b>{row['location_name']}</b><br>"
            f"speed {row['current_speed']:.0f} km/h (free-flow {row['free_flow_speed']:.0f})<br>"
            f"ratio {ratio:.2f}<br>"
            f"incidents {int(row['incident_count'] or 0)}"
            + ("<br><b>ROAD CLOSED</b>" if row.get("road_closure") else "")
        )

        if isinstance(geom, list) and len(geom) >= 2:
            has_geometry = True
            lats = [p[0] for p in geom]
            lons = [p[1] for p in geom]
            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons, mode="lines",
                line=dict(width=6 if not row.get("road_closure") else 4, color=color),
                opacity=0.5 if row.get("road_closure") else 0.95,
                hoverinfo="text", text=[hover] * len(lats),
                showlegend=False,
            ))
            if row.get("road_closure"):
                mid = geom[len(geom) // 2]
                fig.add_trace(go.Scattermapbox(
                    lat=[mid[0]], lon=[mid[1]], mode="markers",
                    marker=dict(size=16, color="#ef4444", symbol="circle"),
                    hoverinfo="text", text=[hover], showlegend=False,
                ))
        else:
            # No captured geometry for this row yet - show a marker so the
            # location isn't silently missing from the map.
            fig.add_trace(go.Scattermapbox(
                lat=[row["lat"]], lon=[row["lon"]], mode="markers",
                marker=dict(size=14, color=color),
                hoverinfo="text", text=[hover], showlegend=False,
            ))

    center_lat = df["lat"].mean() if not df.empty else 18.52
    center_lon = df["lon"].mean() if not df.empty else 73.85
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=10.6),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        height=height,
    )
    if not has_geometry:
        st.caption(
            "⚠️ No road geometry captured yet for these rows — showing point markers. "
            "New snapshots collected after this update will draw full road segments."
        )
    return fig


def db_error_banner():
    err = st.session_state.get("_db_error")
    if err:
        st.error(
            "Couldn't reach Postgres. Make sure the stack is running: "
            "`docker compose up -d postgres collector`.\n\n"
            f"Details: {err}"
        )
        st.session_state["_db_error"] = None
        return True
    return False


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚦 Pune Traffic")
    st.caption("Command Center")
    page = st.radio(
        "Dashboard",
        [
            "1 · Real-Time Monitoring",
            "2 · Traffic Prediction",
            "3 · Risk & Incident Forecast",
            "4 · Traffic Analytics",
            "5 · Model Performance",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    auto = st.toggle("Auto-refresh", value=True)
    if auto:
        st_autorefresh(interval=REFRESH_SECONDS * 1000, key="refresh")
    st.caption(f"Refreshing every {REFRESH_SECONDS}s" if auto else "Paused")
    st.divider()
    st.caption(f"Now: {datetime.now():%H:%M:%S}")

# ==========================================================================
# PAGE 1 — Real-Time Monitoring
# ==========================================================================
if page.startswith("1"):
    hero(
        "Real-Time Monitoring",
        "Current speed, travel time, live incidents and weather across Pune",
    )
    df = load_latest_snapshots()
    if db_error_banner():
        pass
    elif df.empty:
        st.warning("No snapshots yet — start the collector: `docker compose up -d collector`.")
    else:
        df["speed_ratio"] = df["current_speed"] / df["free_flow_speed"].replace(0, pd.NA)
        df["delay_s"] = df["current_travel_time"] - df["free_flow_travel_time"]

        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, "Avg Speed", f"{df['current_speed'].mean():.1f} km/h",
                    f"free-flow avg {df['free_flow_speed'].mean():.1f} km/h")
        metric_card(c2, "Avg Delay", f"{df['delay_s'].mean():.0f} s",
                    "vs. free-flow travel time")
        metric_card(c3, "Active Incidents", f"{int(df['incident_count'].sum())}",
                    f"max severity {int(df['incident_severity_max'].max())}/4")
        metric_card(c4, "Road Closures", f"{int(df['road_closure'].sum())}",
                    f"of {len(df)} monitored roads")

        st.markdown("####")
        map_col, list_col = st.columns([1.3, 1])
        with map_col:
            st.subheader("Live Road Congestion Map")
            st.plotly_chart(build_road_map(df), use_container_width=True)
            st.markdown(
                """<div style="display:flex;gap:16px;font-size:12px;color:#94a3b8;">
                    <span>🟢 free flow</span>
                    <span>🟠 moderate</span>
                    <span>🔴 congested</span>
                    <span>⭕ road closed</span>
                </div>""",
                unsafe_allow_html=True,
            )

        with list_col:
            st.subheader("Per-Location Snapshot")
            for _, row in df.sort_values("location_name").iterrows():
                ratio = row["speed_ratio"] or 0
                lvl = "low" if ratio >= 0.7 else "moderate" if ratio >= 0.4 else "high"
                st.markdown(
                    f"""<div class="loc-card">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span class="name">{row['location_name']}</span>{badge(lvl)}
                        </div>
                        <div style="margin-top:6px;font-size:13px;color:#94a3b8;">
                            {row['current_speed']:.0f} km/h · delay {row['delay_s']:.0f}s ·
                            {int(row['incident_count'])} incident(s) ·
                            {row['weather_description'] or 'n/a'}, {row['temperature_c'] if pd.notna(row['temperature_c']) else '—'}°C
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.subheader("Full Snapshot Table")
        show_cols = [
            "location_name", "current_speed", "free_flow_speed", "delay_s",
            "incident_count", "incident_severity_max", "road_closure",
            "temperature_c", "weather_description", "collected_at",
        ]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

# ==========================================================================
# PAGE 2 — Traffic Prediction
# ==========================================================================
elif page.startswith("2"):
    hero(
        "Traffic Prediction",
        "Model-forecast congestion, with estimated future speed & travel time",
    )
    pred = load_latest_predictions()
    snap = load_latest_snapshots()

    if db_error_banner():
        pass
    elif pred.empty:
        st.warning(
            "No live predictions yet. Train + start the streaming scorer:\n\n"
            "`docker compose --profile spark run --rm spark spark-submit "
            "/app/spark_jobs/train_model.py`\n\n"
            "`docker compose --profile kafka --profile streaming up -d spark-streaming`"
        )
    else:
        merged = pred.merge(
            snap[["location_id", "free_flow_speed", "free_flow_travel_time", "current_speed"]],
            on="location_id", how="left",
        )
        merged["expected_ratio"] = merged["predicted_congestion_label"].map(EXPECTED_RATIO)
        merged["predicted_speed"] = merged["expected_ratio"] * merged["free_flow_speed"]
        merged["predicted_travel_time"] = merged["free_flow_travel_time"] / merged["expected_ratio"]

        c1, c2, c3 = st.columns(3)
        n_high = (merged["predicted_congestion_label"] == "high").sum()
        metric_card(c1, "Locations Forecast High", f"{n_high}/{len(merged)}",
                    "predicted congestion = high")
        metric_card(c2, "Avg Predicted Speed", f"{merged['predicted_speed'].mean():.1f} km/h",
                    "estimated from predicted class")
        metric_card(c3, "Avg Prob(High)", f"{merged['prob_high'].mean()*100:.0f}%",
                    "mean across locations")

        st.caption(
            "⚠️ Predicted speed / travel time are display-layer estimates derived from the "
            "predicted congestion class (a classifier), not a direct regression output — "
            "treat as directional, not exact."
        )

        st.markdown("####")
        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Forecast by Location")
            prob_df = merged.melt(
                id_vars=["location_name"],
                value_vars=["prob_low", "prob_moderate", "prob_high"],
                var_name="class", value_name="probability",
            )
            prob_df["class"] = prob_df["class"].str.replace("prob_", "")
            fig = px.bar(
                prob_df, x="location_name", y="probability", color="class",
                color_discrete_map={"low": COLORS["low"], "moderate": COLORS["moderate"], "high": COLORS["high"]},
                barmode="stack", height=380,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1", legend_title="",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Predicted vs Current Speed")
            comp = merged[["location_name", "current_speed", "predicted_speed"]].melt(
                id_vars="location_name", var_name="type", value_name="speed"
            )
            comp["type"] = comp["type"].map({"current_speed": "current", "predicted_speed": "predicted (next)"})
            fig2 = px.bar(
                comp, x="location_name", y="speed", color="type", barmode="group",
                color_discrete_sequence=[COLORS["accent"], "#a855f7"], height=380,
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Prob(High Congestion) — Trend")
        hist = load_recent_predictions(2)
        if hist.empty:
            st.info("Not enough streaming history yet for a trend line.")
        else:
            fig3 = px.line(
                hist, x="collected_at", y="prob_high", color="location_name", height=340,
            )
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Forecast Table")
        table = merged[[
            "location_name", "predicted_congestion_label", "prob_low", "prob_moderate",
            "prob_high", "current_speed", "predicted_speed", "predicted_travel_time", "collected_at",
        ]].rename(columns={"predicted_congestion_label": "predicted_class"})
        st.dataframe(table, use_container_width=True, hide_index=True)

# ==========================================================================
# PAGE 3 — Risk & Incident Forecast
# ==========================================================================
elif page.startswith("3"):
    hero(
        "Risk & Incident Forecast",
        "Incident probability, road-closure risk and a composite risk score per location",
    )
    hist = load_recent_snapshots(6)
    pred = load_latest_predictions()

    if db_error_banner():
        pass
    elif hist.empty:
        st.warning("No history yet — let the collector run a while, then reload.")
    else:
        raw_agg = hist.groupby("location_name").agg(
            avg_incident_count=("incident_count", "mean"),
            closure_rate=("road_closure", "mean"),
            avg_severity=("incident_severity_avg", "mean"),
            samples=("incident_count", "size"),
        ).reset_index()

        # A raw incident *count* threshold doesn't generalize across locations -
        # a city-center bbox naturally sees more incidents than a quieter one on
        # an ordinary day, so any fixed cutoff either saturates everywhere or
        # never fires depending on the area's baseline density. Rank each
        # location relative to the others in this window instead: genuinely
        # busier-than-usual areas score higher, without one area's raw count
        # swamping everyone else's score to near-zero.
        lo, hi = raw_agg["avg_incident_count"].min(), raw_agg["avg_incident_count"].max()
        if hi > lo:
            raw_agg["incident_rate"] = (raw_agg["avg_incident_count"] - lo) / (hi - lo)
        else:
            raw_agg["incident_rate"] = 0.0
        agg = raw_agg

        if not pred.empty:
            agg = agg.merge(
                pred[["location_name", "prob_high", "predicted_congestion_label"]],
                on="location_name", how="left",
            )
        else:
            agg["prob_high"] = 0.0
            agg["predicted_congestion_label"] = "n/a"
        agg["prob_high"] = agg["prob_high"].fillna(0.0)

        agg["risk_score"] = (
            0.5 * agg["prob_high"]
            + 0.3 * agg["incident_rate"]
            + 0.2 * agg["closure_rate"]
        ) * 100
        agg = agg.sort_values("risk_score", ascending=False)

        c1, c2, c3 = st.columns(3)
        top = agg.iloc[0] if not agg.empty else None
        metric_card(c1, "Highest Risk Location", top["location_name"] if top is not None else "—",
                    f"score {top['risk_score']:.0f}/100" if top is not None else "")
        metric_card(c2, "Avg Incident Probability", f"{agg['incident_rate'].mean()*100:.0f}%",
                    "relative incident density vs other locations")
        metric_card(c3, "Avg Closure Risk", f"{agg['closure_rate'].mean()*100:.0f}%",
                    "share of last 6h flagged closed")

        st.markdown("####")
        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Composite Risk Score by Location")
            fig = px.bar(
                agg, x="risk_score", y="location_name", orientation="h",
                color="risk_score", color_continuous_scale="RdYlGn_r", height=380,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                yaxis_title="", xaxis_title="risk score (0-100)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Risk Radar")
            categories = ["incident_rate", "closure_rate", "prob_high", "avg_severity"]
            fig2 = go.Figure()
            for _, row in agg.iterrows():
                vals = [
                    row["incident_rate"], row["closure_rate"],
                    row["prob_high"], (row["avg_severity"] or 0) / 4,
                ]
                fig2.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=categories + [categories[0]],
                    fill="toself", name=row["location_name"], opacity=0.55,
                ))
            fig2.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1], color="#94a3b8"),
                           bgcolor="rgba(0,0,0,0)"),
                paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", height=380,
                showlegend=True, legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Risk Detail")
        display = agg.copy()
        display["avg_incident_count"] = display["avg_incident_count"].round(1)
        display["incident_rate"] = (display["incident_rate"] * 100).round(1)
        display["closure_rate"] = (display["closure_rate"] * 100).round(1)
        display["prob_high"] = (display["prob_high"] * 100).round(1)
        display["risk_score"] = display["risk_score"].round(1)
        st.dataframe(
            display.rename(columns={
                "incident_rate": "incident_prob_%", "closure_rate": "closure_risk_%",
                "prob_high": "model_prob_high_%",
            }),
            use_container_width=True, hide_index=True,
        )

# ==========================================================================
# PAGE 4 — Traffic Analytics
# ==========================================================================
elif page.startswith("4"):
    hero(
        "Traffic Analytics",
        "Historical trends, hotspot analysis and weather impact",
    )
    window = st.select_slider(
        "History window", options=[1, 3, 6, 12, 24, 48], value=6, format_func=lambda h: f"{h}h"
    )
    hist = load_recent_snapshots(window)

    if db_error_banner():
        pass
    elif hist.empty:
        st.warning("No history in this window yet — widen the window or let the collector run longer.")
    else:
        hist["speed_ratio"] = hist["current_speed"] / hist["free_flow_speed"].replace(0, pd.NA)

        st.subheader("Historical Speed Trend")
        fig = px.line(hist, x="collected_at", y="current_speed", color="location_name", height=360)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1")
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("Hotspot Analysis")
            st.caption("Lowest avg speed ratio = most congestion-prone")
            hot = hist.groupby("location_name")["speed_ratio"].mean().sort_values().reset_index()
            fig2 = px.bar(
                hot, x="speed_ratio", y="location_name", orientation="h",
                color="speed_ratio", color_continuous_scale="RdYlGn", height=340,
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                yaxis_title="", xaxis_title="avg speed ratio",
            )
            st.plotly_chart(fig2, use_container_width=True)

        with right:
            st.subheader("Weather Impact")
            wdf = hist.dropna(subset=["rain_1h_mm", "speed_ratio"])
            if len(wdf) > 5:
                corr = wdf["rain_1h_mm"].corr(wdf["speed_ratio"])
                st.caption(f"Correlation (rainfall vs speed ratio): **{corr:.2f}**")
                fig3 = px.scatter(
                    wdf, x="rain_1h_mm", y="speed_ratio", color="location_name",
                    trendline="ols" if len(wdf) > 20 else None, height=300,
                )
                fig3.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Not enough rain data yet in this window to chart weather impact.")

        st.subheader("Congestion by Hour of Day")
        hist["hour"] = pd.to_datetime(hist["collected_at"]).dt.hour
        heat = hist.groupby(["location_name", "hour"])["speed_ratio"].mean().reset_index()
        pivot = heat.pivot(index="location_name", columns="hour", values="speed_ratio")
        fig4 = px.imshow(
            pivot, color_continuous_scale="RdYlGn", aspect="auto",
            labels=dict(color="avg speed ratio"), height=280,
        )
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1")
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Raw History")
        st.dataframe(
            hist[[
                "location_name", "collected_at", "current_speed", "free_flow_speed",
                "incident_count", "temperature_c", "rain_1h_mm", "weather_main",
            ]],
            use_container_width=True, hide_index=True,
        )

# ==========================================================================
# PAGE 5 — Model Performance
# ==========================================================================
else:
    hero(
        "Model Performance",
        "RandomForest congestion classifier — evaluation on the held-out (time-based) test split",
    )
    metrics = load_metrics()

    if metrics is None:
        st.warning(
            "No metrics.json found yet. Train the model to generate it:\n\n"
            "`docker compose --profile spark run --rm spark spark-submit "
            "/app/spark_jobs/train_model.py`\n\n"
            "This writes `data/models/congestion_rf/metrics.json`, which this page reads directly."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, "Accuracy", f"{metrics['accuracy']*100:.1f}%")
        metric_card(c2, "Precision (wt)", f"{metrics['precision_weighted']*100:.1f}%")
        metric_card(c3, "Recall (wt)", f"{metrics['recall_weighted']*100:.1f}%")
        metric_card(c4, "F1 Score (wt)", f"{metrics['f1_weighted']*100:.1f}%")

        c5, c6, c7, c8 = st.columns(4)
        metric_card(c5, "MAE", f"{metrics['mae']:.3f}", "ordinal class error")
        metric_card(c6, "RMSE", f"{metrics['rmse']:.3f}", "ordinal class error")
        metric_card(c7, "Train / Test Rows", f"{metrics['train_rows']} / {metrics['test_rows']}")
        metric_card(c8, "Trees / Depth", f"{metrics['num_trees']} / {metrics['max_depth']}")

        st.caption(
            f"Generated {metrics['generated_at']} · model dir `{metrics['model_dir']}` · "
            "MAE/RMSE treat congestion_level (0/1/2) as ordinal, so a 'high' misclassified as "
            "'moderate' counts as a smaller error than 'high' misclassified as 'low'."
        )

        st.markdown("####")
        left, right = st.columns([1, 1.2])
        with left:
            st.subheader("Confusion Matrix")
            labels = metrics["labels"]
            cm = metrics["confusion_matrix"]
            fig = px.imshow(
                cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="count"), height=380,
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Feature Importance")
            fi = pd.DataFrame(metrics["feature_importance"]).head(15).iloc[::-1]
            fig2 = px.bar(
                fi, x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale="Viridis", height=380,
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                yaxis_title="",
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("All Feature Importances")
        st.dataframe(pd.DataFrame(metrics["feature_importance"]), use_container_width=True, hide_index=True)

        if metrics.get("train_rows", 0) < 2000:
            st.info(
                "Heads-up from train_model.py: with only a few hours of history, location_id can "
                "dominate as a predictor and these numbers are a 'pipeline works' checkpoint, not "
                "a final quality read. Re-train as more varied (peak + off-peak) data accumulates."
            )
