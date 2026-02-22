import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import geopandas as gpd

# ---------- Style ----------
pio.templates.default = "plotly"  # brighter + easier to see than plotly_dark

st.set_page_config(page_title="Cuse RiskLens", layout="wide")

DATA_DIR = "data"
TABLES_DIR = f"{DATA_DIR}/tables"
GPKG_NHOODS = f"{DATA_DIR}/nhood_polygons.gpkg"
GPKG_CENTROIDS = f"{DATA_DIR}/parcels_centroids.gpkg"


# ---------- Helpers ----------
@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(f"{TABLES_DIR}/{name}")


def pick(row: dict, keys, default=np.nan):
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default


def fmt_int(x):
    try:
        if pd.isna(x):
            return "—"
        return f"{int(float(x)):,}"
    except Exception:
        return "—"


def fmt_pct(x, digits=1):
    try:
        if pd.isna(x):
            return "—"
        x = float(x)
        # allow both 0-1 and 0-100
        if x > 1.5:
            return f"{x:.{digits}f}%"
        return f"{100 * x:.{digits}f}%"
    except Exception:
        return "—"


@st.cache_data
def load_nhood_geo():
    # Try common layer name, else default
    try:
        gdf = gpd.read_file(GPKG_NHOODS, layer="nhood_polygons")
    except Exception:
        gdf = gpd.read_file(GPKG_NHOODS)

    if gdf is None or len(gdf) == 0:
        raise RuntimeError("Could not load neighborhood polygons from nhood_polygons.gpkg")

    # Standardize name column to `nhood`
    if "nhood" not in gdf.columns:
        for cand in ["name", "Neighborhood", "neighborhood", "NHOOD", "NBHD"]:
            if cand in gdf.columns:
                gdf = gdf.rename(columns={cand: "nhood"})
                break
    if "nhood" not in gdf.columns:
        gdf["nhood"] = gdf.index.astype(str)

    # Ensure WGS84 for centroid labels
    if gdf.crs is not None:
        try:
            gdf_wgs = gdf.to_crs(4326)
        except Exception:
            gdf_wgs = gdf.copy()
    else:
        gdf_wgs = gdf.copy()

    # centroid points for labels
    cent = gdf_wgs.geometry.centroid
    gdf_wgs["centroid_lon"] = cent.x
    gdf_wgs["centroid_lat"] = cent.y

    geojson = json.loads(gdf_wgs.to_json())
    centroids = gdf_wgs[["nhood", "centroid_lat", "centroid_lon"]].copy()

    return geojson, centroids


@st.cache_data
def load_parcel_centroids():
    # Try common layer name, else default
    try:
        gdf = gpd.read_file(GPKG_CENTROIDS, layer="parcels_centroids")
    except Exception:
        gdf = gpd.read_file(GPKG_CENTROIDS)

    if gdf is None or len(gdf) == 0:
        raise RuntimeError("Could not load parcel centroids from parcels_centroids.gpkg")

    keep = [c for c in [
        "sbl", "nhood", "centroid_lat", "centroid_lon",
        "total_av", "land_av", "yr_built", "n_resunits"
    ] if c in gdf.columns]

    df = pd.DataFrame(gdf[keep]).copy()

    # Derive centroid_lat/lon if missing
    if ("centroid_lat" not in df.columns or "centroid_lon" not in df.columns) and "geometry" in gdf.columns:
        gg = gdf
        if gg.crs is not None:
            try:
                gg = gg.to_crs(4326)
            except Exception:
                pass
        cent = gg.geometry.centroid
        df["centroid_lon"] = cent.x
        df["centroid_lat"] = cent.y

    # Standardize parcel id
    if "sbl" not in df.columns:
        for cand in ["SBL", "parcel_id", "PARCELID", "pid"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "sbl"})
                break

    return df.dropna(subset=["sbl", "centroid_lat", "centroid_lon"])


# ---------- Load tables ----------
city_stats = load_csv("city_overdue_stats.csv")
nhood_debt = load_csv("nhood_overdue_burden.csv")
parcel_forecast = load_csv("parcel_risk_forecast_2025_01_from_2024_12_.csv")
action = load_csv("model_actionability_summary.csv")
feat_imp = load_csv("model_feature_importance.csv")
validated = load_csv("nhood_risk_validated_2024_12.csv")

# Geo artifacts
nhood_geojson, nhood_centroids = load_nhood_geo()
parcel_centroids = load_parcel_centroids()


# ---------- Header ----------
st.title("Cuse RiskLens")
st.caption("Interactive Compliance Debt + Next-Month Risk Forecast Dashboard")

row0 = city_stats.iloc[0].to_dict() if len(city_stats) else {}
open_total = pick(row0, ["open_total", "open_rows", "open_records", "open_total_rows", "open_cases"])
overdue_share = pick(row0, ["overdue_share", "overdue_pct", "overdue_share_open", "share_overdue"])
median_overdue = pick(row0, ["median_overdue_days", "overdue_days_median", "median_days_overdue", "median_overdue"])
p95_overdue = pick(row0, ["p95_overdue_days", "overdue_days_p95", "p95_days_overdue", "p95_overdue"])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Open records", fmt_int(open_total))
k2.metric("Overdue share", fmt_pct(overdue_share, 1))
k3.metric("Median overdue (days)", fmt_int(median_overdue))
k4.metric("95th pct overdue (days)", fmt_int(p95_overdue))

st.divider()

tabs = st.tabs(["Neighborhood Debt", "Parcel Risk", "Model Insights"])


# =====================================================
# TAB 1 — Neighborhood Debt
# =====================================================
with tabs[0]:
    st.subheader("Neighborhood Compliance Debt")

    metric_map = {
        "Compliance debt burden (Overdue-days sum)": "overdue_days_sum",
        "Share overdue among open cases": "overdue_share",
        "Median overdue days": "overdue_days_median",
        "P95 overdue days": "overdue_days_p95",
        "Open overdue rows": "open_overdue_rows",
    }
    available = [k for k, v in metric_map.items() if v in nhood_debt.columns]
    if not available:
        st.error("No known metric columns found in nhood_overdue_burden.csv")
        st.stop()

    metric_label = st.selectbox("Map metric", available, index=0)
    metric_col = metric_map[metric_label]
    top_k = st.slider("Highlight top neighborhoods", 3, 20, 10)

    df = nhood_debt.copy()
    if "nhood" not in df.columns and "name" in df.columns:
        df = df.rename(columns={"name": "nhood"})
    if "nhood" not in df.columns:
        st.error("nhood_overdue_burden.csv must contain a neighborhood name column ('nhood' or 'name').")
        st.stop()

    df = df.dropna(subset=[metric_col]).copy()
    df["rank"] = df[metric_col].rank(method="first", ascending=False).astype(int)
    df["is_topk"] = df["rank"] <= top_k

    # Base choropleth
    fig = px.choropleth(
        df,
        geojson=nhood_geojson,
        featureidkey="properties.nhood",
        locations="nhood",
        color=metric_col,
        hover_name="nhood",
        hover_data={"rank": True, metric_col: True},
        color_continuous_scale="Turbo",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0))

    # Make borders visible + slightly fade fill so outlines/labels pop
    fig.update_traces(marker_line_width=0.6, marker_line_color="rgba(0,0,0,0.35)", marker_opacity=0.92)

    # Overlay: thick red outlines for top-K
    df_top = df[df["is_topk"]].copy()
    if len(df_top) > 0:
        fig_top = px.choropleth(
            df_top,
            geojson=nhood_geojson,
            featureidkey="properties.nhood",
            locations="nhood",
            color=metric_col,  # required, we’ll hide fill
        )
        for tr in fig_top.data:
            tr.marker.opacity = 0.0
            tr.marker.line.width = 5
            tr.marker.line.color = "#FF2D2D"
        fig.add_traces(fig_top.data)

        # Add numbered centroid labels for top-K so the slider change is SUPER obvious
        label_pts = df_top.merge(nhood_centroids, on="nhood", how="left").dropna(subset=["centroid_lat", "centroid_lon"])
        fig.add_trace(
            go.Scattergeo(
                lon=label_pts["centroid_lon"],
                lat=label_pts["centroid_lat"],
                mode="text",
                text=label_pts["rank"].astype(str),
                textfont=dict(size=14, color="white"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown("### Top neighborhoods")
        show_cols = ["nhood", metric_col, "overdue_share"] if "overdue_share" in df.columns else ["nhood", metric_col]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(
            df.sort_values(metric_col, ascending=False)[show_cols].head(top_k),
            width="stretch",
            hide_index=True,
        )
        st.markdown("### One-liner takeaway")
        st.write("Compliance debt is **highly concentrated** — the outlined neighborhoods are your highest-priority areas.")

    st.divider()
    st.markdown("### Validated neighborhood risk (2024-12)")
    st.dataframe(validated, width="stretch", hide_index=True)


# =====================================================
# TAB 2 — Parcel Risk
# =====================================================
with tabs[1]:
    st.subheader("Parcel-level Next-Month Risk")

    pct = st.slider("Show top X% highest-risk parcels", 0.5, 5.0, 1.0, 0.5)
    max_points = st.slider("Max points (performance)", 1000, 15000, 6000, 500)

    df_risk = parcel_forecast.copy()
    if "sbl" not in df_risk.columns:
        st.error("parcel_risk_forecast_*.csv must include column 'sbl'.")
        st.stop()
    if "risk_score" not in df_risk.columns:
        st.error("parcel_risk_forecast_*.csv must include column 'risk_score'.")
        st.stop()

    joined = df_risk.merge(parcel_centroids, on="sbl", how="left")

    # fix nhood_x/nhood_y
    if "nhood" not in joined.columns:
        if "nhood_x" in joined.columns and "nhood_y" in joined.columns:
            joined["nhood"] = joined["nhood_x"].fillna(joined["nhood_y"])
            joined = joined.drop(columns=["nhood_x", "nhood_y"])
        elif "nhood_x" in joined.columns:
            joined = joined.rename(columns={"nhood_x": "nhood"})
        elif "nhood_y" in joined.columns:
            joined = joined.rename(columns={"nhood_y": "nhood"})

    joined = joined.dropna(subset=["centroid_lat", "centroid_lon", "risk_score"]).copy()

    joined["risk_rank"] = joined["risk_score"].rank(method="first", ascending=False)
    joined["risk_pct_rank"] = 100.0 * (joined["risk_rank"] - 1) / max(len(joined) - 1, 1)

    filt = joined[joined["risk_pct_rank"] <= pct].sort_values("risk_score", ascending=False).copy()
    if len(filt) > max_points:
        filt = filt.head(max_points)

    fig2 = px.scatter_map(
        filt,
        lat="centroid_lat",
        lon="centroid_lon",
        color="risk_score",
        hover_name="sbl",
        hover_data=[c for c in ["nhood", "risk_score", "total_av", "yr_built"] if c in filt.columns],
        zoom=11,
        height=650,
        color_continuous_scale="Plasma",
    )
    fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    c1, c2 = st.columns([1.7, 1])
    with c1:
        st.plotly_chart(fig2, width="stretch")
    with c2:
        st.markdown("### What this means")
        st.write("This map surfaces the **highest-risk parcels** to prioritize inspections when capacity is limited.")
        st.markdown("### Top parcels (sample)")
        cols = [c for c in ["sbl", "nhood", "risk_score", "total_av", "yr_built"] if c in filt.columns]
        st.dataframe(filt[cols].head(25), width="stretch", hide_index=True)


# =====================================================
# TAB 3 — Model Insights
# =====================================================
with tabs[2]:
    st.subheader("Model Actionability + Drivers")

    action_df = action.copy()

    xcol = next((c for c in ["k_pct", "k", "top_k_pct", "top_pct"] if c in action_df.columns), None)
    ycol = next((c for c in ["precision", "precision_at_k", "precision_at_k_pct"] if c in action_df.columns), None)

    if xcol and ycol:
        fig_a = px.line(
            action_df.sort_values(xcol),
            x=xcol,
            y=ycol,
            markers=True,
            title="Precision among top-ranked parcels",
        )
        fig_a.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_a, width="stretch")
    else:
        st.dataframe(action_df, width="stretch", hide_index=True)

    st.divider()

    fi = feat_imp.copy()
    if "feature" not in fi.columns:
        for cand in ["feat", "name", "feature_name"]:
            if cand in fi.columns:
                fi = fi.rename(columns={cand: "feature"})
                break
    if "importance" not in fi.columns:
        for cand in ["gain", "split", "value", "imp"]:
            if cand in fi.columns:
                fi = fi.rename(columns={cand: "importance"})
                break

    if "feature" in fi.columns and "importance" in fi.columns:
        topN = st.slider("Top features to show", 10, 40, 20, 5)
        fi2 = fi.sort_values("importance", ascending=False).head(topN).sort_values("importance", ascending=True)

        fig_f = px.bar(
            fi2,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Viridis",
            title="Top features driving predicted risk",
        )
        fig_f.update_layout(height=650, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_f, width="stretch")
    else:
        st.dataframe(fi, width="stretch", hide_index=True)