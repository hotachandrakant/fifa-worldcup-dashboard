    # ============================================================
#  ⚽  FIFA World Cup — Premium Data Visualization Dashboard
#  app.py  |  Run:  python app.py
#  Then open:       http://127.0.0.1:8050
# ============================================================
#  pip install numpy pandas plotly dash dash-bootstrap-components
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

CSV_PATH = os.path.join(os.path.dirname(__file__), "WorldCupMatches.csv")

df = pd.read_csv(CSV_PATH)
df.columns       = df.columns.str.strip()
df.dropna(how="all", inplace=True)
df.reset_index(drop=True, inplace=True)
df["TotalGoals"] = df["Home Team Goals"] + df["Away Team Goals"]
df["GoalDiff"]   = (df["Home Team Goals"] - df["Away Team Goals"]).abs()
df["Year"]       = df["Year"].astype(int)

print(f"✅  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── Palette ──────────────────────────────────────────────────
GOLD    = "#FFD700"
DARK    = "#04080F"
CARD    = "#0D1B2A"
CARD2   = "#112240"
ACCENT  = "#00F5FF"
PURPLE  = "#BF5AF2"
TEXT    = "#E2E8F0"
SUB     = "#8892A4"
GREEN   = "#00FF9C"
RED     = "#FF4D6D"

# ── Dropdown options ─────────────────────────────────────────
GRAPH_OPTIONS = [
    {"label": "📊  Bar Chart – Goals per Year",         "value": "bar_goals_year"},
    {"label": "📈  Line Chart – Matches per Year",      "value": "line_matches_year"},
    {"label": "🥧  Pie Chart – Home vs Away Wins",      "value": "pie_win_type"},
    {"label": "🔵  Scatter – Goals vs Goal Difference", "value": "scatter_goals_diff"},
    {"label": "🌡️  Heatmap – Goals by Stage & Year",   "value": "heatmap_goals"},
    {"label": "🎻  Violin – Total Goals Distribution",  "value": "violin_goals"},
    {"label": "📦  Box Plot – Goals per Stage",         "value": "box_stage_goals"},
    {"label": "🏔️  Area Chart – Cumulative Goals",      "value": "area_cumulative"},
    {"label": "🌍  3D Scatter – Year/Home/Away Goals",  "value": "3d_scatter"},
    {"label": "🏔️  3D Surface – Goals Grid",           "value": "3d_surface"},
    {"label": "🌐  Geo Map – Matches per Country",      "value": "geo_map"},
    {"label": "🌲  Treemap – Wins by Country",          "value": "treemap_wins"},
    {"label": "☀️  Sunburst – Stage → Country",         "value": "sunburst"},
    {"label": "🕸️  Radar – Top 5 Teams Stats",          "value": "radar_teams"},
    {"label": "📉  Histogram – Goal Distribution",      "value": "histogram_goals"},
]

ATTR_OPTIONS = [
    {"label": "🗓️  Year",             "value": "Year"},
    {"label": "⚽  Total Goals",      "value": "TotalGoals"},
    {"label": "🏠  Home Team Goals",  "value": "Home Team Goals"},
    {"label": "🚌  Away Team Goals",  "value": "Away Team Goals"},
    {"label": "📐  Goal Difference",  "value": "GoalDiff"},
    {"label": "🏟️  Stage",           "value": "Stage"},
    {"label": "🏠  Home Team Name",   "value": "Home Team Name"},
    {"label": "🚌  Away Team Name",   "value": "Away Team Name"},
    {"label": "🏙️  City",            "value": "City"},
    {"label": "🌍  RoundID",         "value": "RoundID"},
]

# ── Figure builder ────────────────────────────────────────────
def build_figure(graph_val, attr_val):
    template     = "plotly_dark"
    paper        = CARD2
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols     = [c for c in df.columns if c not in numeric_cols]

    num_attr = attr_val if attr_val in numeric_cols else "TotalGoals"
    cat_attr = attr_val if attr_val in cat_cols     else "Stage"

    if graph_val == "bar_goals_year":
        if attr_val in numeric_cols and attr_val != "Year":
            agg = df.groupby("Year")[attr_val].sum().reset_index()
            fig = px.bar(agg, x="Year", y=attr_val,
                         color=attr_val, color_continuous_scale="Turbo",
                         title=f"📊 {attr_val} per Year", template=template)
        else:
            agg = df.groupby("Year")["TotalGoals"].sum().reset_index()
            fig = px.bar(agg, x="Year", y="TotalGoals",
                         color="TotalGoals", color_continuous_scale="Turbo",
                         title="📊 Total Goals per Year", template=template)

    elif graph_val == "line_matches_year":
        if attr_val in numeric_cols and attr_val != "Year":
            agg = df.groupby("Year")[attr_val].mean().reset_index()
            fig = px.line(agg, x="Year", y=attr_val, markers=True,
                          title=f"📈 Average {attr_val} per Year", template=template)
        else:
            agg = df.groupby("Year").size().reset_index(name="Matches")
            fig = px.line(agg, x="Year", y="Matches", markers=True,
                          title="📈 Matches per Year", template=template)
        fig.update_traces(line=dict(width=4, color=ACCENT), marker=dict(size=10, color=GOLD))

    elif graph_val == "pie_win_type":
        if attr_val in cat_cols:
            counts = df[attr_val].fillna("Unknown").value_counts().reset_index()
            counts.columns = [attr_val, "Count"]
            if len(counts) > 8:
                top   = counts.head(8).copy()
                other = pd.DataFrame({attr_val: ["Others"], "Count": [counts.iloc[8:]["Count"].sum()]})
                counts = pd.concat([top, other], ignore_index=True)
            fig = px.pie(counts, names=attr_val, values="Count",
                         title=f"🥧 Distribution of {attr_val}",
                         color_discrete_sequence=px.colors.sequential.Plasma_r,
                         template=template)
        elif attr_val in numeric_cols:
            agg = df.groupby("Stage")[attr_val].sum().reset_index()
            agg = agg.sort_values(by=attr_val, ascending=False)
            if len(agg) > 8:
                top   = agg.head(8).copy()
                other = pd.DataFrame({"Stage": ["Others"], attr_val: [agg.iloc[8:][attr_val].sum()]})
                agg = pd.concat([top, other], ignore_index=True)
            fig = px.pie(agg, names="Stage", values=attr_val,
                         title=f"🥧 Share of {attr_val} by Stage",
                         color_discrete_sequence=px.colors.sequential.Plasma_r,
                         template=template)
        else:
            hw = (df["Home Team Goals"] > df["Away Team Goals"]).sum()
            aw = (df["Away Team Goals"] > df["Home Team Goals"]).sum()
            dr = (df["Home Team Goals"] == df["Away Team Goals"]).sum()
            fig = px.pie(values=[hw, aw, dr], names=["Home Win", "Away Win", "Draw"],
                         title="🥧 Match Outcome Distribution",
                         color_discrete_sequence=[GOLD, ACCENT, PURPLE])
            fig.update_layout(template=template)

    elif graph_val == "scatter_goals_diff":
        color_col = attr_val if attr_val in df.columns else "Year"
        fig = px.scatter(df, x="TotalGoals", y="GoalDiff",
                         color=color_col, size="TotalGoals", opacity=0.8,
                         hover_data=["Home Team Name", "Away Team Name"],
                         title=f"🔵 Total Goals vs Goal Difference (Color: {color_col})",
                         color_continuous_scale="Plasma", template=template)

    elif graph_val == "heatmap_goals":
        if attr_val in numeric_cols:
            pivot = df.groupby(["Stage", "Year"])[attr_val].mean().unstack(fill_value=0)
            title = f"🌡️ Average {attr_val} by Stage & Year"
        else:
            pivot = df.groupby(["Stage", "Year"])["TotalGoals"].mean().unstack(fill_value=0)
            title = "🌡️ Average TotalGoals by Stage & Year"
        fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns.tolist(),
                                   y=pivot.index.tolist(), colorscale="Plasma", showscale=True))
        fig.update_layout(title=title, xaxis_title="Year", yaxis_title="Stage", template=template)

    elif graph_val == "violin_goals":
        y_col = attr_val if attr_val in numeric_cols else "TotalGoals"
        fig = px.violin(df, y=y_col, box=True, points="all",
                        title=f"🎻 Distribution of {y_col}",
                        color_discrete_sequence=[PURPLE], template=template)

    elif graph_val == "box_stage_goals":
        y_col = attr_val if attr_val in numeric_cols else "TotalGoals"
        fig = px.box(df, x="Stage", y=y_col, color="Stage",
                     title=f"📦 {y_col} by Stage",
                     color_discrete_sequence=px.colors.qualitative.Bold,
                     template=template)
        fig.update_xaxes(tickangle=-30)

    elif graph_val == "area_cumulative":
        y_col = attr_val if (attr_val in numeric_cols and attr_val != "Year") else "TotalGoals"
        agg   = df.groupby("Year")[y_col].sum().cumsum().reset_index(name=f"Cumulative_{y_col}")
        fig   = px.area(agg, x="Year", y=f"Cumulative_{y_col}",
                        title=f"🏔️ Cumulative {y_col} Over Years",
                        color_discrete_sequence=[ACCENT], template=template)

    elif graph_val == "3d_scatter":
        z_col = attr_val if (attr_val in numeric_cols and attr_val != "Year") else "Away Team Goals"
        fig   = px.scatter_3d(df, x="Year", y="Home Team Goals", z=z_col,
                              color="TotalGoals", size="TotalGoals",
                              hover_name="Home Team Name",
                              color_continuous_scale="Turbo",
                              title=f"🌍 3D Scatter: Year / Home Goals / {z_col}")
        fig.update_layout(template=template)

    elif graph_val == "3d_surface":
        val_col = attr_val if attr_val in numeric_cols else "TotalGoals"
        years   = sorted(df["Year"].unique())
        stages  = df["Stage"].unique().tolist()
        z_data  = []
        for s in stages:
            row = []
            for y in years:
                val = df[(df["Stage"] == s) & (df["Year"] == y)][val_col].mean()
                row.append(val if not np.isnan(val) else 0)
            z_data.append(row)
        fig = go.Figure(go.Surface(z=z_data, x=years,
                                   y=list(range(len(stages))), colorscale="Plasma"))
        fig.update_layout(
            title=f"🏔️ 3D Surface – Avg {val_col} (Stage × Year)",
            scene=dict(xaxis_title="Year", yaxis_title="Stage Index", zaxis_title=val_col),
            template=template)

    elif graph_val == "geo_map":
        if attr_val in numeric_cols:
            agg = df.groupby("Home Team Name")[attr_val].sum().reset_index()
            fig = px.choropleth(agg, locations="Home Team Name", locationmode="country names",
                                color=attr_val, color_continuous_scale="Plasma",
                                title=f"🌐 {attr_val} by Country")
        else:
            agg = df.groupby("Home Team Name").size().reset_index(name="Matches")
            fig = px.choropleth(agg, locations="Home Team Name", locationmode="country names",
                                color="Matches", color_continuous_scale="Plasma",
                                title="🌐 Matches by Country")
        fig.update_layout(template=template)

    elif graph_val == "treemap_wins":
        if attr_val in numeric_cols:
            agg = df.groupby("Home Team Name")[attr_val].sum().reset_index().nlargest(30, attr_val)
            fig = px.treemap(agg, path=["Home Team Name"], values=attr_val,
                             color=attr_val, color_continuous_scale="Turbo",
                             title=f"🌲 Top 30 Countries by {attr_val}")
        else:
            wins = (df[df["Home Team Goals"] > df["Away Team Goals"]]
                      .groupby("Home Team Name").size()
                      .reset_index(name="Wins").nlargest(30, "Wins"))
            fig  = px.treemap(wins, path=["Home Team Name"], values="Wins",
                              color="Wins", color_continuous_scale="Turbo",
                              title="🌲 Top 30 Countries by Wins")
        fig.update_layout(template=template)

    elif graph_val == "sunburst":
        if attr_val in numeric_cols:
            agg = df.groupby(["Stage", "Home Team Name"])[attr_val].sum().reset_index()
            agg = agg[agg[attr_val] > 0]
            fig = px.sunburst(agg, path=["Stage", "Home Team Name"], values=attr_val,
                              color=attr_val, color_continuous_scale="Plasma",
                              title=f"☀️ Stage → Country by {attr_val}")
        else:
            counts = df.groupby(["Stage", "Home Team Name"]).size().reset_index(name="Count")
            fig    = px.sunburst(counts, path=["Stage", "Home Team Name"], values="Count",
                                 color="Count", color_continuous_scale="Plasma",
                                 title="☀️ Stage → Country Count")
        fig.update_layout(template=template)

    elif graph_val == "radar_teams":
        top5   = df.groupby("Home Team Name")["Home Team Goals"].sum().nlargest(5).index.tolist()
        cats   = ["Goals Scored", "Matches Played", "Avg Goals", "Max Goals", "Win Rate"]
        colors = [GOLD, ACCENT, PURPLE, GREEN, RED]
        fig    = go.Figure()
        for i, team in enumerate(top5):
            sub  = df[df["Home Team Name"] == team]
            wins = (sub["Home Team Goals"] > sub["Away Team Goals"]).sum()
            vals = [sub["Home Team Goals"].sum(), len(sub),
                    sub["Home Team Goals"].mean() * 10,
                    sub["Home Team Goals"].max() * 5,
                    (wins / len(sub) * 100) if len(sub) > 0 else 0]
            vals += [vals[0]]
            fig.add_trace(go.Scatterpolar(r=vals, theta=cats + [cats[0]],
                                          fill="toself", name=team,
                                          line=dict(color=colors[i], width=2)))
        fig.update_layout(title="🕸️ Radar – Top 5 Teams Stats",
                          polar=dict(radialaxis=dict(visible=True)), template=template)

    elif graph_val == "histogram_goals":
        x_col = attr_val if attr_val in numeric_cols else "TotalGoals"
        fig   = px.histogram(df, x=x_col, nbins=30,
                             color_discrete_sequence=[ACCENT],
                             title=f"📉 Histogram of {x_col}", template=template)
    else:
        fig = go.Figure()
        fig.update_layout(title="Select a graph type", template=template)

    fig.update_layout(
        paper_bgcolor=paper,
        plot_bgcolor="#060E1C",
        font=dict(color=TEXT, family="Rajdhani, Segoe UI, sans-serif", size=13),
        title_font=dict(size=20, color=GOLD, family="Orbitron, sans-serif"),
        margin=dict(l=40, r=40, t=65, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.08)"),
    )
    return fig


# ── Dash app ─────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700"
        "&family=Orbitron:wght@700&display=swap",
    ],
    suppress_callback_exceptions=True,
)
app.title = "FIFA World Cup Dashboard"

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* scrollbar */
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: #04080F; }
            ::-webkit-scrollbar-thumb { background: #FFD700; border-radius: 4px; }

            /* animated gradient bg */
            body {
                background: linear-gradient(135deg,#04080F 0%,#0D1B2A 40%,#0A0F1E 70%,#04080F 100%) !important;
                background-size: 400% 400% !important;
                animation: bgShift 14s ease infinite !important;
            }
            @keyframes bgShift {
                0%   { background-position: 0% 50%; }
                50%  { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            /* stat card hover glow + lift */
            .stat-card {
                transition: transform 0.25s ease, box-shadow 0.25s ease !important;
                cursor: default;
            }
            .stat-card:hover { transform: translateY(-7px) scale(1.04) !important; }
            .stat-gold:hover  { box-shadow: 0 0 36px rgba(255,215,0,0.6)  !important; }
            .stat-cyan:hover  { box-shadow: 0 0 36px rgba(0,245,255,0.6)  !important; }
            .stat-green:hover { box-shadow: 0 0 36px rgba(0,255,156,0.6)  !important; }
            .stat-red:hover   { box-shadow: 0 0 36px rgba(255,77,109,0.6) !important; }

            /* rainbow animated border around chart */
            .chart-panel {
                position: relative;
                border-radius: 22px;
                padding: 3px;
                background: linear-gradient(135deg,#FFD700,#00F5FF,#BF5AF2,#FF4D6D,#FFD700);
                background-size: 300% 300%;
                animation: borderSpin 6s linear infinite;
            }
            @keyframes borderSpin {
                0%   { background-position: 0% 50%; }
                50%  { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            .chart-inner {
                background: #112240;
                border-radius: 19px;
                overflow: hidden;
            }

            /* title glow */
            .title-glow {
                text-shadow: 0 0 24px rgba(255,215,0,0.9),
                             0 0 60px rgba(255,215,0,0.4),
                             0 0 120px rgba(255,215,0,0.15);
            }

            /* label pills */
            .label-pill {
                display: inline-block;
                padding: 4px 14px;
                border-radius: 20px;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 2px;
                margin-bottom: 10px;
            }

            /* shimmer footer */
            .footer-text {
                background: linear-gradient(90deg,#8892A4,#FFD700,#00F5FF,#BF5AF2,#8892A4);
                background-size: 300% auto;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: shimmer 5s linear infinite;
            }
            @keyframes shimmer { to { background-position: 300% center; } }

            /* dropdown internals */
            .Select-control        { background-color: #112240 !important; border: none !important; }
            .Select-value-label    { color: #000 !important; font-weight: 600 !important; }
            .Select-placeholder    { color: #8892A4 !important; }
            .Select-input input    { color: #000 !important; }
            .Select-menu-outer     { background-color: #112240 !important;
                                     border: 1px solid #1E3A5F !important;
                                     box-shadow: 0 8px 32px rgba(0,245,255,0.15) !important; }
            .Select-option         { background-color: #112240 !important; color: #E2E8F0 !important; }
            .Select-option:hover,
            .Select-option.is-focused { background-color: #1A3A5C !important; color: #FFD700 !important; }
            .Select-option.is-selected { background-color: #1E3A5F !important; color: #00F5FF !important; }
            .Select-arrow          { border-top-color: #8892A4 !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ── Stat card helper ──────────────────────────────────────────
def stat_card(value, label, icon, color, glow_class, border_rgba):
    return dbc.Col(html.Div([
        html.Div(icon, style={"fontSize": "22px", "marginBottom": "6px",
                               "filter": f"drop-shadow(0 0 8px {color})"}),
        html.Div(value, style={
            "color": color, "fontSize": "28px", "fontWeight": "800",
            "fontFamily": "Orbitron, sans-serif", "lineHeight": "1",
            "textShadow": f"0 0 16px {color}",
        }),
        html.Div(label, style={
            "color": SUB, "fontSize": "10px", "letterSpacing": "3px",
            "marginTop": "6px", "textTransform": "uppercase", "fontWeight": "600",
        }),
        html.Div(style={
            "width": "36px", "height": "3px", "borderRadius": "2px",
            "backgroundColor": color, "margin": "10px auto 0",
            "boxShadow": f"0 0 12px {color}",
        }),
    ], className=f"stat-card {glow_class}", style={
        "textAlign": "center", "padding": "22px 16px",
        "backgroundColor": CARD2,
        "borderRadius": "18px",
        "border": f"1px solid {border_rgba}",
        "boxShadow": "0 4px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04)",
    }), md=3)

STAT_ROW = dbc.Row([
    stat_card(f"{int(df['TotalGoals'].sum()):,}", "Total Goals",      "⚽", GOLD,   "stat-gold",  "rgba(255,215,0,0.3)"),
    stat_card(f"{df['TotalGoals'].mean():.1f}",   "Avg Goals/Match",  "📊", ACCENT, "stat-cyan",  "rgba(0,245,255,0.3)"),
    stat_card(str(df["Home Team Name"].nunique()), "Unique Nations",   "🌍", GREEN,  "stat-green", "rgba(0,255,156,0.3)"),
    stat_card(str(df["Stage"].nunique()),          "Match Stages",     "🏆", RED,    "stat-red",   "rgba(255,77,109,0.3)"),
], className="g-3", style={"marginBottom": "32px"})

# ── Header ───────────────────────────────────────────────────
HEADER = html.Div([
    html.Div([
        html.Div("⚽", style={
            "fontSize": "56px", "lineHeight": "1",
            "filter": "drop-shadow(0 0 20px rgba(255,215,0,0.9))",
            "animation": "none",
        }),
        html.Div([
            html.H1("FIFA WORLD CUP", className="title-glow", style={
                "fontFamily": "Orbitron, sans-serif",
                "fontSize": "clamp(18px,4vw,44px)",
                "color": GOLD, "letterSpacing": "8px", "margin": "0",
            }),
            html.P("✦  Interactive Data Visualization Dashboard  ✦", style={
                "color": ACCENT, "fontSize": "11px", "letterSpacing": "4px",
                "margin": "8px 0 0 2px", "fontWeight": "600",
                "textShadow": f"0 0 12px {ACCENT}",
            }),
        ]),
    ], style={"display": "flex", "alignItems": "center", "gap": "20px"}),

    html.Div([
        html.Div([
            html.Span(f"{df.shape[0]:,} ", style={
                "color": GOLD, "fontWeight": "800", "fontSize": "22px",
                "fontFamily": "Orbitron", "textShadow": f"0 0 12px {GOLD}",
            }),
            html.Span("matches", style={"color": SUB, "fontSize": "11px"}),
        ]),
        html.Div([
            html.Span(f"{df['Year'].nunique()} ", style={
                "color": GREEN, "fontWeight": "800", "fontSize": "22px",
                "fontFamily": "Orbitron", "textShadow": f"0 0 12px {GREEN}",
            }),
            html.Span("tournaments", style={"color": SUB, "fontSize": "11px"}),
        ]),
        html.Div([
            html.Span(f"{df['Home Team Name'].nunique()} ", style={
                "color": ACCENT, "fontWeight": "800", "fontSize": "22px",
                "fontFamily": "Orbitron", "textShadow": f"0 0 12px {ACCENT}",
            }),
            html.Span("nations", style={"color": SUB, "fontSize": "11px"}),
        ]),
    ], style={
        "marginLeft": "auto", "display": "flex",
        "flexDirection": "column", "gap": "4px", "textAlign": "right",
    }),
], style={
    "display": "flex", "alignItems": "center", "justifyContent": "space-between",
    "marginBottom": "32px", "paddingBottom": "24px",
    "borderBottom": "1px solid rgba(255,215,0,0.15)",
})

# ── Controls ─────────────────────────────────────────────────
CONTROLS = dbc.Row([
    dbc.Col([
        html.Div("📊  GRAPH TYPE", className="label-pill", style={
            "backgroundColor": "rgba(255,215,0,0.1)",
            "color": GOLD, "border": "1px solid rgba(255,215,0,0.35)",
        }),
        dcc.Dropdown(
            id="graph-select", options=GRAPH_OPTIONS, value="bar_goals_year",
            placeholder="🔍  Search graph type...", searchable=True, clearable=False,
            style={"backgroundColor": CARD2, "color": "black",
                   "border": "1px solid rgba(255,215,0,0.4)", "borderRadius": "12px",
                   "boxShadow": "0 0 20px rgba(255,215,0,0.12)"},
        ),
    ], md=6),
    dbc.Col([
        html.Div("🏷️  ATTRIBUTE", className="label-pill", style={
            "backgroundColor": "rgba(0,245,255,0.1)",
            "color": ACCENT, "border": "1px solid rgba(0,245,255,0.35)",
        }),
        dcc.Dropdown(
            id="attr-select", options=ATTR_OPTIONS, value="Year",
            placeholder="🔍  Search attribute...", searchable=True, clearable=False,
            style={"backgroundColor": CARD2, "color": "black",
                   "border": "1px solid rgba(0,245,255,0.4)", "borderRadius": "12px",
                   "boxShadow": "0 0 20px rgba(0,245,255,0.12)"},
        ),
    ], md=6),
], style={"marginBottom": "28px"})

# ── Layout ────────────────────────────────────────────────────
app.layout = html.Div(
    style={"minHeight": "100vh", "padding": "32px 24px",
           "fontFamily": "Rajdhani, sans-serif"},
    children=[
        html.Div(style={
            "maxWidth": "1360px", "margin": "0 auto",
            "backgroundColor": CARD, "borderRadius": "28px", "padding": "40px",
            "boxShadow": (
                "0 0 0 1px rgba(255,215,0,0.1), "
                "0 0 100px rgba(255,215,0,0.05), "
                "0 40px 80px rgba(0,0,0,0.7)"
            ),
        }, children=[
            HEADER,
            STAT_ROW,
            CONTROLS,

            # animated rainbow border chart panel
            html.Div(className="chart-panel", children=[
                html.Div(className="chart-inner", children=[
                    dcc.Loading(
                        id="loading", type="circle", color=GOLD,
                        children=dcc.Graph(
                            id="main-chart",
                            config={"displayModeBar": True, "scrollZoom": True},
                            style={"height": "590px"},
                        ),
                    ),
                ]),
            ]),

            html.Div(id="chart-info", style={
                "marginTop": "18px", "color": SUB,
                "fontSize": "11px", "textAlign": "center",
                "letterSpacing": "2px", "fontWeight": "500",
            }),
        ]),

        html.Div([
            html.Span(
                "⚽  FIFA World Cup  ·  Data Visualization Dashboard  ·  "
                "Python | Pandas | Plotly | Dash  ·  ⚽",
                className="footer-text"
            ),
        ], style={"textAlign": "center", "marginTop": "28px",
                  "fontSize": "11px", "letterSpacing": "2px"}),
    ],
)


# ── Callback ──────────────────────────────────────────────────
@app.callback(
    Output("main-chart", "figure"),
    Output("chart-info",  "children"),
    Input("graph-select", "value"),
    Input("attr-select",  "value"),
)
def update_chart(graph_val, attr_val):
    fig  = build_figure(graph_val, attr_val)
    info = f"▸  Graph: {graph_val}   ·   Attribute: {attr_val}   ·   {df.shape[0]:,} records"
    return fig, info


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "━" * 55)
    print("  ⚽  FIFA World Cup Dashboard  —  launching …")
    print("  📡  Open your browser → http://127.0.0.1:8050")
    print("━" * 55 + "\n")
    app.run(debug=False, port=8050)
