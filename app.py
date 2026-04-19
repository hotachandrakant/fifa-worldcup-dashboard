import numpy as np
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output

# ===============================
# LOAD DATA (IMPORTANT CHANGE)
# ===============================
# Put CSV file in same folder as app.py
df = pd.read_csv("WorldCupMatches.csv")

# Basic cleaning
df = df.dropna()

# Create new feature (optional but useful)
df["Total Goals"] = df["Home Team Goals"] + df["Away Team Goals"]

# ===============================
# DASH APP
# ===============================
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("FIFA World Cup Dashboard", style={'textAlign': 'center'}),

    # Graph Type Dropdown
    dcc.Dropdown(
        id='graph-type',
        options=[
            {'label': 'Bar Chart', 'value': 'bar'},
            {'label': 'Line Chart', 'value': 'line'},
            {'label': 'Scatter Plot', 'value': 'scatter'},
            {'label': 'Histogram', 'value': 'hist'}
        ],
        value='bar',
        style={'width': '50%', 'margin': 'auto'}
    ),

    # Attribute Dropdown
    dcc.Dropdown(
        id='attribute',
        options=[
            {'label': col, 'value': col}
            for col in df.select_dtypes(include=np.number).columns
        ],
        value='Total Goals',
        style={'width': '50%', 'margin': '20px auto'}
    ),

    dcc.Graph(id='graph-output')
])

# ===============================
# CALLBACK FUNCTION
# ===============================
@app.callback(
    Output('graph-output', 'figure'),
    Input('graph-type', 'value'),
    Input('attribute', 'value')
)
def update_graph(graph_type, attribute):

    if graph_type == 'bar':
        fig = px.bar(df, x='Year', y=attribute)

    elif graph_type == 'line':
        fig = px.line(df, x='Year', y=attribute)

    elif graph_type == 'scatter':
        fig = px.scatter(df, x='Year', y=attribute)

    elif graph_type == 'hist':
        fig = px.histogram(df, x=attribute)

    return fig

# ===============================
# RUN APP
# ===============================
if __name__ == '__main__':
    app.run(debug=True)