import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet  # AI-Powered Sales Forecasting
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

# Load dataset
df = pd.read_csv("electronics_sales_with_names.csv", encoding="ISO-8859-1")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df.dropna(subset=['CustomerID', 'StockCode'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df = df[df['TotalPrice'] > 0]

# 1️⃣ Sales Forecasting
sales_data = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum().reset_index()
sales_data.columns = ['ds', 'y']
from prophet import Prophet
import logging

logging.getLogger("cmdstanpy").disabled = True  # Suppress cmdstanpy logs

model = Prophet()  # Remove stan_backend parameter
model.fit(sales_data)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# 2️⃣ Customer Segmentation
customer_data = df.groupby('CustomerID').agg({'TotalPrice': 'sum', 'InvoiceNo': 'count'}).reset_index()
scaler = StandardScaler()
customer_scaled = scaler.fit_transform(customer_data[['TotalPrice', 'InvoiceNo']])
kmeans = KMeans(n_clusters=4, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(customer_scaled)

# 3️⃣ Product Recommendation System using KNN
product_data = df.pivot_table(index='CustomerID', columns='StockCode', values='TotalPrice', fill_value=0)
nbrs = NearestNeighbors(n_neighbors=5, metric='cosine').fit(product_data)

def recommend_products(customer_id):
    if customer_id not in product_data.index:
        return ["Customer not found"]
    distances, indices = nbrs.kneighbors([product_data.loc[customer_id]])
    recommended_products = product_data.iloc[indices[0]].sum().sort_values(ascending=False).index[:5]
    return recommended_products.tolist()

# 4️⃣ Fraud Detection
X = df[['Quantity', 'UnitPrice', 'TotalPrice']]
iso_forest = IsolationForest(contamination=0.01, random_state=42)
df['Fraud'] = iso_forest.fit_predict(X)
fraud_cases = df[df['Fraud'] == -1]

# Dash App
app = dash.Dash(__name__)  # ✅ Correct


app.layout = html.Div(style={'backgroundColor': '#D2B48C', 'color': '#5C4033', 'padding': '20px'}, children=[
    html.H1("AI-Powered Business Analytics Dashboard", style={'textAlign': 'center', 'color': '#8B5A2B'}),
    dcc.Tabs(style={'fontWeight': 'bold', 'fontSize': '18px'}, children=[
        dcc.Tab(label='Sales Forecasting', children=[
            dcc.Graph(figure=px.line(forecast, x='ds', y='yhat', title='Sales Forecast',
                                     template='plotly', line_shape='spline'))
        ]),
        dcc.Tab(label='Customer Segmentation', children=[
            dcc.Graph(figure=px.scatter(customer_data, x='TotalPrice', y='InvoiceNo', color='Cluster',
                                        title='Customer Segmentation', template='plotly',
                                        size_max=15, opacity=0.9))
        ]),
        dcc.Tab(label='Product Recommendations', children=[
            dcc.Input(id='customer-id', type='number', placeholder='Enter Customer ID',
                      style={'margin': '10px', 'padding': '5px', 'fontSize': '16px'}),
            html.Button('Recommend', id='recommend-btn',
                        style={'backgroundColor': '#8B5A2B', 'color': 'white', 'borderRadius': '5px', 'padding': '10px'}),
            html.Div(id='recommend-output', style={'marginTop': '20px', 'fontSize': '18px'})
        ]),
        dcc.Tab(label='Fraud Detection', children=[
            dash_table.DataTable(
                data=fraud_cases.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in fraud_cases.columns],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#8B5A2B', 'fontWeight': 'bold', 'color': 'white'},
                style_data={'backgroundColor': '#F4E1C6', 'color': '#5C4033'}
            )
        ])
    ])
])

@app.callback(
    Output('recommend-output', 'children'),
    Input('recommend-btn', 'n_clicks'),
    Input('customer-id', 'value')
)
def update_recommendations(n_clicks, customer_id):
    if customer_id:
        recommended = recommend_products(customer_id)
        return f"Recommended Products: {', '.join(map(str, recommended))}"
    return ""

if __name__ == '__main__':
    app.run(debug=True)