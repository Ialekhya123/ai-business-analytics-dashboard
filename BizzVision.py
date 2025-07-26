import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
import datetime

# Load dataset
df = pd.read_csv(r"C:\Users\alekh\PycharmProjects\PythonProject\electronics_sales_with_names.csv", encoding="ISO-8859-1")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df.dropna(subset=['CustomerID', 'StockCode'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df = df[df['TotalPrice'] > 0]

# Sales Forecasting Model
sales_data = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum().reset_index()
sales_data.columns = ['ds', 'y']
model = Prophet()
model.fit(sales_data)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Customer Segmentation
customer_data = df.groupby('CustomerID').agg({'TotalPrice': 'sum', 'InvoiceNo': 'count'}).reset_index()
scaler = StandardScaler()
customer_scaled = scaler.fit_transform(customer_data[['TotalPrice', 'InvoiceNo']])
kmeans = KMeans(n_clusters=4, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(customer_scaled)

# Product Recommendation System
product_data = df.pivot_table(index='CustomerID', columns='StockCode', values='TotalPrice', fill_value=0)
nbrs = NearestNeighbors(n_neighbors=5, metric='cosine').fit(product_data)

def recommend_products(customer_id):
    if customer_id not in product_data.index:
        return ["Customer not found"]
    distances, indices = nbrs.kneighbors([product_data.loc[customer_id]])
    recommended_products = product_data.iloc[indices[0]].sum().sort_values(ascending=False).index[:5]
    return recommended_products.tolist()

# Fraud Detection
X = df[['Quantity', 'UnitPrice', 'TotalPrice']]
iso_forest = IsolationForest(contamination=0.01, random_state=42)
df['FraudScore'] = iso_forest.fit_predict(X)
df['Risk Level'] = df['FraudScore'].map({-1: 'High Risk', 1: 'Low Risk'})
fraud_cases = df[df['FraudScore'] == -1]

# Dash App
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("AI-Powered Business Analytics Dashboard", style={'textAlign': 'center'}),
    dcc.Tabs(id='tabs', children=[
        dcc.Tab(label=' Sales Forecasting', children=[
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=sales_data['ds'].min(),
                max_date_allowed=sales_data['ds'].max(),
                start_date=sales_data['ds'].min(),
                end_date=sales_data['ds'].max()
            ),
            dcc.Graph(id='sales-forecast-graph')
        ]),
        dcc.Tab(label=' Customer Segmentation', children=[
            dcc.Dropdown(
                id='cluster-dropdown',
                options=[{'label': f'Cluster {i}', 'value': i} for i in range(4)],
                multi=True,
                placeholder='Select Clusters'
            ),
            dcc.Graph(id='customer-segmentation-graph')
        ]),
        dcc.Tab(label='Product Recommendations', children=[
            dcc.Dropdown(
                id='customer-dropdown',
                options=[{'label': str(cust), 'value': cust} for cust in df['CustomerID'].unique()],
                placeholder='Select Customer'
            ),
            html.Div(id='recommend-output')
        ]),
        dcc.Tab(label='Fraud Detection', children=[
            dcc.Dropdown(
                id='fraud-filter',
                options=[
                    {'label': 'All', 'value': 'All'},
                    {'label': 'High Risk', 'value': 'High Risk'},
                    {'label': 'Low Risk', 'value': 'Low Risk'}
                ],
                value='All',
                placeholder='Filter Fraud Cases'
            ),
            dash_table.DataTable(id='fraud-table')
        ])
    ])
])

@app.callback(
    Output('sales-forecast-graph', 'figure'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date')
)
def update_forecast(start_date, end_date):
    filtered_forecast = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]
    return px.line(filtered_forecast, x='ds', y='yhat', title='Sales Forecast')

@app.callback(
    Output('customer-segmentation-graph', 'figure'),
    Input('cluster-dropdown', 'value')
)
def update_customer_segmentation(selected_clusters):
    filtered_data = customer_data if not selected_clusters else customer_data[customer_data['Cluster'].isin(selected_clusters)]
    return px.scatter(filtered_data, x='TotalPrice', y='InvoiceNo', color='Cluster', title='Customer Segmentation')

@app.callback(
    Output('recommend-output', 'children'),
    Input('customer-dropdown', 'value')
)
def update_recommendations(customer_id):
    if customer_id:
        recommended = recommend_products(customer_id)
        return f"Recommended Products: {', '.join(map(str, recommended))}"
    return ""

@app.callback(
    Output('fraud-table', 'data'),
    Input('fraud-filter', 'value')
)
def update_fraud_table(filter_value):
    filtered_df = df if filter_value == 'All' else df[df['Risk Level'] == filter_value]
    return filtered_df.to_dict('records')

if __name__ == '__main__':
    app.run(debug=True)
