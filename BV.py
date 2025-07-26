import dash
from dash import dcc, html, Input, Output, dash_table, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
import datetime
import time
import threading
from datetime import datetime, timedelta
import random

# Global variables for real-time data
real_time_data = []
last_update_time = datetime.now()
is_real_time_enabled = True

# External stylesheets for better styling
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
]

df = pd.read_csv("electronics_sales_with_names.csv", encoding="ISO-8859-1")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df.dropna(subset=['CustomerID', 'StockCode'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df = df[df['TotalPrice'] > 0]

# Real-time data generation function
def generate_real_time_data():
    """Generate simulated real-time sales data"""
    global real_time_data, last_update_time
    
    while is_real_time_enabled:
        try:
            # Simulate new sales data
            current_time = datetime.now()
            
            # Generate random new transactions
            for _ in range(random.randint(1, 5)):  # 1-5 new transactions per update
                new_transaction = {
                    'timestamp': current_time,
                    'customer_id': random.choice(df['CustomerID'].unique()),
                    'product_name': random.choice(df['ProductName'].unique()),
                    'quantity': random.randint(1, 10),
                    'unit_price': random.uniform(10, 1000),
                    'total_price': 0
                }
                new_transaction['total_price'] = new_transaction['quantity'] * new_transaction['unit_price']
                real_time_data.append(new_transaction)
            
            # Keep only last 1000 transactions for performance
            if len(real_time_data) > 1000:
                real_time_data = real_time_data[-1000:]
            
            last_update_time = current_time
            time.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            print(f"Error in real-time data generation: {e}")
            time.sleep(5)

# Start real-time data generation in background thread
real_time_thread = threading.Thread(target=generate_real_time_data, daemon=True)
real_time_thread.start()

# Simple forecasting using moving average instead of Prophet
sales_data = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum().reset_index()
sales_data.columns = ['ds', 'y']
sales_data['ds'] = pd.to_datetime(sales_data['ds'])

# Create a simple forecast using moving average
def simple_forecast(data, periods=30):
    # Calculate moving average
    ma = data['y'].rolling(window=7).mean()
    
    # Create future dates
    last_date = data['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
    
    # Use the last moving average value for forecast
    last_ma = ma.iloc[-1]
    forecast_values = [last_ma] * periods
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': forecast_values
    })
    
    return forecast_df

# Generate forecast
forecast = simple_forecast(sales_data, 30)

# Real-time data processing functions
def get_real_time_sales_data():
    """Get real-time sales data for the last hour"""
    global real_time_data
    
    if not real_time_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    rt_df = pd.DataFrame(real_time_data)
    rt_df['timestamp'] = pd.to_datetime(rt_df['timestamp'])
    
    # Get data from last hour
    one_hour_ago = datetime.now() - timedelta(hours=1)
    recent_data = rt_df[rt_df['timestamp'] >= one_hour_ago]
    
    return recent_data

def get_real_time_stats():
    """Get real-time statistics"""
    global real_time_data, df
    
    # Combine historical and real-time data
    rt_df = pd.DataFrame(real_time_data) if real_time_data else pd.DataFrame()
    
    if not rt_df.empty:
        rt_df['timestamp'] = pd.to_datetime(rt_df['timestamp'])
        # Get last 24 hours of real-time data
        last_24h = datetime.now() - timedelta(hours=24)
        recent_rt_data = rt_df[rt_df['timestamp'] >= last_24h]
        
        # Combine with historical data
        combined_df = pd.concat([df, recent_rt_data[['customer_id', 'product_name', 'quantity', 'unit_price', 'total_price']].rename(columns={
            'customer_id': 'CustomerID',
            'product_name': 'ProductName',
            'total_price': 'TotalPrice'
        })], ignore_index=True)
    else:
        combined_df = df
    
    total_sales = combined_df['TotalPrice'].sum()
    total_customers = combined_df['CustomerID'].nunique()
    total_products = combined_df['ProductName'].nunique()
    avg_order_value = combined_df.groupby('InvoiceNo')['TotalPrice'].sum().mean() if 'InvoiceNo' in combined_df.columns else combined_df['TotalPrice'].mean()
    
    return total_sales, total_customers, total_products, avg_order_value


customer_data = df.groupby('CustomerID').agg({'TotalPrice': 'sum', 'InvoiceNo': 'count'}).reset_index()
scaler = StandardScaler()
customer_scaled = scaler.fit_transform(customer_data[['TotalPrice', 'InvoiceNo']])
kmeans = KMeans(n_clusters=4, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(customer_scaled)

product_data = df.pivot_table(index='CustomerID', columns='ProductName', values='TotalPrice', fill_value=0)
nbrs = NearestNeighbors(n_neighbors=5, metric='cosine').fit(product_data)

# Enhanced recommendation algorithms
def get_customer_profile(customer_id):
    """Get detailed customer profile"""
    customer_data = df[df['CustomerID'] == customer_id]
    if customer_data.empty:
        return None
    
    profile = {
        'total_spent': customer_data['TotalPrice'].sum(),
        'avg_order_value': customer_data.groupby('InvoiceNo')['TotalPrice'].sum().mean(),
        'total_orders': customer_data['InvoiceNo'].nunique(),
        'favorite_products': customer_data['ProductName'].value_counts().head(3).index.tolist(),
        'avg_quantity': customer_data['Quantity'].mean(),
        'price_preference': customer_data['UnitPrice'].mean(),
        'purchase_frequency': customer_data['InvoiceNo'].nunique() / max(1, (customer_data['InvoiceDate'].max() - customer_data['InvoiceDate'].min()).days),
        'total_products_bought': customer_data['ProductName'].nunique()
    }
    return profile

def collaborative_filtering_recommendations(customer_id, n_recommendations=5):
    """Collaborative filtering based on similar customers"""
    try:
        if customer_id not in product_data.index:
            return []
        
        customer_purchases = df[df['CustomerID'] == customer_id]['ProductName'].unique()
        distances, indices = nbrs.kneighbors([product_data.loc[customer_id]])
        similar_customers = product_data.iloc[indices[0]]
        
        all_products = set(product_data.columns)
        customer_products = set(customer_purchases)
        potential_recommendations = all_products - customer_products
        
        if len(potential_recommendations) == 0:
            recommended_products = similar_customers.sum().sort_values(ascending=False).index[:n_recommendations]
        else:
            similar_customer_products = similar_customers[list(potential_recommendations)].sum().sort_values(ascending=False)
            recommended_products = similar_customer_products.index[:n_recommendations]
        
        return recommended_products.tolist()
    except Exception as e:
        print(f"Error in collaborative filtering: {e}")
        return []

def content_based_recommendations(customer_id, n_recommendations=5):
    """Content-based filtering based on customer's purchase history"""
    try:
        customer_profile = get_customer_profile(customer_id)
        if not customer_profile:
            return []
        
        # Get products similar to customer's favorite products
        favorite_products = customer_profile['favorite_products']
        price_preference = customer_profile['price_preference']
        
        # Find products with similar price range (±20%)
        price_range = (price_preference * 0.8, price_preference * 1.2)
        similar_price_products = df[
            (df['UnitPrice'] >= price_range[0]) & 
            (df['UnitPrice'] <= price_range[1])
        ]['ProductName'].unique()
        
        # Get products bought by customers who bought the same favorite products
        similar_customers = df[df['ProductName'].isin(favorite_products)]['CustomerID'].unique()
        similar_customer_products = df[df['CustomerID'].isin(similar_customers)]['ProductName'].value_counts()
        
        # Filter out already purchased products
        customer_purchases = df[df['CustomerID'] == customer_id]['ProductName'].unique()
        recommendations = [prod for prod in similar_customer_products.index if prod not in customer_purchases]
        
        return recommendations[:n_recommendations]
    except Exception as e:
        print(f"Error in content-based filtering: {e}")
        return []

def trending_products_recommendations(n_recommendations=5):
    """Get trending products based on recent sales"""
    try:
        # Get recent sales (last 30 days if available, otherwise use all data)
        recent_sales = df.groupby('ProductName')['TotalPrice'].sum().sort_values(ascending=False)
        return recent_sales.head(n_recommendations).index.tolist()
    except Exception as e:
        print(f"Error in trending products: {e}")
        return []

def price_based_recommendations(customer_id, budget_range='medium', n_recommendations=5):
    """Price-based recommendations"""
    try:
        customer_profile = get_customer_profile(customer_id)
        if not customer_profile:
            return []
        
        avg_spent = customer_profile['avg_order_value']
        
        # Define budget ranges
        if budget_range == 'low':
            price_range = (0, avg_spent * 0.5)
        elif budget_range == 'high':
            price_range = (avg_spent * 1.5, float('inf'))
        else:  # medium
            price_range = (avg_spent * 0.5, avg_spent * 1.5)
        
        # Get products in price range
        price_filtered_products = df[
            (df['UnitPrice'] >= price_range[0]) & 
            (df['UnitPrice'] <= price_range[1])
        ]['ProductName'].unique()
        
        # Get customer's purchases to exclude
        customer_purchases = df[df['CustomerID'] == customer_id]['ProductName'].unique()
        recommendations = [prod for prod in price_filtered_products if prod not in customer_purchases]
        
        return recommendations[:n_recommendations]
    except Exception as e:
        print(f"Error in price-based recommendations: {e}")
        return []

# Enhanced recommendation algorithms with multiple parameters
def get_customer_profile(customer_id):
    """Get detailed customer profile"""
    customer_data = df[df['CustomerID'] == customer_id]
    if customer_data.empty:
        return None
    
    profile = {
        'total_spent': customer_data['TotalPrice'].sum(),
        'avg_order_value': customer_data.groupby('InvoiceNo')['TotalPrice'].sum().mean(),
        'total_orders': customer_data['InvoiceNo'].nunique(),
        'favorite_products': customer_data['ProductName'].value_counts().head(3).index.tolist(),
        'avg_quantity': customer_data['Quantity'].mean(),
        'price_preference': customer_data['UnitPrice'].mean(),
        'purchase_frequency': customer_data['InvoiceNo'].nunique() / max(1, (customer_data['InvoiceDate'].max() - customer_data['InvoiceDate'].min()).days),
        'total_products_bought': customer_data['ProductName'].nunique()
    }
    return profile

def get_product_categories():
    """Extract product categories from product names"""
    # Simple category extraction based on product names
    categories = {}
    for product in df['ProductName'].unique():
        product_lower = product.lower()
        if any(word in product_lower for word in ['phone', 'mobile', 'cell']):
            categories[product] = 'Electronics'
        elif any(word in product_lower for word in ['laptop', 'computer', 'pc']):
            categories[product] = 'Computers'
        elif any(word in product_lower for word in ['book', 'magazine']):
            categories[product] = 'Books'
        elif any(word in product_lower for word in ['bag', 'handbag', 'purse']):
            categories[product] = 'Fashion'
        elif any(word in product_lower for word in ['toy', 'game']):
            categories[product] = 'Toys'
        else:
            categories[product] = 'Other'
    return categories

# Get product categories
product_categories = get_product_categories()

def recommend_by_product_category(category, price_range=None, n_recommendations=5):
    """Recommend products by category"""
    try:
        category_products = [prod for prod, cat in product_categories.items() if cat == category]
        
        if not category_products:
            return []
        
        # Filter by price range if specified
        if price_range:
            min_price, max_price = price_range
            category_products = df[
                (df['ProductName'].isin(category_products)) & 
                (df['UnitPrice'] >= min_price) & 
                (df['UnitPrice'] <= max_price)
            ]['ProductName'].unique()
        
        # Get most popular products in category
        category_sales = df[df['ProductName'].isin(category_products)].groupby('ProductName')['TotalPrice'].sum().sort_values(ascending=False)
        
        return category_sales.head(n_recommendations).index.tolist()
    except Exception as e:
        print(f"Error in category recommendations: {e}")
        return []

def recommend_by_price_range(min_price, max_price, n_recommendations=5):
    """Recommend products by price range"""
    try:
        price_filtered_products = df[
            (df['UnitPrice'] >= min_price) & 
            (df['UnitPrice'] <= max_price)
        ]['ProductName'].unique()
        
        # Get most popular products in price range
        price_sales = df[df['ProductName'].isin(price_filtered_products)].groupby('ProductName')['TotalPrice'].sum().sort_values(ascending=False)
        
        return price_sales.head(n_recommendations).index.tolist()
    except Exception as e:
        print(f"Error in price range recommendations: {e}")
        return []

def recommend_by_purchase_pattern(total_spent_range, order_frequency_range, n_recommendations=5):
    """Recommend products based on customer spending patterns"""
    try:
        # Find customers with similar spending patterns
        customer_patterns = df.groupby('CustomerID').agg({
            'TotalPrice': 'sum',
            'InvoiceNo': 'nunique'
        }).reset_index()
        
        customer_patterns.columns = ['CustomerID', 'TotalSpent', 'OrderCount']
        
        # Filter customers by spending pattern
        pattern_customers = customer_patterns[
            (customer_patterns['TotalSpent'] >= total_spent_range[0]) & 
            (customer_patterns['TotalSpent'] <= total_spent_range[1]) &
            (customer_patterns['OrderCount'] >= order_frequency_range[0]) & 
            (customer_patterns['OrderCount'] <= order_frequency_range[1])
        ]['CustomerID'].tolist()
        
        if not pattern_customers:
            return []
        
        # Get products bought by customers with similar patterns
        pattern_products = df[df['CustomerID'].isin(pattern_customers)]['ProductName'].value_counts()
        
        return pattern_products.head(n_recommendations).index.tolist()
    except Exception as e:
        print(f"Error in pattern recommendations: {e}")
        return []

def recommend_by_seasonal_trends(month=None, n_recommendations=5):
    """Recommend products based on seasonal trends"""
    try:
        if month is None:
            month = datetime.now().month
        
        # Get products sold in the specified month
        seasonal_products = df[df['InvoiceDate'].dt.month == month]['ProductName'].value_counts()
        
        return seasonal_products.head(n_recommendations).index.tolist()
    except Exception as e:
        print(f"Error in seasonal recommendations: {e}")
        return []

def recommend_by_product_similarity(target_product, n_recommendations=5):
    """Recommend products similar to a target product"""
    try:
        # Find customers who bought the target product
        target_customers = df[df['ProductName'] == target_product]['CustomerID'].unique()
        
        if len(target_customers) == 0:
            return []
        
        # Get other products bought by these customers
        similar_products = df[df['CustomerID'].isin(target_customers)]['ProductName'].value_counts()
        
        # Remove the target product from recommendations
        if target_product in similar_products.index:
            similar_products = similar_products.drop(target_product)
        
        return similar_products.head(n_recommendations).index.tolist()
    except Exception as e:
        print(f"Error in product similarity recommendations: {e}")
        return []

def recommend_by_brand_preference(brand_keywords, n_recommendations=5):
    """Recommend products by brand preference"""
    try:
        # Find products matching brand keywords
        brand_products = []
        for product in df['ProductName'].unique():
            product_lower = product.lower()
            if any(keyword.lower() in product_lower for keyword in brand_keywords):
                brand_products.append(product)
        
        if not brand_products:
            return []
        
        # Get most popular products from the brand
        brand_sales = df[df['ProductName'].isin(brand_products)].groupby('ProductName')['TotalPrice'].sum().sort_values(ascending=False)
        
        return brand_sales.head(n_recommendations).index.tolist()
    except Exception as e:
        print(f"Error in brand recommendations: {e}")
        return []

def recommend_by_customer_segment(segment_type, n_recommendations=5):
    """Recommend products based on customer segments"""
    try:
        if segment_type == 'budget':
            # Budget customers (low spending)
            segment_customers = customer_data[customer_data['Cluster'] == 0]['CustomerID'].tolist()
        elif segment_type == 'regular':
            # Regular customers (medium spending)
            segment_customers = customer_data[customer_data['Cluster'] == 1]['CustomerID'].tolist()
        elif segment_type == 'premium':
            # Premium customers (high spending)
            segment_customers = customer_data[customer_data['Cluster'] == 2]['CustomerID'].tolist()
        elif segment_type == 'vip':
            # VIP customers (very high spending)
            segment_customers = customer_data[customer_data['Cluster'] == 3]['CustomerID'].tolist()
        else:
            return []
        
        if not segment_customers:
            return []
        
        # Get products popular among this segment
        segment_products = df[df['CustomerID'].isin(segment_customers)]['ProductName'].value_counts()
        
        return segment_products.head(n_recommendations).index.tolist()
    except Exception as e:
        print(f"Error in segment recommendations: {e}")
        return []

def recommend_by_recent_trends(days=30, n_recommendations=5):
    """Recommend products based on recent trends"""
    try:
        # Get recent data
        recent_date = df['InvoiceDate'].max() - timedelta(days=days)
        recent_products = df[df['InvoiceDate'] >= recent_date]['ProductName'].value_counts()
        
        return recent_products.head(n_recommendations).index.tolist()
    except Exception as e:
        print(f"Error in recent trends recommendations: {e}")
        return []

def advanced_recommendations(recommendation_type, **kwargs):
    """Advanced recommendation system with multiple parameters"""
    try:
        n_recommendations = kwargs.get('n_recommendations', 5)
        
        if recommendation_type == 'category':
            category = kwargs.get('category', 'Electronics')
            price_range = kwargs.get('price_range')
            return recommend_by_product_category(category, price_range, n_recommendations)
        
        elif recommendation_type == 'price_range':
            min_price = kwargs.get('min_price', 0)
            max_price = kwargs.get('max_price', 1000)
            return recommend_by_price_range(min_price, max_price, n_recommendations)
        
        elif recommendation_type == 'purchase_pattern':
            total_spent_range = kwargs.get('total_spent_range', (0, 10000))
            order_frequency_range = kwargs.get('order_frequency_range', (1, 100))
            return recommend_by_purchase_pattern(total_spent_range, order_frequency_range, n_recommendations)
        
        elif recommendation_type == 'seasonal':
            month = kwargs.get('month')
            return recommend_by_seasonal_trends(month, n_recommendations)
        
        elif recommendation_type == 'product_similarity':
            target_product = kwargs.get('target_product')
            return recommend_by_product_similarity(target_product, n_recommendations)
        
        elif recommendation_type == 'brand':
            brand_keywords = kwargs.get('brand_keywords', [])
            return recommend_by_brand_preference(brand_keywords, n_recommendations)
        
        elif recommendation_type == 'segment':
            segment_type = kwargs.get('segment_type', 'regular')
            return recommend_by_customer_segment(segment_type, n_recommendations)
        
        elif recommendation_type == 'trending':
            days = kwargs.get('days', 30)
            return recommend_by_recent_trends(days, n_recommendations)
        
        elif recommendation_type == 'hybrid':
            # Combine multiple recommendation types
            all_recommendations = []
            
            # Get recommendations from different methods
            category_recs = recommend_by_product_category('Electronics', n_recommendations=n_recommendations)
            trending_recs = recommend_by_recent_trends(30, n_recommendations=n_recommendations)
            segment_recs = recommend_by_customer_segment('regular', n_recommendations=n_recommendations)
            
            # Combine and rank
            all_products = category_recs + trending_recs + segment_recs
            product_counts = {}
            
            for product in all_products:
                product_counts[product] = product_counts.get(product, 0) + 1
            
            # Sort by frequency and return top recommendations
            sorted_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
            return [product for product, count in sorted_products[:n_recommendations]]
        
        else:
            return {"error": "Invalid recommendation type"}
            
    except Exception as e:
        print(f"Error in advanced recommendations: {e}")
        return {"error": "Unable to generate recommendations"}

# Keep the original function for backward compatibility
def recommend_products(customer_id, algorithm='hybrid', budget_range='medium', n_recommendations=5):
    """Enhanced recommendation system with multiple algorithms"""
    try:
        if customer_id not in product_data.index:
            return {"error": "Customer not found"}
        
        recommendations = {}
        
        if algorithm == 'collaborative' or algorithm == 'hybrid':
            recommendations['collaborative'] = collaborative_filtering_recommendations(customer_id, n_recommendations)
        
        if algorithm == 'content' or algorithm == 'hybrid':
            recommendations['content'] = content_based_recommendations(customer_id, n_recommendations)
        
        if algorithm == 'trending' or algorithm == 'hybrid':
            recommendations['trending'] = trending_products_recommendations(n_recommendations)
        
        if algorithm == 'price' or algorithm == 'hybrid':
            recommendations['price'] = price_based_recommendations(customer_id, budget_range, n_recommendations)
        
        # For hybrid, combine and rank recommendations
        if algorithm == 'hybrid':
            all_recommendations = []
            for rec_type, recs in recommendations.items():
                for i, rec in enumerate(recs):
                    all_recommendations.append({
                        'product': rec,
                        'type': rec_type,
                        'score': len(recs) - i  # Higher score for higher ranked items
                    })
            
            # Count occurrences and calculate final scores
            product_scores = {}
            for rec in all_recommendations:
                product = rec['product']
                if product not in product_scores:
                    product_scores[product] = {'score': 0, 'types': []}
                product_scores[product]['score'] += rec['score']
                product_scores[product]['types'].append(rec['type'])
            
            # Sort by score and return top recommendations
            sorted_products = sorted(product_scores.items(), key=lambda x: x[1]['score'], reverse=True)
            final_recommendations = []
            
            for product, info in sorted_products[:n_recommendations]:
                final_recommendations.append({
                    'product': product,
                    'confidence': min(info['score'] / 10, 1.0),  # Normalize confidence
                    'types': info['types']
                })
            
            return final_recommendations
        
        # For single algorithm, return simple list
        if algorithm in recommendations:
            return [{'product': prod, 'confidence': 0.8, 'types': [algorithm]} for prod in recommendations[algorithm]]
        
        return {"error": "Invalid algorithm specified"}
        
    except Exception as e:
        print(f"Error in recommend_products: {e}")
        return {"error": "Unable to generate recommendations"}

X = df[['Quantity', 'UnitPrice', 'TotalPrice']]
iso_forest = IsolationForest(contamination=0.01, random_state=42)
df['FraudScore'] = iso_forest.fit_predict(X)
df['Risk Level'] = df['FraudScore'].map({-1: 'High Risk', 1: 'Low Risk'})
fraud_cases = df[df['FraudScore'] == -1]

# Calculate summary statistics
total_sales = df['TotalPrice'].sum()
total_customers = df['CustomerID'].nunique()
total_products = df['ProductName'].nunique()
avg_order_value = df.groupby('InvoiceNo')['TotalPrice'].sum().mean()

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Custom CSS for better styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>AI Business Analytics Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                font-family: 'Inter', sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .dashboard-container {
                padding: 20px;
                max-width: 1400px;
                margin: 0 auto;
            }
            .header {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                text-align: center;
            }
            .header h1 {
                margin: 0;
                color: #2d3748;
                font-size: 2.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 25px;
                text-align: center;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                transition: transform 0.3s ease;
            }
            .stat-card:hover {
                transform: translateY(-5px);
            }
            .stat-value {
                font-size: 2rem;
                font-weight: 700;
                color: #667eea;
                margin-bottom: 5px;
            }
            .stat-label {
                color: #718096;
                font-size: 0.9rem;
                font-weight: 500;
            }
            .tabs-container {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            .tab-content {
                padding: 20px 0;
            }
            .control-panel {
                background: rgba(255, 255, 255, 0.8);
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 20px;
                backdrop-filter: blur(10px);
            }
            .control-panel h3 {
                margin: 0 0 15px 0;
                color: #2d3748;
                font-size: 1.2rem;
                font-weight: 600;
            }
            .Select-control {
                border-radius: 10px !important;
                border: 2px solid #e2e8f0 !important;
            }
            .Select-control:hover {
                border-color: #667eea !important;
            }
            .DateInput_input {
                border-radius: 10px !important;
                border: 2px solid #e2e8f0 !important;
                padding: 10px !important;
            }
            .DateInput_input:focus {
                border-color: #667eea !important;
                outline: none !important;
            }
            .dash-table-container {
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            }
            .dash-spreadsheet-container {
                border-radius: 15px !important;
            }
            .dash-spreadsheet-inner {
                border-radius: 15px !important;
            }
            .dash-spreadsheet-container th {
                background: linear-gradient(135deg, #667eea, #764ba2) !important;
                color: white !important;
                font-weight: 600 !important;
            }
            .dash-spreadsheet-container td {
                border-bottom: 1px solid #e2e8f0 !important;
            }
            .dash-spreadsheet-container tr:nth-child(even) {
                background-color: #f7fafc !important;
            }
            .dash-spreadsheet-container tr:hover {
                background-color: #edf2f7 !important;
            }
            /* Additional styling for better appearance */
            .js-plotly-plot .plotly .main-svg {
                border-radius: 15px !important;
            }
            .dash-graph {
                background: rgba(255, 255, 255, 0.9) !important;
                border-radius: 15px !important;
                padding: 20px !important;
                margin: 10px 0 !important;
                backdrop-filter: blur(10px) !important;
            }
            .dash-dropdown .Select-control {
                background: rgba(255, 255, 255, 0.9) !important;
                backdrop-filter: blur(10px) !important;
            }
            .dash-datepicker .DateInput {
                background: rgba(255, 255, 255, 0.9) !important;
                backdrop-filter: blur(10px) !important;
                border-radius: 10px !important;
            }
            .dash-tab-label {
                background: rgba(255, 255, 255, 0.8) !important;
                border-radius: 10px 10px 0 0 !important;
                margin-right: 5px !important;
                padding: 15px 20px !important;
                font-weight: 600 !important;
                color: #2d3748 !important;
                transition: all 0.3s ease !important;
            }
            .dash-tab-label:hover {
                background: rgba(255, 255, 255, 0.95) !important;
                transform: translateY(-2px) !important;
            }
            .dash-tab-label--selected {
                background: rgba(102, 126, 234, 0.9) !important;
                color: white !important;
            }
            .dash-tab-content {
                background: rgba(255, 255, 255, 0.95) !important;
                border-radius: 0 15px 15px 15px !important;
                padding: 30px !important;
                margin-top: -5px !important;
            }
        </style>
        {%metas%}
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </body>
</html>
'''

app.layout = html.Div([
    # Real-time update interval
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Update every 5 seconds
        n_intervals=0
    ),
    
    # Real-time status indicator
    html.Div([
        html.Div([
            html.I(className="fas fa-circle", style={'color': '#48bb78', 'marginRight': '8px'}),
            html.Span("LIVE", style={'color': '#48bb78', 'fontWeight': 'bold'}),
            html.Span(" • Real-time updates every 5 seconds", style={'color': '#718096', 'marginLeft': '10px'})
        ], style={
            'position': 'fixed',
            'top': '10px',
            'right': '20px',
            'background': 'rgba(255, 255, 255, 0.95)',
            'padding': '8px 15px',
            'borderRadius': '20px',
            'backdropFilter': 'blur(10px)',
            'zIndex': '1000',
            'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'
        })
    ]),
    
    html.Div([
        html.Div([
            html.H1([
                html.I(className="fas fa-chart-line", style={'marginRight': '15px', 'color': '#667eea'}),
                "AI-Powered Business Analytics Dashboard"
            ]),
            html.P("Advanced analytics and insights for your business", 
                   style={'color': '#718096', 'fontSize': '1.1rem', 'margin': '10px 0 0 0'})
        ], className='header'),
        
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-dollar-sign", style={'fontSize': '2rem', 'color': '#667eea', 'marginBottom': '10px'}),
                        html.Div(id='total-sales-value', className='stat-value'),
                        html.Div("Total Sales", className='stat-label')
                    ])
                ], className='stat-card'),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-users", style={'fontSize': '2rem', 'color': '#667eea', 'marginBottom': '10px'}),
                        html.Div(id='total-customers-value', className='stat-value'),
                        html.Div("Total Customers", className='stat-label')
                    ])
                ], className='stat-card'),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-box", style={'fontSize': '2rem', 'color': '#667eea', 'marginBottom': '10px'}),
                        html.Div(id='total-products-value', className='stat-value'),
                        html.Div("Products Sold", className='stat-label')
                    ])
                ], className='stat-card'),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-shopping-cart", style={'fontSize': '2rem', 'color': '#667eea', 'marginBottom': '10px'}),
                        html.Div(id='avg-order-value', className='stat-value'),
                        html.Div("Avg Order Value", className='stat-label')
                    ])
                ], className='stat-card')
            ], className='stats-grid'),
            
            html.Div([
                dcc.Tabs(id='tabs', children=[
                    dcc.Tab(label="📈 Sales Forecasting", children=[
                        html.Div([
                            html.H3([
                                html.I(className="fas fa-calendar-alt", style={'marginRight': '8px'}),
                                "Select Date Range"
                            ]),
                            dcc.DatePickerRange(
                                id='date-picker',
                                min_date_allowed=sales_data['ds'].min(),
                                max_date_allowed=sales_data['ds'].max(),
                                start_date=sales_data['ds'].min(),
                                end_date=sales_data['ds'].max(),
                                style={'width': '100%'}
                            )
                        ], className='control-panel'),
                        dcc.Graph(id='sales-forecast-graph')
                    ]),
                    
                    dcc.Tab(label="👥 Customer Segmentation", children=[
                        html.Div([
                            html.H3([
                                html.I(className="fas fa-filter", style={'marginRight': '8px'}),
                                "Filter by Customer Clusters"
                            ]),
                            dcc.Dropdown(
                                id='cluster-dropdown',
                                options=[{'label': f'Cluster {i} - {["Budget", "Regular", "Premium", "VIP"][i]}', 'value': i} for i in range(4)],
                                multi=True,
                                placeholder='Select Clusters to Display',
                                style={'width': '100%'}
                            )
                        ], className='control-panel'),
                        dcc.Graph(id='customer-segmentation-graph')
                    ]),
                    
                    dcc.Tab(label="💡 Product Recommendations", children=[
                        html.Div([
                            html.H3([
                                html.I(className="fas fa-cog", style={'marginRight': '8px'}),
                                "Multi-Parameter Recommendation System"
                            ]),
                            
                            # Recommendation Type Selection
                            html.Div([
                                html.Label("Recommendation Type:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                                dcc.Dropdown(
                                    id='recommendation-type-dropdown',
                                    options=[
                                        {'label': '👤 Customer-Based (Traditional)', 'value': 'customer'},
                                        {'label': '📂 Product Category', 'value': 'category'},
                                        {'label': '💰 Price Range', 'value': 'price_range'},
                                        {'label': '📊 Purchase Patterns', 'value': 'purchase_pattern'},
                                        {'label': '🌤️ Seasonal Trends', 'value': 'seasonal'},
                                        {'label': '🔗 Product Similarity', 'value': 'product_similarity'},
                                        {'label': '🏷️ Brand Preference', 'value': 'brand'},
                                        {'label': '👥 Customer Segments', 'value': 'segment'},
                                        {'label': '📈 Recent Trends', 'value': 'trending'},
                                        {'label': '🤖 Hybrid (All Methods)', 'value': 'hybrid'}
                                    ],
                                    value='customer',
                                    style={'width': '100%'}
                                )
                            ], style={'marginBottom': '20px'}),
                            
                            # Dynamic Parameters Container
                            html.Div(id='dynamic-parameters-container', style={'marginBottom': '20px'}),
                            
                            # Number of Recommendations
                            html.Div([
                                html.Label("Number of Recommendations:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                                dcc.Slider(
                                    id='recommendations-slider',
                                    min=3,
                                    max=10,
                                    step=1,
                                    value=5,
                                    marks={i: str(i) for i in range(3, 11)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ])
                        ], className='control-panel'),
                        html.Div(id='recommend-output', style={
                            'background': 'rgba(255, 255, 255, 0.9)',
                            'padding': '20px',
                            'borderRadius': '15px',
                            'marginTop': '20px',
                            'backdropFilter': 'blur(10px)'
                        })
                    ]),
                    
                    dcc.Tab(label="🛡️ Fraud Detection", children=[
                        html.Div([
                            html.H3([
                                html.I(className="fas fa-exclamation-triangle", style={'marginRight': '8px'}),
                                "Filter Transactions by Risk Level"
                            ]),
                            dcc.Dropdown(
                                id='fraud-filter',
                                options=[
                                    {'label': '🔍 All Transactions', 'value': 'All'},
                                    {'label': '⚠️ High Risk Transactions', 'value': 'High Risk'},
                                    {'label': '✅ Low Risk Transactions', 'value': 'Low Risk'}
                                ],
                                value='All',
                                placeholder='Select Risk Level',
                                style={'width': '100%'}
                            )
                        ], className='control-panel'),
                        dash_table.DataTable(
                            id='fraud-table',
                            columns=[
                                {"name": "Invoice No", "id": "InvoiceNo"},
                                {"name": "Customer ID", "id": "CustomerID"},
                                {"name": "Product", "id": "ProductName"},
                                {"name": "Quantity", "id": "Quantity"},
                                {"name": "Unit Price", "id": "UnitPrice"},
                                {"name": "Total Price", "id": "TotalPrice"},
                                {"name": "Risk Level", "id": "Risk Level"}
                            ],
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'textAlign': 'left',
                                'padding': '12px',
                                'fontFamily': 'Inter, sans-serif'
                            },
                            style_header={
                                'backgroundColor': '#667eea',
                                'color': 'white',
                                'fontWeight': 'bold'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'column_id': 'Risk Level', 'filter_query': '{Risk Level} = "High Risk"'},
                                    'backgroundColor': '#fed7d7',
                                    'color': '#c53030'
                                },
                                {
                                    'if': {'column_id': 'Risk Level', 'filter_query': '{Risk Level} = "Low Risk"'},
                                    'backgroundColor': '#c6f6d5',
                                    'color': '#22543d'
                                }
                            ],
                            page_size=10
                        )
                    ]),
                    
                    dcc.Tab(label="⚡ Real-Time Sales", children=[
                        html.Div([
                            html.H3([
                                html.I(className="fas fa-bolt", style={'marginRight': '8px'}),
                                "Live Sales Activity"
                            ]),
                            html.P("Real-time sales data from the last hour", 
                                   style={'color': '#718096', 'marginBottom': '20px'})
                        ], className='control-panel'),
                        dcc.Graph(id='real-time-sales-graph'),
                        html.Div([
                            html.H4("Recent Transactions", style={'marginBottom': '15px'}),
                            dash_table.DataTable(
                                id='recent-transactions-table',
                                columns=[
                                    {"name": "Time", "id": "timestamp"},
                                    {"name": "Customer", "id": "customer_id"},
                                    {"name": "Product", "id": "product_name"},
                                    {"name": "Quantity", "id": "quantity"},
                                    {"name": "Total", "id": "total_price"}
                                ],
                                style_table={'overflowX': 'auto'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '8px',
                                    'fontFamily': 'Inter, sans-serif'
                                },
                                style_header={
                                    'backgroundColor': '#667eea',
                                    'color': 'white',
                                    'fontWeight': 'bold'
                                },
                                page_size=5
                            )
                        ], style={
                            'background': 'rgba(255, 255, 255, 0.9)',
                            'padding': '20px',
                            'borderRadius': '15px',
                            'marginTop': '20px',
                            'backdropFilter': 'blur(10px)'
                        })
                    ])
                ], style={'backgroundColor': 'transparent'})
            ], className='tabs-container')
        ])
    ], className='dashboard-container')
])

# Add some additional CSS for better tab styling
app.css.append_css({
    'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
})

@app.callback(
    Output('sales-forecast-graph', 'figure'),
    Input('date-picker', 'start_date'),
    Input('date-picker', 'end_date')
)
def update_forecast(start_date, end_date):
    try:
        if start_date and end_date:
            # Convert string dates to datetime
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter data
            filtered_sales = sales_data[(sales_data['ds'] >= start_dt) & (sales_data['ds'] <= end_dt)]
            filtered_forecast = forecast[(forecast['ds'] >= start_dt) & (forecast['ds'] <= end_dt)]
            
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=filtered_sales['ds'],
                y=filtered_sales['y'],
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='#667eea', width=3),
                marker=dict(size=6, color='#667eea')
            ))
            
            # Add forecast if available
            if not filtered_forecast.empty:
                fig.add_trace(go.Scatter(
                    x=filtered_forecast['ds'],
                    y=filtered_forecast['yhat'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#764ba2', width=3, dash='dash'),
                    marker=dict(size=6, color='#764ba2')
                ))
            
            fig.update_layout(
                title={
                    'text': '📈 Sales Forecast Analysis',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24, 'color': '#2d3748'}
                },
                xaxis_title='Date',
                yaxis_title='Sales Amount ($)',
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
                paper_bgcolor='rgba(255, 255, 255, 0)',
                font={'family': 'Inter, sans-serif'},
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            return fig
        else:
            # Show all data if no date range selected
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sales_data['ds'],
                y=sales_data['y'],
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='#667eea', width=3),
                marker=dict(size=6, color='#667eea')
            ))
            
            fig.update_layout(
                title={
                    'text': '📈 Sales Data Overview',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24, 'color': '#2d3748'}
                },
                xaxis_title='Date',
                yaxis_title='Sales Amount ($)',
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
                paper_bgcolor='rgba(255, 255, 255, 0)',
                font={'family': 'Inter, sans-serif'},
                hovermode='x unified'
            )
            
            return fig
    except Exception as e:
        print(f"Error in forecast callback: {e}")
        return {}

@app.callback(
    Output('customer-segmentation-graph', 'figure'),
    Input('cluster-dropdown', 'value')
)
def update_customer_segmentation(selected_clusters):
    try:
        print(f"Customer segmentation callback triggered with: {selected_clusters}")
        print(f"Customer data shape: {customer_data.shape}")
        print(f"Customer data columns: {customer_data.columns.tolist()}")
        
        cluster_names = ['Budget', 'Regular', 'Premium', 'VIP']
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        
        filtered_data = customer_data if not selected_clusters else customer_data[customer_data['Cluster'].isin(selected_clusters)]
        print(f"Filtered data shape: {filtered_data.shape}")
        
        # Create a simpler scatter plot to avoid any issues
        fig = go.Figure()
        
        for cluster_id in range(4):
            cluster_data = filtered_data[filtered_data['Cluster'] == cluster_id]
            if not cluster_data.empty:
                fig.add_trace(go.Scatter(
                    x=cluster_data['TotalPrice'],
                    y=cluster_data['InvoiceNo'],
                    mode='markers',
                    name=cluster_names[cluster_id],
                    marker=dict(
                        size=cluster_data['TotalPrice'] / 1000,  # Scale size
                        color=colors[cluster_id],
                        opacity=0.7
                    ),
                    text=cluster_data['CustomerID'],
                    hovertemplate='Customer: %{text}<br>Spending: $%{x:,.2f}<br>Orders: %{y}<extra></extra>'
                ))
        
        fig.update_layout(
            title={
                'text': '👥 Customer Segmentation',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#2d3748'}
            },
            xaxis_title='Total Spending ($)',
            yaxis_title='Number of Orders',
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            paper_bgcolor='rgba(255, 255, 255, 0)',
            font={'family': 'Inter, sans-serif'},
            legend_title_text='Customer Segments',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update legend labels
        for i, name in enumerate(cluster_names):
            if i < len(fig.data):
                fig.data[i].name = name
        
        print("Customer segmentation figure created successfully")
        return fig
    except Exception as e:
        print(f"Error in segmentation callback: {e}")
        import traceback
        traceback.print_exc()
        return {}

# Callback to generate dynamic parameters based on recommendation type
@app.callback(
    Output('dynamic-parameters-container', 'children'),
    [Input('recommendation-type-dropdown', 'value')]
)
def update_dynamic_parameters(recommendation_type):
    if recommendation_type == 'customer':
        return html.Div([
            html.Div([
                html.Label("Select Customer:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='customer-dropdown',
                    options=[{'label': f"Customer {cust} - ${df[df['CustomerID']==cust]['TotalPrice'].sum():.2f}", 'value': cust} for cust in sorted(df['CustomerID'].unique())[:50]],
                    placeholder='Choose a customer to get personalized recommendations',
                    style={'width': '100%'}
                )
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.Label("Algorithm:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='algorithm-dropdown',
                    options=[
                        {'label': '🤖 Hybrid (All Algorithms)', 'value': 'hybrid'},
                        {'label': '👥 Collaborative Filtering', 'value': 'collaborative'},
                        {'label': '📊 Content-Based', 'value': 'content'},
                        {'label': '🔥 Trending Products', 'value': 'trending'},
                        {'label': '💰 Price-Based', 'value': 'price'}
                    ],
                    value='hybrid',
                    style={'width': '100%'}
                )
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.Label("Budget Range:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='budget-dropdown',
                    options=[
                        {'label': '💸 Low Budget', 'value': 'low'},
                        {'label': '💰 Medium Budget', 'value': 'medium'},
                        {'label': '💎 High Budget', 'value': 'high'}
                    ],
                    value='medium',
                    style={'width': '100%'}
                )
            ])
        ])
    
    elif recommendation_type == 'category':
        return html.Div([
            html.Div([
                html.Label("Product Category:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='category-dropdown',
                    options=[
                        {'label': '📱 Electronics', 'value': 'Electronics'},
                        {'label': '💻 Computers', 'value': 'Computers'},
                        {'label': '📚 Books', 'value': 'Books'},
                        {'label': '👗 Fashion', 'value': 'Fashion'},
                        {'label': '🎮 Toys', 'value': 'Toys'},
                        {'label': '📦 Other', 'value': 'Other'}
                    ],
                    value='Electronics',
                    style={'width': '100%'}
                )
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.Label("Price Range (Optional):", style={'fontWeight': '600', 'marginBottom': '5px'}),
                dcc.RangeSlider(
                    id='category-price-range',
                    min=0,
                    max=1000,
                    step=50,
                    value=[0, 1000],
                    marks={0: '$0', 250: '$250', 500: '$500', 750: '$750', 1000: '$1000'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ])
    
    elif recommendation_type == 'price_range':
        return html.Div([
            html.Div([
                html.Label("Price Range:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                dcc.RangeSlider(
                    id='price-range-slider',
                    min=0,
                    max=1000,
                    step=50,
                    value=[0, 500],
                    marks={0: '$0', 250: '$250', 500: '$500', 750: '$750', 1000: '$1000'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ])
    
    elif recommendation_type == 'purchase_pattern':
        return html.Div([
            html.Div([
                html.Label("Total Spending Range:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                dcc.RangeSlider(
                    id='spending-range-slider',
                    min=0,
                    max=10000,
                    step=500,
                    value=[0, 5000],
                    marks={0: '$0', 2500: '$2.5K', 5000: '$5K', 7500: '$7.5K', 10000: '$10K'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.Label("Order Frequency Range:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                dcc.RangeSlider(
                    id='frequency-range-slider',
                    min=1,
                    max=50,
                    step=1,
                    value=[1, 20],
                    marks={1: '1', 10: '10', 20: '20', 30: '30', 40: '40', 50: '50'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ])
    
    elif recommendation_type == 'seasonal':
        return html.Div([
            html.Div([
                html.Label("Month:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='month-dropdown',
                    options=[
                        {'label': 'January', 'value': 1},
                        {'label': 'February', 'value': 2},
                        {'label': 'March', 'value': 3},
                        {'label': 'April', 'value': 4},
                        {'label': 'May', 'value': 5},
                        {'label': 'June', 'value': 6},
                        {'label': 'July', 'value': 7},
                        {'label': 'August', 'value': 8},
                        {'label': 'September', 'value': 9},
                        {'label': 'October', 'value': 10},
                        {'label': 'November', 'value': 11},
                        {'label': 'December', 'value': 12}
                    ],
                    value=datetime.now().month,
                    style={'width': '100%'}
                )
            ])
        ])
    
    elif recommendation_type == 'product_similarity':
        return html.Div([
            html.Div([
                html.Label("Target Product:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='target-product-dropdown',
                    options=[{'label': prod, 'value': prod} for prod in sorted(df['ProductName'].unique())[:100]],
                    placeholder='Select a product to find similar items',
                    style={'width': '100%'}
                )
            ])
        ])
    
    elif recommendation_type == 'brand':
        return html.Div([
            html.Div([
                html.Label("Brand Keywords:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                dcc.Input(
                    id='brand-keywords-input',
                    type='text',
                    placeholder='Enter brand keywords (e.g., Apple, Samsung, Sony)',
                    style={'width': '100%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ddd'}
                )
            ])
        ])
    
    elif recommendation_type == 'segment':
        return html.Div([
            html.Div([
                html.Label("Customer Segment:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id='segment-dropdown',
                    options=[
                        {'label': '💸 Budget Customers', 'value': 'budget'},
                        {'label': '💰 Regular Customers', 'value': 'regular'},
                        {'label': '💎 Premium Customers', 'value': 'premium'},
                        {'label': '👑 VIP Customers', 'value': 'vip'}
                    ],
                    value='regular',
                    style={'width': '100%'}
                )
            ])
        ])
    
    elif recommendation_type == 'trending':
        return html.Div([
            html.Div([
                html.Label("Recent Days:", style={'fontWeight': '600', 'marginBottom': '5px'}),
                dcc.Slider(
                    id='trending-days-slider',
                    min=7,
                    max=90,
                    step=7,
                    value=30,
                    marks={7: '7 days', 30: '30 days', 60: '60 days', 90: '90 days'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ])
        ])
    
    else:  # hybrid
        return html.Div([
            html.P("Hybrid recommendations combine multiple methods automatically.", 
                   style={'color': '#718096', 'fontStyle': 'italic'})
        ])

@app.callback(
    Output('recommend-output', 'children'),
    [Input('recommendation-type-dropdown', 'value'),
     Input('recommendations-slider', 'value')]
)
def update_recommendations(recommendation_type, n_recommendations):
    try:
        print(f"Recommendations callback triggered with type: {recommendation_type}")
        
        # For now, let's show a simple message based on recommendation type
        # We'll implement the full functionality once we fix the component issues
        if not recommendation_type:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-info-circle", style={'fontSize': '3rem', 'color': '#667eea', 'marginBottom': '20px'}),
                    html.H4("Select Recommendation Type", style={'color': '#2d3748', 'marginBottom': '10px'}),
                    html.P("Choose a recommendation type from the dropdown above to get started", 
                           style={'color': '#718096', 'textAlign': 'center', 'lineHeight': '1.6'})
                ], style={'textAlign': 'center', 'padding': '40px 20px'})
            ])
        
        # Show different content based on recommendation type
        recommendation_info = {
            'customer': {
                'title': '👤 Customer-Based Recommendations',
                'description': 'Personalized recommendations based on customer behavior and preferences',
                'icon': '👤'
            },
            'category': {
                'title': '📂 Category-Based Recommendations',
                'description': 'Product recommendations by category with price filtering',
                'icon': '📂'
            },
            'price_range': {
                'title': '💰 Price Range Recommendations',
                'description': 'Products within your specified price range',
                'icon': '💰'
            },
            'purchase_pattern': {
                'title': '📊 Purchase Pattern Analysis',
                'description': 'Recommendations based on spending patterns and order frequency',
                'icon': '📊'
            },
            'seasonal': {
                'title': '🌤️ Seasonal Trends',
                'description': 'Products popular during specific months',
                'icon': '🌤️'
            },
            'product_similarity': {
                'title': '🔗 Product Similarity',
                'description': 'Products similar to your selected item',
                'icon': '🔗'
            },
            'brand': {
                'title': '🏷️ Brand Preferences',
                'description': 'Products from your preferred brands',
                'icon': '🏷️'
            },
            'segment': {
                'title': '👥 Customer Segment',
                'description': 'Products popular among specific customer segments',
                'icon': '👥'
            },
            'trending': {
                'title': '📈 Recent Trends',
                'description': 'Currently trending products',
                'icon': '📈'
            },
            'hybrid': {
                'title': '🤖 Hybrid Recommendations',
                'description': 'Combined recommendations from multiple algorithms',
                'icon': '🤖'
            }
        }
        
        info = recommendation_info.get(recommendation_type, {
            'title': 'Unknown Type',
            'description': 'Please select a valid recommendation type',
            'icon': '❓'
        })
        
        return html.Div([
            # Recommendation Info Section
            html.Div([
                html.H4([
                    html.Span(info['icon'], style={'marginRight': '8px'}),
                    info['title']
                ], style={'color': '#2d3748', 'marginBottom': '10px'}),
                html.P(info['description'], style={'color': '#718096', 'marginBottom': '0'})
            ], style={'background': 'rgba(102, 126, 234, 0.1)', 'padding': '20px', 'borderRadius': '15px', 'marginBottom': '25px'}),
            
            # Placeholder for recommendations
            html.Div([
                html.H5([
                    html.I(className="fas fa-lightbulb", style={'marginRight': '8px', 'color': '#667eea'}),
                    f"Recommendations (Coming Soon)"
                ], style={'color': '#2d3748', 'marginBottom': '15px'}),
                
                html.Div([
                    html.I(className="fas fa-cog", style={'fontSize': '2rem', 'color': '#667eea', 'marginBottom': '15px'}),
                    html.P("Dynamic parameter controls are being configured. Full functionality will be available shortly.", 
                           style={'color': '#718096', 'textAlign': 'center', 'lineHeight': '1.6'})
                ], style={'textAlign': 'center', 'padding': '30px 20px', 'background': 'rgba(255, 255, 255, 0.9)', 'borderRadius': '15px'})
            ])
        ])
            
    except Exception as e:
        print(f"Error in recommendations callback: {e}")
        return html.Div("Error loading recommendations", style={'color': 'red'})

@app.callback(
    Output('fraud-table', 'data'),
    Input('fraud-filter', 'value')
)
def update_fraud_table(filter_value):
    try:
        if filter_value == 'All':
            filtered_df = df.head(100)  # Limit to first 100 rows for performance
        else:
            filtered_df = df[df['Risk Level'] == filter_value].head(100)
        
        # Format the data for better display
        display_df = filtered_df.copy()
        display_df['TotalPrice'] = display_df['TotalPrice'].round(2)
        display_df['UnitPrice'] = display_df['UnitPrice'].round(2)
        
        return display_df.to_dict('records')
    except Exception as e:
        print(f"Error in fraud table callback: {e}")
        return []

# Real-time callbacks
@app.callback(
    [Output('total-sales-value', 'children'),
     Output('total-customers-value', 'children'),
     Output('total-products-value', 'children'),
     Output('avg-order-value', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_real_time_stats(n):
    try:
        total_sales, total_customers, total_products, avg_order_value = get_real_time_stats()
        return [
            f"${total_sales:,.0f}",
            f"{total_customers:,}",
            f"{total_products:,}",
            f"${avg_order_value:.2f}"
        ]
    except Exception as e:
        print(f"Error updating real-time stats: {e}")
        return ["$0", "0", "0", "$0.00"]

@app.callback(
    Output('real-time-sales-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_real_time_sales_graph(n):
    try:
        rt_data = get_real_time_sales_data()
        
        if rt_data.empty:
            # Return empty figure if no real-time data
            fig = go.Figure()
            fig.add_annotation(
                text="No real-time data available yet. Data will appear here as new transactions occur.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#718096")
            )
            fig.update_layout(
                title="⚡ Real-Time Sales Activity",
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
                paper_bgcolor='rgba(255, 255, 255, 0)',
                font={'family': 'Inter, sans-serif'}
            )
            return fig
        
        # Group by minute for better visualization
        rt_data['minute'] = rt_data['timestamp'].dt.floor('min')
        sales_by_minute = rt_data.groupby('minute')['total_price'].sum().reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sales_by_minute['minute'],
            y=sales_by_minute['total_price'],
            mode='lines+markers',
            name='Real-Time Sales',
            line=dict(color='#48bb78', width=3),
            marker=dict(size=8, color='#48bb78'),
            fill='tonexty',
            fillcolor='rgba(72, 187, 120, 0.1)'
        ))
        
        fig.update_layout(
            title={
                'text': '⚡ Real-Time Sales Activity (Last Hour)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#2d3748'}
            },
            xaxis_title='Time',
            yaxis_title='Sales Amount ($)',
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            paper_bgcolor='rgba(255, 255, 255, 0)',
            font={'family': 'Inter, sans-serif'},
            hovermode='x unified',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    except Exception as e:
        print(f"Error updating real-time sales graph: {e}")
        return {}

@app.callback(
    Output('recent-transactions-table', 'data'),
    [Input('interval-component', 'n_intervals')]
)
def update_recent_transactions(n):
    try:
        rt_data = get_real_time_sales_data()
        
        if rt_data.empty:
            return []
        
        # Get last 10 transactions
        recent_data = rt_data.tail(10).copy()
        recent_data['timestamp'] = recent_data['timestamp'].dt.strftime('%H:%M:%S')
        recent_data['total_price'] = recent_data['total_price'].round(2)
        
        return recent_data.to_dict('records')
    except Exception as e:
        print(f"Error updating recent transactions: {e}")
        return []

if __name__ == '__main__':
    app.run(debug=True)