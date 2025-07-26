# API Documentation

## Overview

This document provides detailed API documentation for the AI Business Analytics Dashboard functions and components.

## Core Functions

### Customer Profile Functions

#### `get_customer_profile(customer_id)`

Retrieves detailed customer profile information.

**Parameters:**
- `customer_id` (int): Unique customer identifier

**Returns:**
- `dict` or `None`: Customer profile containing:
  - `total_spent` (float): Total amount spent by customer
  - `avg_order_value` (float): Average order value
  - `total_orders` (int): Number of orders placed
  - `favorite_products` (list): Top 3 most purchased products
  - `avg_quantity` (float): Average quantity per order
  - `price_preference` (float): Average unit price preference
  - `purchase_frequency` (float): Orders per day
  - `total_products_bought` (int): Unique products purchased

**Example:**
```python
profile = get_customer_profile(12345)
if profile:
    print(f"Customer spent: ${profile['total_spent']:.2f}")
```

### Recommendation Algorithms

#### `collaborative_filtering_recommendations(customer_id, n_recommendations=5)`

Generates recommendations based on similar customers' behavior.

**Parameters:**
- `customer_id` (int): Target customer ID
- `n_recommendations` (int): Number of recommendations to return

**Returns:**
- `list`: List of recommended product names

#### `content_based_recommendations(customer_id, n_recommendations=5)`

Generates recommendations based on customer's own purchase history.

**Parameters:**
- `customer_id` (int): Target customer ID
- `n_recommendations` (int): Number of recommendations to return

**Returns:**
- `list`: List of recommended product names

#### `trending_products_recommendations(n_recommendations=5)`

Returns currently trending products based on sales volume.

**Parameters:**
- `n_recommendations` (int): Number of recommendations to return

**Returns:**
- `list`: List of trending product names

#### `price_based_recommendations(customer_id, budget_range='medium', n_recommendations=5)`

Generates recommendations based on customer's price preferences.

**Parameters:**
- `customer_id` (int): Target customer ID
- `budget_range` (str): 'low', 'medium', or 'high'
- `n_recommendations` (int): Number of recommendations to return

**Returns:**
- `list`: List of recommended product names

### Advanced Recommendation Functions

#### `recommend_by_product_category(category, price_range=None, n_recommendations=5)`

Recommends products within a specific category.

**Parameters:**
- `category` (str): Product category (Electronics, Computers, Books, etc.)
- `price_range` (tuple, optional): (min_price, max_price)
- `n_recommendations` (int): Number of recommendations to return

**Returns:**
- `list`: List of recommended product names

#### `recommend_by_price_range(min_price, max_price, n_recommendations=5)`

Recommends products within a specific price range.

**Parameters:**
- `min_price` (float): Minimum price
- `max_price` (float): Maximum price
- `n_recommendations` (int): Number of recommendations to return

**Returns:**
- `list`: List of recommended product names

#### `recommend_by_purchase_pattern(total_spent_range, order_frequency_range, n_recommendations=5)`

Recommends products based on customer spending patterns.

**Parameters:**
- `total_spent_range` (tuple): (min_spent, max_spent)
- `order_frequency_range` (tuple): (min_orders, max_orders)
- `n_recommendations` (int): Number of recommendations to return

**Returns:**
- `list`: List of recommended product names

#### `recommend_by_seasonal_trends(month=None, n_recommendations=5)`

Recommends products based on seasonal popularity.

**Parameters:**
- `month` (int, optional): Month number (1-12), defaults to current month
- `n_recommendations` (int): Number of recommendations to return

**Returns:**
- `list`: List of recommended product names

#### `recommend_by_product_similarity(target_product, n_recommendations=5)`

Recommends products similar to a target product.

**Parameters:**
- `target_product` (str): Name of the target product
- `n_recommendations` (int): Number of recommendations to return

**Returns:**
- `list`: List of recommended product names

#### `recommend_by_brand_preference(brand_keywords, n_recommendations=5)`

Recommends products from specific brands.

**Parameters:**
- `brand_keywords` (list): List of brand keywords
- `n_recommendations` (int): Number of recommendations to return

**Returns:**
- `list`: List of recommended product names

#### `recommend_by_customer_segment(segment_type, n_recommendations=5)`

Recommends products popular among specific customer segments.

**Parameters:**
- `segment_type` (str): 'budget', 'regular', 'premium', or 'vip'
- `n_recommendations` (int): Number of recommendations to return

**Returns:**
- `list`: List of recommended product names

#### `recommend_by_recent_trends(days=30, n_recommendations=5)`

Recommends products based on recent sales activity.

**Parameters:**
- `days` (int): Number of recent days to consider
- `n_recommendations` (int): Number of recommendations to return

**Returns:**
- `list`: List of recommended product names

### Master Recommendation Function

#### `advanced_recommendations(recommendation_type, **kwargs)`

Master function that routes to specific recommendation algorithms.

**Parameters:**
- `recommendation_type` (str): Type of recommendation algorithm
- `**kwargs`: Additional parameters specific to the recommendation type

**Supported Types:**
- `'category'`: Category-based recommendations
- `'price_range'`: Price range recommendations
- `'purchase_pattern'`: Purchase pattern recommendations
- `'seasonal'`: Seasonal trend recommendations
- `'product_similarity'`: Product similarity recommendations
- `'brand'`: Brand preference recommendations
- `'segment'`: Customer segment recommendations
- `'trending'`: Recent trends recommendations
- `'hybrid'`: Combined multiple algorithms

**Returns:**
- `list` or `dict`: Recommendations or error message

**Example:**
```python
# Category-based recommendations
recs = advanced_recommendations('category', category='Electronics', n_recommendations=5)

# Price range recommendations
recs = advanced_recommendations('price_range', min_price=0, max_price=100, n_recommendations=5)

# Hybrid recommendations
recs = advanced_recommendations('hybrid', n_recommendations=5)
```

## Data Processing Functions

### `simple_forecast(data, periods=30)`

Generates sales forecasts using moving averages.

**Parameters:**
- `data` (DataFrame): Historical sales data with 'ds' and 'y' columns
- `periods` (int): Number of periods to forecast

**Returns:**
- `DataFrame`: Forecast data with 'ds' and 'yhat' columns

### Real-Time Data Functions

#### `generate_real_time_data()`

Background function that generates simulated real-time sales data.

**Returns:**
- `None`: Updates global real_time_data list

#### `get_real_time_sales_data()`

Retrieves real-time sales data from the last hour.

**Returns:**
- `DataFrame`: Recent sales transactions

#### `get_real_time_stats()`

Calculates real-time business statistics.

**Returns:**
- `tuple`: (total_sales, total_customers, total_products, avg_order_value)

## Error Handling

All functions include comprehensive error handling:

- **Invalid Inputs**: Functions return empty lists or None for invalid inputs
- **Missing Data**: Graceful handling of missing or corrupted data
- **Performance**: Timeout protection for long-running operations

## Performance Considerations

- **Caching**: Customer profiles and recommendations are cached for performance
- **Batch Processing**: Large datasets are processed in batches
- **Memory Management**: Real-time data is limited to last 1000 transactions
- **Async Updates**: Background threads handle real-time data generation

## Usage Examples

### Basic Customer Recommendations
```python
# Get customer profile
profile = get_customer_profile(12345)

# Get collaborative filtering recommendations
recs = collaborative_filtering_recommendations(12345, 5)

# Get price-based recommendations
price_recs = price_based_recommendations(12345, 'medium', 5)
```

### Advanced Recommendations
```python
# Category-based recommendations
electronics = recommend_by_product_category('Electronics', price_range=(0, 500), n_recommendations=10)

# Seasonal recommendations
summer_products = recommend_by_seasonal_trends(6, 5)  # June

# Brand recommendations
apple_products = recommend_by_brand_preference(['Apple', 'iPhone'], 5)
```

### Real-Time Analytics
```python
# Get real-time statistics
total_sales, customers, products, avg_order = get_real_time_stats()

# Get recent sales data
recent_sales = get_real_time_sales_data()
```

## Data Schema

### Customer Profile Schema
```json
{
  "total_spent": 1500.50,
  "avg_order_value": 75.25,
  "total_orders": 20,
  "favorite_products": ["Product A", "Product B", "Product C"],
  "avg_quantity": 3.5,
  "price_preference": 45.75,
  "purchase_frequency": 0.1,
  "total_products_bought": 15
}
```

### Recommendation Schema
```json
{
  "product": "Product Name",
  "confidence": 0.85,
  "types": ["collaborative", "content"]
}
```

## Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning algorithms
- **plotly**: Data visualization
- **dash**: Web framework
- **threading**: Real-time processing 