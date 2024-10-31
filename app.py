import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load and clean data (cached as shown previously)
@st.cache_data
def load_and_clean_data(file_path):
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    data['TotalSales'] = data['Quantity'] * data['UnitPrice']
    return data[data['Quantity'] > 0]

data = load_and_clean_data('data/data.csv')
selected_country = "All"

title = st.sidebar.title("E-commerce Data Analysis")

# Sidebar for global country filter
selected_country = st.sidebar.selectbox("Filter by country", options=["All"] + data['Country'].unique().tolist())
filtered_data = data if selected_country == "All" else data[data['Country'] == selected_country]

title.title(f"E-commerce Data Analysis {'for ' + selected_country if selected_country != 'All' else ''}")

tab = st.sidebar.radio("Navigate", ["Data Exploration", "Sales Trends", "Product Performance", "Country Insights", "RFM Analysis"])

# Tab 1: Data Exploration
if tab == "Data Exploration":
    st.header("Data Exploration")
    st.write("### Sample of Filtered Data")
    st.dataframe(filtered_data.head(20))
    
    # Summary statistics
    total_rows = len(filtered_data)
    unique_products = filtered_data['StockCode'].nunique()
    unique_customers = filtered_data['CustomerID'].nunique()
    total_sales_value = filtered_data['TotalSales'].sum()
    
    st.write(f"**Total Rows:** {total_rows}")
    st.write(f"**Unique Products:** {unique_products}")
    st.write(f"**Unique Customers:** {unique_customers}")
    st.write(f"**Total Sales Value:** ${total_sales_value:,.2f}")

# Tab 2: Sales Trends
elif tab == "Sales Trends":
    st.header("Sales Trends Over Time")
    start_date, end_date = st.slider(
        'Select Date Range',
        min_value=filtered_data['InvoiceDate'].min().date(),
        max_value=filtered_data['InvoiceDate'].max().date(),
        value=(st.session_state.get('start_date', filtered_data['InvoiceDate'].min().date()), 
               st.session_state.get('end_date', filtered_data['InvoiceDate'].max().date()))
        # value=(data['InvoiceDate'].min().date(), data['InvoiceDate'].max().date())
    )
    
    st.session_state['start_date'], st.session_state['end_date'] = start_date, end_date

    filtered_time_data = filtered_data[(filtered_data['InvoiceDate'] >= pd.Timestamp(start_date)) & 
                                       (filtered_data['InvoiceDate'] <= pd.Timestamp(end_date))]

    daily_sales = filtered_time_data.groupby(filtered_time_data['InvoiceDate'].dt.date)['TotalSales'].sum()
    st.line_chart(daily_sales)
    st.write(f"### Total Sales from {start_date} to {end_date}: ${daily_sales.sum():,.2f}")

# Tab 3: Product Performance
elif tab == "Product Performance":
    st.header("Product Performance")
    product_performance = filtered_data.groupby('Description').agg({
        'TotalSales': 'sum',
        'Quantity': 'sum'
    }).reset_index()

    num_products = st.slider("Select number of top products to display", min_value=5, max_value=20, value=st.session_state.get('num_products', 10))
    st.session_state['num_products'] = num_products

    top_products = product_performance.sort_values(by='TotalSales', ascending=False).head(num_products)

    st.write(f"### Top {num_products} Products by Total Sales")
    st.bar_chart(top_products[['Description', 'TotalSales']].set_index('Description'))
    st.write(f"### Top {num_products} Products - Sales and Quantity Sold")
    st.dataframe(top_products[['Description', 'TotalSales', 'Quantity']])

# Tab 4: Country Insights
elif tab == "Country Insights":
    st.header("Country-Based Sales Insights")
    if selected_country == "All":
        country_sales = filtered_data.groupby('Country').agg({
            'TotalSales': 'sum',
            'CustomerID': 'nunique'
        }).reset_index()

        fig = px.choropleth(
            country_sales,
            locations="Country",
            locationmode="country names",
            color="TotalSales",
            hover_name="Country",
            hover_data=["TotalSales", "CustomerID"],
            title="Total Sales by Country",
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig)
        st.write("### Top 10 Countries by Total Sales")
        st.dataframe(country_sales.sort_values(by='TotalSales', ascending=False).head(10))
    else:
        st.write(f"### Sales and Customer Insights for {selected_country}")
        st.write(f"**Total Sales:** ${filtered_data['TotalSales'].sum():,.2f}")
        st.write(f"**Unique Customers:** {filtered_data['CustomerID'].nunique()}")

# RFM Analysis Tab
elif tab == "RFM Analysis":
    st.header("RFM Analysis")
    
    # Set a reference date for Recency calculations (e.g., the day after the last transaction)
    reference_date = filtered_data['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Calculate RFM metrics
    rfm = filtered_data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSales': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # RFM scoring with error-handling for duplicate bin edges
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])

    # Dynamic Frequency Score
    for bins in range(5, 1, -1):  # Start with 5 bins, reduce if duplicates are an issue
        try:
            rfm['F_Score'] = pd.qcut(rfm['Frequency'], bins, labels=range(1, bins + 1), duplicates="drop")
            break  # Exit loop if binning succeeds
        except ValueError:
            continue  # Try with one less bin if duplicates error occurs

    # Dynamic Monetary Score
    for bins in range(5, 1, -1):  # Start with 5 bins, reduce if duplicates are an issue
        try:
            rfm['M_Score'] = pd.qcut(rfm['Monetary'], bins, labels=range(1, bins + 1), duplicates="drop")
            break  # Exit loop if binning succeeds
        except ValueError:
            continue  # Try with one less bin if duplicates error occurs

    # Combine scores into RFM Segment and RFM Score
    rfm['RFM_Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
    
    # Display RFM table
    st.write("### RFM Table")
    st.dataframe(rfm[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'RFM_Segment', 'RFM_Score']])

    # Feature 1: Customer Count by Segment
    st.write("### Customer Count by RFM Segment")
    segment_counts = rfm['RFM_Segment'].value_counts()
    st.bar_chart(segment_counts)

    # Feature 2: Sales by Segment
    st.write("### Sales Summary by RFM Segment")
    sales_by_segment = rfm.groupby('RFM_Segment').agg({
        'Monetary': ['sum', 'mean'],
        'CustomerID': 'count'
    }).reset_index()
    sales_by_segment.columns = ['RFM_Segment', 'Total Sales', 'Average Sales per Customer', 'Customer Count']
    st.dataframe(sales_by_segment)

    # Feature 3: Top RFM Segments by Customer Count
    st.write("### Top 10 RFM Segments by Customer Count")
    top_segment_counts = segment_counts.head(10)
    st.bar_chart(top_segment_counts)

    # Additional Feature 4: RFM Score Distribution Histograms
    st.write("### RFM Metric Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(rfm['Recency'], bins=10, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title("Recency Distribution")
    sns.histplot(rfm['Frequency'], bins=10, kde=True, ax=axes[1], color='lightgreen')
    axes[1].set_title("Frequency Distribution")
    sns.histplot(rfm['Monetary'], bins=10, kde=True, ax=axes[2], color='salmon')
    axes[2].set_title("Monetary Distribution")
    st.pyplot(fig)

    # Additional Feature 5: RFM Heatmap (Recency vs Frequency, Average Monetary Value)
    st.write("### RFM Heatmap (Average Monetary Value by Recency and Frequency Scores)")
    heatmap_data = rfm.pivot_table(index='R_Score', columns='F_Score', values='Monetary', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_xlabel("Frequency Score")
    ax.set_ylabel("Recency Score")
    st.pyplot(fig)

    # Additional Feature 6: Scatter Plot of Recency vs. Frequency by Segment
    st.write("### Recency vs. Frequency by Segment")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(rfm['Recency'], rfm['Frequency'], c=rfm['RFM_Score'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, ax=ax, label="RFM Score")
    ax.set_xlabel("Recency")
    ax.set_ylabel("Frequency")
    ax.set_title("Scatter Plot of Recency vs. Frequency by RFM Score")
    st.pyplot(fig)
