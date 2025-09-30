"""
Telecom Revenue Optimization Dashboard

Interactive Streamlit dashboard for exploring customer insights, 
churn predictions, and revenue optimization opportunities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Dashboard configuration
st.set_page_config(
    page_title="Telecom Revenue Optimization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the telecom customer dataset."""
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'master_dataset.csv')
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please run the data generation notebook first.")
        return None

def calculate_business_metrics(df):
    """Calculate key business metrics."""
    if df is None:
        return {}
    
    metrics = {
        'total_customers': len(df),
        'total_revenue': df['arpu'].sum(),
        'avg_arpu': df['arpu'].mean(),
        'churn_rate': df['churned'].mean(),
        'avg_satisfaction': df['satisfaction_score'].mean(),
        'revenue_at_risk': df[df['churned'] == 1]['arpu'].sum(),
        'high_value_customers': (df['arpu'] > 75).sum(),
        'campaign_responsive': (df['total_conversions'] > 0).sum()
    }
    return metrics

def create_revenue_dashboard(df):
    """Create revenue analysis dashboard."""
    st.header("💰 Revenue Analysis")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = df['arpu'].sum()
        st.metric("Total Monthly Revenue", f"${total_revenue:,.2f}")
    
    with col2:
        avg_arpu = df['arpu'].mean()
        st.metric("Average ARPU", f"${avg_arpu:.2f}")
    
    with col3:
        high_value = (df['arpu'] > 75).sum()
        st.metric("High-Value Customers", f"{high_value:,}")
    
    with col4:
        revenue_at_risk = df[df['churned'] == 1]['arpu'].sum()
        st.metric("Revenue at Risk", f"${revenue_at_risk:,.2f}")
    
    # Revenue visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # ARPU distribution
        fig_arpu = px.histogram(df, x='arpu', nbins=50, title='ARPU Distribution')
        fig_arpu.add_vline(x=avg_arpu, line_dash="dash", line_color="red", 
                          annotation_text=f"Mean: ${avg_arpu:.2f}")
        st.plotly_chart(fig_arpu, use_container_width=True)
    
    with col2:
        # Revenue by plan type
        plan_revenue = df.groupby('plan_type').agg({
            'arpu': 'mean',
            'customer_id': 'count'
        }).reset_index()
        plan_revenue.columns = ['Plan Type', 'Avg ARPU', 'Customer Count']
        
        fig_plan = px.bar(plan_revenue, x='Plan Type', y='Avg ARPU', 
                         title='Average ARPU by Plan Type')
        st.plotly_chart(fig_plan, use_container_width=True)
    
    # Geographic revenue analysis
    st.subheader("Geographic Revenue Distribution")
    city_revenue = df.groupby('city').agg({
        'arpu': ['mean', 'sum'],
        'customer_id': 'count'
    }).round(2)
    city_revenue.columns = ['Avg ARPU', 'Total Revenue', 'Customer Count']
    city_revenue = city_revenue.sort_values('Total Revenue', ascending=False)
    
    fig_geo = px.bar(city_revenue.head(10), x=city_revenue.head(10).index, 
                     y='Total Revenue', title='Top 10 Cities by Total Revenue')
    st.plotly_chart(fig_geo, use_container_width=True)

def create_churn_dashboard(df):
    """Create churn analysis dashboard."""
    st.header("🔄 Churn Analysis")
    
    # Churn metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        churn_rate = df['churned'].mean()
        st.metric("Overall Churn Rate", f"{churn_rate:.2%}")
    
    with col2:
        churned_customers = df['churned'].sum()
        st.metric("Churned Customers", f"{churned_customers:,}")
    
    with col3:
        churned_customers = df[df['churned'] == 1]
        if len(churned_customers) > 0:
            avg_churn_arpu = churned_customers['arpu'].mean()
            st.metric("Avg ARPU (Churned)", f"${avg_churn_arpu:.2f}")
        else:
            st.metric("Avg ARPU (Churned)", "N/A")
    
    with col4:
        high_risk_churn = ((df['arpu'] > 75) & (df['churned'] == 1)).sum()
        st.metric("High-Value Churned", f"{high_risk_churn:,}")
    
    # Churn analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn by satisfaction score - handle empty bins
        try:
            satisfaction_bins = pd.cut(df['satisfaction_score'], bins=5)
            churn_by_satisfaction = df.groupby(satisfaction_bins)['churned'].mean()
            
            fig_satisfaction = px.bar(
                x=[str(x) for x in churn_by_satisfaction.index],
                y=churn_by_satisfaction.values * 100,
                title='Churn Rate by Satisfaction Score Range',
                labels={'x': 'Satisfaction Score Range', 'y': 'Churn Rate (%)'}
            )
            st.plotly_chart(fig_satisfaction, use_container_width=True)
        except ValueError as e:
            st.warning("Unable to create satisfaction score chart: " + str(e))
    
    with col2:
        # Churn by tenure - handle empty bins
        try:
            tenure_bins = pd.cut(df['tenure_months'], bins=5)
            churn_by_tenure = df.groupby(tenure_bins)['churned'].mean()
            
            fig_tenure = px.bar(
                x=[str(x) for x in churn_by_tenure.index],
                y=churn_by_tenure.values * 100,
                title='Churn Rate by Tenure Range',
                labels={'x': 'Tenure Range (months)', 'y': 'Churn Rate (%)'}
            )
            st.plotly_chart(fig_tenure, use_container_width=True)
        except ValueError as e:
            st.warning("Unable to create tenure chart: " + str(e))
    
    # Risk segmentation
    st.subheader("Customer Risk Segmentation")
    
    # Create risk scores safely
    satisfaction_component = (10 - df['satisfaction_score']) * 0.4
    complaints_component = df['num_complaints_12m'] * 0.3
    late_payments_component = df['late_payments_12m'] * 0.3
    
    df['risk_score'] = satisfaction_component + complaints_component + late_payments_component
    
    # Risk categories
    df['risk_category'] = pd.cut(df['risk_score'], 
                                bins=[0, 2, 4, 6, float('inf')],
                                labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'])
    
    risk_analysis = df.groupby('risk_category').agg({
        'customer_id': 'count',
        'churned': 'mean',
        'arpu': 'mean'
    }).round(3)
    risk_analysis.columns = ['Customer Count', 'Churn Rate', 'Avg ARPU']
    
    st.dataframe(risk_analysis)

def create_campaign_dashboard(df):
    """Create campaign effectiveness dashboard."""
    st.header("📢 Campaign Effectiveness")
    
    # Campaign metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_campaigns = df['campaigns_exposed'].mean()
        st.metric("Avg Campaigns/Customer", f"{avg_campaigns:.1f}")
    
    with col2:
        total_impressions = df['total_impressions'].sum()
        total_clicks = df['total_clicks'].sum()
        if total_impressions > 0:
            overall_ctr = total_clicks / total_impressions
            st.metric("Overall CTR", f"{overall_ctr:.2%}")
        else:
            st.metric("Overall CTR", "N/A")
    
    with col3:
        total_clicks = df['total_clicks'].sum()
        total_conversions = df['total_conversions'].sum()
        if total_clicks > 0:
            conversion_rate = total_conversions / total_clicks
            st.metric("Conversion Rate", f"{conversion_rate:.2%}")
        else:
            st.metric("Conversion Rate", "N/A")
    
    with col4:
        responsive_customers = (df['total_conversions'] > 0).sum()
        st.metric("Responsive Customers", f"{responsive_customers:,}")
    
    # Campaign performance by channel
    st.subheader("Channel Performance Analysis")
    
    channel_performance = df.groupby('primary_channel').agg({
        'total_impressions': 'sum',
        'total_clicks': 'sum',
        'total_conversions': 'sum'
    }).reset_index()
    
    # Safely calculate CTR and Conversion Rate
    channel_performance['CTR'] = np.where(
        channel_performance['total_impressions'] > 0,
        channel_performance['total_clicks'] / channel_performance['total_impressions'],
        0
    )
    channel_performance['Conversion_Rate'] = np.where(
        channel_performance['total_clicks'] > 0,
        channel_performance['total_conversions'] / channel_performance['total_clicks'],
        0
    )
    
    # Remove channels with no data
    channel_performance = channel_performance[channel_performance['total_impressions'] > 0].copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_ctr = px.bar(channel_performance.sort_values('CTR', ascending=False),
                        x='primary_channel', y='CTR',
                        title='Click-Through Rate by Channel')
        fig_ctr.update_layout(yaxis_tickformat='.2%')
        st.plotly_chart(fig_ctr, use_container_width=True)
    
    with col2:
        fig_conv = px.bar(channel_performance.sort_values('Conversion_Rate', ascending=False),
                         x='primary_channel', y='Conversion_Rate',
                         title='Conversion Rate by Channel')
        fig_conv.update_layout(yaxis_tickformat='.2%')
        st.plotly_chart(fig_conv, use_container_width=True)

def create_customer_insights(df):
    """Create customer insights dashboard."""
    st.header("👥 Customer Insights")
    
    # Customer segmentation
    st.subheader("Customer Segmentation")
    
    # Value-based segmentation
    def assign_value_segment(row):
        if row['arpu'] >= 80 and row['satisfaction_score'] >= 8:
            return 'Champions'
        elif row['arpu'] >= 80 and row['satisfaction_score'] >= 6:
            return 'Loyal Customers'
        elif row['arpu'] >= 60 and row['satisfaction_score'] >= 7:
            return 'Potential Loyalists'
        elif row['arpu'] >= 40 and row['satisfaction_score'] >= 6:
            return 'New Customers'
        elif row['arpu'] >= 40 and row['satisfaction_score'] < 6:
            return 'At Risk'
        else:
            return 'Need Attention'
    
    df['customer_segment'] = df.apply(assign_value_segment, axis=1)
    
    # Segment distribution
    segment_counts = df['customer_segment'].value_counts()
    fig_segments = px.pie(values=segment_counts.values, names=segment_counts.index,
                         title='Customer Segment Distribution')
    st.plotly_chart(fig_segments, use_container_width=True)
    
    # Segment analysis table
    segment_analysis = df.groupby('customer_segment').agg({
        'customer_id': 'count',
        'arpu': 'mean',
        'churned': 'mean',
        'satisfaction_score': 'mean',
        'total_conversions': 'mean'
    }).round(2)
    segment_analysis.columns = ['Count', 'Avg ARPU', 'Churn Rate', 'Avg Satisfaction', 'Avg Conversions']
    
    st.subheader("Segment Performance Metrics")
    st.dataframe(segment_analysis)
    
    # Usage patterns
    st.subheader("Usage Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data usage by age group
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], 
                                labels=['18-30', '31-45', '46-60', '60+'])
        age_usage = df.groupby('age_group')['monthly_data_gb'].mean()
        
        fig_age_usage = px.bar(x=age_usage.index, y=age_usage.values,
                              title='Average Data Usage by Age Group',
                              labels={'x': 'Age Group', 'y': 'Data Usage (GB)'})
        st.plotly_chart(fig_age_usage, use_container_width=True)
    
    with col2:
        # Digital engagement
        df['total_sessions'] = df['monthly_web_sessions'] + df['monthly_app_sessions']
        engagement_income = df.groupby(pd.qcut(df['annual_income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']))['total_sessions'].mean()
        
        fig_engagement = px.bar(x=engagement_income.index, y=engagement_income.values,
                               title='Digital Engagement by Income Quartile',
                               labels={'x': 'Income Quartile', 'y': 'Monthly Sessions'})
        st.plotly_chart(fig_engagement, use_container_width=True)

def create_predictive_insights(df):
    """Create predictive insights dashboard."""
    st.header("🔮 Predictive Insights")
    
    st.info("This section shows simulated predictions. In production, this would use trained ML models.")
    
    # Simulate churn predictions
    np.random.seed(42)
    df['churn_probability'] = np.random.beta(2, 5, len(df))  # Simulated probabilities
    
    # High-risk customers
    high_risk_threshold = 0.7
    high_risk_customers = df[df['churn_probability'] > high_risk_threshold]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High-Risk Customers", f"{len(high_risk_customers):,}")
    
    with col2:
        risk_revenue = high_risk_customers['arpu'].sum()
        st.metric("Revenue at Risk", f"${risk_revenue:,.2f}")
    
    with col3:
        if len(high_risk_customers) > 0:
            avg_risk_arpu = high_risk_customers['arpu'].mean()
            st.metric("Avg ARPU (High-Risk)", f"${avg_risk_arpu:.2f}")
    
    # Churn probability distribution
    fig_churn_dist = px.histogram(df, x='churn_probability', nbins=50,
                                 title='Churn Probability Distribution')
    fig_churn_dist.add_vline(x=high_risk_threshold, line_dash="dash", line_color="red",
                            annotation_text="High Risk Threshold")
    st.plotly_chart(fig_churn_dist, use_container_width=True)
    
    # Top at-risk customers
    st.subheader("Top 10 At-Risk High-Value Customers")
    at_risk_high_value = df[(df['churn_probability'] > 0.5) & (df['arpu'] > 60)].nlargest(10, 'churn_probability')
    
    display_cols = ['customer_id', 'arpu', 'satisfaction_score', 'tenure_months', 'churn_probability']
    st.dataframe(at_risk_high_value[display_cols])

def main():
    """Main dashboard application."""
    st.title("📊 Telecom Revenue Optimization Dashboard")
    st.markdown("**Powered by Dentsu's Data-Driven Analytics**")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a dashboard:", [
        "📊 Overview",
        "💰 Revenue Analysis", 
        "🔄 Churn Analysis",
        "📢 Campaign Effectiveness",
        "👥 Customer Insights",
        "🔮 Predictive Insights"
    ])
    
    # Overview page
    if page == "📊 Overview":
        st.header("Executive Overview")
        
        # Key metrics
        metrics = calculate_business_metrics(df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{metrics['total_customers']:,}")
        
        with col2:
            st.metric("Monthly Revenue", f"${metrics['total_revenue']:,.2f}")
        
        with col3:
            st.metric("Churn Rate", f"{metrics['churn_rate']:.2%}")
        
        with col4:
            st.metric("Avg Satisfaction", f"{metrics['avg_satisfaction']:.1f}/10")
        
        # Quick insights
        st.subheader("Key Business Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"💰 **Revenue Opportunity**: ${metrics['revenue_at_risk']:,.2f} monthly revenue at risk from churn")
            st.info(f"🎯 **High-Value Segment**: {metrics['high_value_customers']:,} customers with ARPU > $75")
        
        with col2:
            st.warning(f"📢 **Campaign Potential**: {metrics['campaign_responsive']:,} customers responsive to campaigns")
            responsive_rate = metrics['campaign_responsive'] / metrics['total_customers']
            st.info(f"📈 **Engagement Rate**: {responsive_rate:.1%} of customers engage with campaigns")
    
    elif page == "💰 Revenue Analysis":
        create_revenue_dashboard(df)
    
    elif page == "🔄 Churn Analysis":
        create_churn_dashboard(df)
    
    elif page == "📢 Campaign Effectiveness":
        create_campaign_dashboard(df)
    
    elif page == "👥 Customer Insights":
        create_customer_insights(df)
    
    elif page == "🔮 Predictive Insights":
        create_predictive_insights(df)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About this Dashboard**")
    st.sidebar.markdown("Built for Dentsu's telecom revenue optimization strategy using privacy-first synthetic data and advanced analytics.")

if __name__ == "__main__":
    main()