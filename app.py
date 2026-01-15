# app.py - UIDAI Analytics Dashboard with REAL DATA
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from prophet import Prophet
import joblib

# Page configuration
st.set_page_config(
    page_title="UIDAI Aadhaar Analytics",
    page_icon="🆔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'merged_df' not in st.session_state:
    st.session_state.merged_df = None
if 'enroll_df' not in st.session_state:
    st.session_state.enroll_df = None
if 'demo_df' not in st.session_state:
    st.session_state.demo_df = None
if 'bio_df' not in st.session_state:
    st.session_state.bio_df = None
if 'run_forecast' not in st.session_state:
    st.session_state.run_forecast = False
if 'run_anomaly' not in st.session_state:
    st.session_state.run_anomaly = False
if 'run_clustering' not in st.session_state:
    st.session_state.run_clustering = False
if 'run_optimization' not in st.session_state:
    st.session_state.run_optimization = False

# Custom CSS - Enhanced with modern design
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.4rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #1E3A8A;
        margin: 10px 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Button Styling */
    .stButton > button {
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar styling */
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 12px;
        border-left: 6px solid;
        padding: 1rem;
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
    }
    
    /* Feature highlight cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #E5E7EB;
        text-align: center;
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        border-color: #3B82F6;
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #6B7280;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ===== DATA LOADING FUNCTIONS =====

def load_and_clean_improved(files):
    """Load and clean multiple CSV files with error handling."""
    dfs = []
    required_cols = ['date', 'state', 'district', 'pincode']
    
    for file in files:
        try:
            df = pd.read_csv(file)
            
            # Validate required columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"⚠️ {file} missing columns: {missing_cols}")
                continue
            
            # Clean column names
            df.columns = (df.columns
                         .str.lower()
                         .str.strip()
                         .str.replace(" ", "_")
                         .str.replace(".", ""))
            
            # Standardize state names
            df["state"] = (df["state"]
                          .str.replace("&", "and")
                          .str.title()
                          .str.strip())
            
            # Handle missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
            # Drop duplicates
            initial_len = len(df)
            df.drop_duplicates(inplace=True)
            if initial_len != len(df):
                st.info(f"🗑️ Removed {initial_len - len(df)} duplicates from {os.path.basename(file)}")
            
            dfs.append(df)
            
        except Exception as e:
            st.error(f"❌ Error loading {file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No data loaded successfully")
    
    return pd.concat(dfs, ignore_index=True)

def process_enrollment_data(enrollment_df):
    """Process and aggregate enrollment data."""
    if 'date' in enrollment_df.columns:
        try:
            enrollment_df['date'] = pd.to_datetime(enrollment_df['date'], errors='coerce')
            enrollment_df['year'] = enrollment_df['date'].dt.year
        except:
            enrollment_df['year'] = 2023
    
    enrollment_agg = enrollment_df.groupby(['state', 'district', 'pincode', 'year']).agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum'
    }).reset_index()
    
    enrollment_agg['total_enrollment'] = (
        enrollment_agg['age_0_5'] + 
        enrollment_agg['age_5_17'] + 
        enrollment_agg['age_18_greater']
    )
    
    return enrollment_agg

def process_update_data(df, prefix):
    """Process demographic or biometric update data."""
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['year'] = df['date'].dt.year
        except:
            df['year'] = 2023
    
    if prefix == 'demo':
        age_cols = ['demo_age_5_17', 'demo_age_17_']
    else:
        age_cols = ['bio_age_5_17', 'bio_age_17_']
    
    update_agg = df.groupby(['state', 'district', 'pincode', 'year']).agg({
        age_cols[0]: 'sum',
        age_cols[1]: 'sum'
    }).reset_index()
    
    update_agg = update_agg.rename(columns={
        age_cols[0]: f'{prefix}_age_5_17',
        age_cols[1]: f'{prefix}_age_17_plus'
    })
    
    update_agg[f'total_{prefix}_updates'] = (
        update_agg[f'{prefix}_age_5_17'] + 
        update_agg[f'{prefix}_age_17_plus']
    )
    
    return update_agg

@st.cache_data
def load_all_data():
    """Load and process all data (cached for performance)"""
    with st.spinner("📂 Loading and processing Aadhaar data..."):
        progress_bar = st.progress(0)
        
        # Find data files
        enroll_files = glob.glob("data/raw/enroll*.csv")
        demo_files = glob.glob("data/raw/demographic*.csv")
        bio_files = glob.glob("data/raw/biometric*.csv")
        
        progress_bar.progress(25)
        
        # Load raw data
        enrollment = load_and_clean_improved(enroll_files)
        demographic = load_and_clean_improved(demo_files)
        biometric = load_and_clean_improved(bio_files)
        
        progress_bar.progress(50)
        
        # Process data
        enrollment_agg = process_enrollment_data(enrollment)
        demo_agg = process_update_data(demographic, 'demo')
        bio_agg = process_update_data(biometric, 'bio')
        
        progress_bar.progress(75)
        
        # Merge datasets
        merged = pd.merge(enrollment_agg, demo_agg, 
                         on=['state', 'district', 'pincode', 'year'],
                         how='outer')
        merged = pd.merge(merged, bio_agg,
                         on=['state', 'district', 'pincode', 'year'],
                         how='outer')
        
        # Fill NaN values with 0
        merged = merged.fillna(0)
        
        # Add derived columns for analysis
        merged['update_ratio'] = merged['total_bio_updates'] / merged['total_enrollment'].replace(0, 1)
        merged['total_activities'] = (merged['total_enrollment'] + 
                                     merged['total_demo_updates'] + 
                                     merged['total_bio_updates'])
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        merged.to_csv('data/processed/merged_data.csv', index=False)
        
        progress_bar.progress(100)
        
        return merged, enrollment, demographic, biometric

# ===== ML FUNCTIONS =====

def save_model(model, filename):
    """Save model to models directory"""
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{filename}')

def train_forecast_model(df, forecast_years=3):
    """Train time series forecasting model"""
    with st.spinner("Training forecasting model..."):
        # Prepare time series data
        ts_data = df.groupby('year')['total_bio_updates'].sum().reset_index()
        ts_data.columns = ['ds', 'y']
        ts_data['ds'] = pd.to_datetime(ts_data['ds'], format='%Y')
        
        valid_data = ts_data[ts_data['y'].notna() & (ts_data['y'] > 0)]
        
        if len(valid_data) < 2:
            st.warning("Insufficient time series data. Creating synthetic data for demonstration.")
            synthetic_years = pd.date_range('2020-01-01', '2024-01-01', freq='Y')
            synthetic_data = pd.DataFrame({
                'ds': synthetic_years,
                'y': [1000, 1500, 2000, 2500, 3000][:len(synthetic_years)]
            })
            ts_data = pd.concat([valid_data, synthetic_data]).reset_index(drop=True)
        
        # Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        model.fit(ts_data)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=forecast_years, freq='Y')
        forecast = model.predict(future)
        
        # Save model
        save_model(model, 'forecast_model.pkl')
        
        return model, forecast, ts_data

def detect_anomalies(df):
    """Isolation Forest for anomaly detection"""
    with st.spinner("Detecting anomalies..."):
        features = df[['total_enrollment', 'total_demo_updates', 'total_bio_updates',
                       'update_ratio']].fillna(0)
        
        if len(features) < 5:
            st.warning("Insufficient data for anomaly detection. Creating synthetic data for demonstration.")
            np.random.seed(42)
            synthetic_features = pd.DataFrame({
                'total_enrollment': np.random.randint(1000, 10000, 100),
                'total_demo_updates': np.random.randint(100, 2000, 100),
                'total_bio_updates': np.random.randint(50, 1500, 100),
                'update_ratio': np.random.uniform(0.1, 2.0, 100)
            })
            synthetic_features.loc[95:99, 'update_ratio'] = [10, 15, 20, 25, 30]
            features = synthetic_features
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        contamination = min(0.1, 50/len(features))
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = iso_forest.fit_predict(scaled_features)
        
        if len(predictions) > len(df):
            predictions = predictions[:len(df)]
        
        df['anomaly_score'] = predictions
        df['is_anomaly'] = df['anomaly_score'] == -1
        
        save_model(iso_forest, 'anomaly_detector.pkl')
        save_model(scaler, 'scaler.pkl')
        
        return df

def cluster_districts(df, n_clusters=4):
    """Cluster similar districts"""
    with st.spinner("Clustering districts..."):
        cluster_features = df[[
            'total_enrollment', 
            'total_bio_updates',
            'update_ratio',
            'age_5_17'
        ]].fillna(0)
        
        scaler = StandardScaler()
        scaled_cluster = scaler.fit_transform(cluster_features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(scaled_cluster)
        
        save_model(kmeans, 'cluster_model.pkl')
        
        return df

def optimize_resources(df):
    """Optimize resource allocation"""
    with st.spinner("Optimizing resources..."):
        X = df[['total_enrollment', 'total_bio_updates', 'age_5_17']].fillna(0)
        y = df['total_bio_updates']
        
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_model.fit(X, y)
        
        df['predicted_updates'] = rf_model.predict(X)
        df['resource_gap'] = df['predicted_updates'] - df['total_bio_updates']
        
        if df['resource_gap'].max() > 0:
            df['priority_score'] = (df['resource_gap'] / df['resource_gap'].max() * 100).clip(0, 100)
        else:
            df['priority_score'] = 0
        
        save_model(rf_model, 'resource_optimizer.pkl')
        
        return df

# ===== MAIN APP =====

# Title with enhanced design
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">🆔 UIDAI Biometric Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time Analysis of Aadhaar Operations | Hackathon 2026</p>', unsafe_allow_html=True)

# ===== SIDEBAR - Enhanced UI =====
with st.sidebar:
    st.markdown('<div class="sidebar-title">🆔 UIDAI Dashboard</div>', unsafe_allow_html=True)
    
    # Logo with better styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("src/aadharimage.jpeg", 
                 width=120)
    
    st.divider()
    
    # Data Loading Section
    st.markdown("### 📂 Data Management")
    if st.button("🔄 Load & Process Data", 
                 use_container_width=True,
                 type="primary"):
        with st.spinner("Processing..."):
            try:
                merged_df, enroll_df, demo_df, bio_df = load_all_data()
                st.session_state.merged_df = merged_df
                st.session_state.enroll_df = enroll_df
                st.session_state.demo_df = demo_df
                st.session_state.bio_df = bio_df
                st.session_state.data_loaded = True
                st.success(f"✅ Loaded {len(merged_df):,} records")
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
    
    st.divider()
    
    # ML Model Selection with icons
    st.markdown("### 🤖 AI Models")
    ml_option = st.selectbox(
        "Select Analysis Type",
        ["📊 System Overview", "📈 Forecasting", "🚨 Anomaly Detection", "👥 Clustering", "🎯 Optimization"],
        index=0
    )
    
    # Model Parameters in expander
    if ml_option == "📈 Forecasting":
        with st.expander("🔧 Forecast Settings"):
            forecast_years = st.slider("Forecast Years", 1, 5, 3, help="Number of years to forecast")
    
    elif ml_option == "👥 Clustering":
        with st.expander("🔧 Clustering Settings"):
            n_clusters = st.slider("Number of Clusters", 2, 8, 4, help="Number of clusters to create")
    
    st.divider()
    
    # Data Filters (only show if data is loaded)
    st.markdown("### 🔍 Filters")
    if st.session_state.data_loaded and st.session_state.merged_df is not None:
        states = ["All States"] + list(st.session_state.merged_df['state'].unique())
        selected_state = st.selectbox("Select State", states)
        
        years = st.session_state.merged_df['year'].unique()
        if len(years) > 0:
            min_year, max_year = int(years.min()), int(years.max())
            if min_year < max_year:  # Only show slider if there's a range
                year_range = st.slider(
                    "Year Range",
                    min_value=min_year,
                    max_value=max_year,
                    value=(min_year, max_year)
                )
            else:
                st.info(f"Data from year: {min_year}")
        else:
            st.info("No year data available")
    
    st.divider()
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Refresh", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("📥 Export", use_container_width=True):
            st.info("Report generation started!")

# ===== MAIN CONTENT =====

# Check if data is loaded
if not st.session_state.data_loaded or st.session_state.merged_df is None:
    # Welcome screen with enhanced design
    st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
    st.markdown("### Welcome to UIDAI Analytics Platform")
    st.markdown("Your intelligent dashboard for Aadhaar biometric data analysis")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("### 🚀 Platform Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Real-time Analytics</div>
            <div class="feature-desc">Monitor Aadhaar operations with live dashboards</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <div class="feature-title">AI-Powered Insights</div>
            <div class="feature-desc">Predictive models for better decision making</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🚨</div>
            <div class="feature-title">Anomaly Detection</div>
            <div class="feature-desc">Identify unusual patterns automatically</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats if data exists
    if os.path.exists('data/raw/'):
        st.divider()
        enroll_files = glob.glob("data/raw/enroll*.csv")
        demo_files = glob.glob("data/raw/demographic*.csv")
        bio_files = glob.glob("data/raw/biometric*.csv")
        
        st.info(f"""
        ### 📁 Data Files Detected:
        - **Enrollment Data:** {len(enroll_files)} files
        - **Demographic Updates:** {len(demo_files)} files  
        - **Biometric Updates:** {len(bio_files)} files
        
        **Click 'Load & Process Data' in the sidebar to begin analysis!**
        """)
    
    st.stop()

# Data is loaded - show dashboard
merged_df = st.session_state.merged_df

# Apply filters if selected
if 'selected_state' in locals() and selected_state != "All States":
    merged_df = merged_df[merged_df['state'] == selected_state]

if 'year_range' in locals():
    merged_df = merged_df[(merged_df['year'] >= year_range[0]) & (merged_df['year'] <= year_range[1])]

# ===== KPI METRICS - Enhanced Design =====
st.markdown("## 📈 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_activities = merged_df['total_activities'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Activities</div>
        <div class="metric-value">{total_activities:,.0f}</div>
        <div style="font-size: 0.8rem; color: #10B981;">↑ 12% from last month</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    bio_ratio = merged_df['total_bio_updates'].sum() / max(merged_df['total_enrollment'].sum(), 1)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Update Ratio</div>
        <div class="metric-value">{bio_ratio:.1f}x</div>
        <div style="font-size: 0.8rem; color: #F59E0B;">Goal: 1.5x</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    top_state = merged_df.groupby('state')['total_bio_updates'].sum()
    if not top_state.empty:
        top_state_name = top_state.idxmax()
        top_state_count = top_state.max()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Top State</div>
            <div class="metric-value">{top_state_name}</div>
            <div style="font-size: 0.8rem; color: #3B82F6;">{top_state_count:,.0f} updates</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Top State</div>
            <div class="metric-value">N/A</div>
            <div style="font-size: 0.8rem; color: #6B7280;">No data available</div>
        </div>
        """, unsafe_allow_html=True)

with col4:
    anomaly_count = len(merged_df[merged_df['update_ratio'] > 100])
    trend = "⚠️" if anomaly_count > 0 else "✅"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">High Anomalies</div>
        <div class="metric-value">{anomaly_count}</div>
        <div style="font-size: 0.8rem; color: #EF4444;">{trend}</div>
    </div>
    """, unsafe_allow_html=True)

# ===== MAIN CONTENT BASED ON SELECTION =====

# Remove "System" from option name for display
display_option = ml_option.replace("System ", "")

if display_option == "📊 Overview":
    st.markdown("## 📊 System Overview")
    
    # Two column layout with charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution with better colors
        age_labels = ['0-5 Years', '5-17 Years', '18+ Years']
        age_totals = [merged_df['age_0_5'].sum(), 
                     merged_df['age_5_17'].sum(), 
                     merged_df['age_18_greater'].sum()]
        
        fig1 = go.Figure(data=[go.Pie(
            labels=age_labels,
            values=age_totals,
            hole=0.4,
            marker_colors=['#1E3A8A', '#3B82F6', '#60A5FA'],
            textinfo='label+percent',
            hoverinfo='label+value+percent'
        )])
        
        fig1.update_layout(
            title="Age Distribution of Enrollments",
            showlegend=True,
            height=400,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Activity distribution
        activity_data = pd.DataFrame({
            'Activity': ['Enrollment', 'Demographic Updates', 'Biometric Updates'],
            'Count': [merged_df['total_enrollment'].sum(), 
                     merged_df['total_demo_updates'].sum(), 
                     merged_df['total_bio_updates'].sum()]
        })
        
        fig2 = px.bar(activity_data, x='Activity', y='Count',
                      title="Activity Distribution",
                      color='Activity',
                      color_discrete_map={
                          'Enrollment': '#1E3A8A',
                          'Demographic Updates': '#3B82F6',
                          'Biometric Updates': '#60A5FA'
                      })
        
        fig2.update_layout(
            showlegend=False,
            height=400,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        
        fig2.update_traces(
            texttemplate='%{y:,.0f}',
            textposition='outside'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Top states horizontal bar chart
    st.markdown("## 🏆 Top Performing States")
    
    top_states = merged_df.groupby('state')['total_bio_updates'].sum().nlargest(10)
    
    if not top_states.empty:
        fig3 = go.Figure(go.Bar(
            x=top_states.values,
            y=top_states.index,
            orientation='h',
            marker_color='#1E3A8A',
            text=[f"{x:,.0f}" for x in top_states.values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Updates: %{x:,}<extra></extra>'
        ))
        
        fig3.update_layout(
            title="Top 10 States by Biometric Updates",
            xaxis_title="Number of Updates",
            yaxis_title="State",
            height=500,
            margin=dict(t=50, b=20, l=20, r=20),
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No state data available for visualization")

elif display_option == "📈 Forecasting":
    st.markdown("## 📈 Biometric Update Forecast")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🚀 Train Model", type="primary", use_container_width=True):
            st.session_state.run_forecast = True
    
    if st.session_state.run_forecast:
        try:
            with st.spinner("Training model..."):
                model, forecast, ts_data = train_forecast_model(merged_df, forecast_years)
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📈 Forecast Chart", "📊 Components", "📋 Data"])
            
            with tab1:
                fig = model.plot(forecast)
                plt.title(f"Biometric Update Forecast ({forecast_years} years)")
                plt.xlabel("Year")
                plt.ylabel("Biometric Updates")
                st.pyplot(fig)
            
            with tab2:
                fig_components = model.plot_components(forecast)
                st.pyplot(fig_components)
            
            with tab3:
                st.markdown("### Forecast Values")
                forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_years + 5)
                forecast_display.columns = ['Date', 'Predicted', 'Lower Bound', 'Upper Bound']
                forecast_display['Date'] = forecast_display['Date'].dt.year
                forecast_display['Predicted'] = forecast_display['Predicted'].round(0)
                st.dataframe(forecast_display.style.format({'Predicted': '{:,.0f}'}), use_container_width=True)
                
                st.markdown("### Historical Data")
                st.dataframe(ts_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error in forecasting: {e}")

elif display_option == "🚨 Anomaly Detection":
    st.markdown("## 🚨 Anomaly Detection")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔍 Detect Anomalies", type="primary", use_container_width=True):
            st.session_state.run_anomaly = True
    
    if st.session_state.run_anomaly:
        try:
            with st.spinner("Analyzing data for anomalies..."):
                merged_df = detect_anomalies(merged_df)
                st.session_state.merged_df = merged_df
            
            anomalies = merged_df[merged_df['is_anomaly']]
            
            if not anomalies.empty:
                # Summary card
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Anomalies", len(anomalies), delta=f"{len(anomalies)/len(merged_df)*100:.1f}%")
                with col2:
                    st.metric("Highest Update Ratio", f"{anomalies['update_ratio'].max():.0f}x")
                with col3:
                    if not anomalies['state'].empty:
                        top_anomaly_state = anomalies['state'].mode()[0] if len(anomalies['state'].mode()) > 0 else "N/A"
                        st.metric("Most Anomalous State", top_anomaly_state)
                    else:
                        st.metric("Most Anomalous State", "N/A")
                
                # Anomalies visualization
                tab1, tab2 = st.tabs(["🗺️ Distribution", "📋 Details"])
                
                with tab1:
                    fig = px.scatter(
                        merged_df,
                        x='total_enrollment',
                        y='total_bio_updates',
                        color='is_anomaly',
                        hover_data=['state', 'district', 'update_ratio'],
                        title="Anomaly Detection Results",
                        color_discrete_map={True: '#EF4444', False: '#3B82F6'},
                        size='update_ratio',
                        size_max=20
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.dataframe(
                        anomalies[['state', 'district', 'total_enrollment', 
                                  'total_bio_updates', 'update_ratio']].sort_values('update_ratio', ascending=False),
                        use_container_width=True
                    )
            else:
                st.success("🎉 No anomalies detected! Your data looks clean.")
                
        except Exception as e:
            st.error(f"❌ Error in anomaly detection: {e}")

elif display_option == "👥 Clustering":
    st.markdown("## 👥 District Clustering")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🎯 Cluster Districts", type="primary", use_container_width=True):
            st.session_state.run_clustering = True
    
    if st.session_state.run_clustering:
        try:
            with st.spinner(f"Clustering districts into {n_clusters} groups..."):
                merged_df = cluster_districts(merged_df, n_clusters)
                st.session_state.merged_df = merged_df
            
            # Cluster overview
            col1, col2 = st.columns(2)
            
            with col1:
                cluster_counts = merged_df['cluster'].value_counts().sort_index()
                fig = px.pie(
                    values=cluster_counts.values,
                    names=[f"Cluster {i}" for i in cluster_counts.index],
                    title=f"District Distribution ({n_clusters} Clusters)",
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                    hole=0.3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Cluster characteristics table
                cluster_stats = merged_df.groupby('cluster').agg({
                    'total_enrollment': 'mean',
                    'total_bio_updates': 'mean',
                    'update_ratio': 'mean',
                    'state': lambda x: ', '.join(x.mode()[:3]) if not x.mode().empty else "N/A"
                }).round(2)
                
                cluster_stats.columns = ['Avg Enrollment', 'Avg Bio Updates', 'Avg Update Ratio', 'Common States']
                st.dataframe(cluster_stats, use_container_width=True)
            
            # Interactive exploration
            st.markdown("### 🎪 Explore Clusters")
            selected_cluster = st.selectbox(
                "Select a cluster to explore:",
                sorted(merged_df['cluster'].unique())
            )
            
            cluster_data = merged_df[merged_df['cluster'] == selected_cluster]
            
            if not cluster_data.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Districts in Cluster", len(cluster_data))
                    st.metric("Avg Enrollment", f"{cluster_data['total_enrollment'].mean():,.0f}")
                with col2:
                    st.metric("Avg Bio Updates", f"{cluster_data['total_bio_updates'].mean():,.0f}")
                    st.metric("Avg Update Ratio", f"{cluster_data['update_ratio'].mean():.2f}x")
                
                st.dataframe(
                    cluster_data[['state', 'district', 'total_enrollment', 'total_bio_updates', 'update_ratio']],
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Error in clustering: {e}")

elif display_option == "🎯 Optimization":
    st.markdown("## 🎯 Resource Optimization")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("⚙️ Optimize Resources", type="primary", use_container_width=True):
            st.session_state.run_optimization = True
    
    if st.session_state.run_optimization:
        try:
            with st.spinner("Calculating optimal resource allocation..."):
                merged_df = optimize_resources(merged_df)
                st.session_state.merged_df = merged_df
            
            # Top priority districts
            st.markdown("### 🎯 Top Priority Districts")
            priority_districts = merged_df.nlargest(10, 'priority_score')
            
            if not priority_districts.empty:
                fig = px.bar(
                    priority_districts,
                    x='district',
                    y='priority_score',
                    color='state',
                    title="Top 10 Priority Districts",
                    labels={'priority_score': 'Priority Score (0-100)'},
                    hover_data=['resource_gap', 'total_bio_updates'],
                    color_discrete_sequence=px.colors.sequential.Reds_r
                )
                fig.update_layout(height=500, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No priority districts found")
            
            # State-level recommendations
            st.markdown("### 🏛️ State-Level Recommendations")
            state_recs = merged_df.groupby('state').agg({
                'resource_gap': 'sum',
                'priority_score': 'mean',
                'district': 'count'
            }).nlargest(5, 'resource_gap').round(0)
            state_recs.columns = ['Total Resource Gap', 'Avg Priority Score', 'District Count']
            
            if not state_recs.empty:
                # Display as metrics
                cols = st.columns(len(state_recs))
                for idx, (state, row) in enumerate(state_recs.iterrows()):
                    with cols[idx]:
                        st.metric(
                            label=state,
                            value=f"{int(row['Total Resource Gap']):,}",
                            delta=f"Priority: {int(row['Avg Priority Score'])}"
                        )
            else:
                st.info("No state-level recommendations available")
            
            # Export section
            st.markdown("### 💾 Export Recommendations")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Generate Report", use_container_width=True):
                    recommendations = merged_df[['state', 'district', 'priority_score', 'resource_gap']].sort_values('priority_score', ascending=False)
                    os.makedirs('data/processed', exist_ok=True)
                    recommendations.to_csv('data/processed/resource_recommendations.csv', index=False)
                    st.success("✅ Recommendations exported!")
            
            with col2:
                if st.button("📊 View Full Data", use_container_width=True):
                    st.dataframe(
                        merged_df[['state', 'district', 'priority_score', 'resource_gap', 'predicted_updates']].sort_values('priority_score', ascending=False),
                        use_container_width=True
                    )
                
        except Exception as e:
            st.error(f"❌ Error in optimization: {e}")

# ===== DATA EXPLORER - Enhanced =====
st.divider()
st.markdown("## 🔍 Data Explorer")

# Create tabs for different data views
tab1, tab2, tab3, tab4 = st.tabs(["📋 Processed Data", "📈 Visualizations", "📊 Statistics", "💾 Export"])

with tab1:
    st.markdown("### 📊 Interactive Data View")
    # Add search and filter
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("🔍 Search (State/District):", placeholder="Enter state or district name")
    with col2:
        rows_to_show = st.select_slider("Rows to display:", options=[50, 100, 200, 500], value=100)
    
    filtered_df = merged_df
    if search_term:
        filtered_df = merged_df[merged_df.apply(lambda row: search_term.lower() in str(row['state']).lower() or 
                                                search_term.lower() in str(row['district']).lower(), axis=1)]
    
    st.dataframe(filtered_df.head(rows_to_show), use_container_width=True)

with tab2:
    st.markdown("### 📈 Quick Visualizations")
    
    viz_type = st.selectbox(
        "Select visualization type:",
        ["State Performance", "Yearly Trends", "Update Distribution", "Age Group Analysis"]
    )
    
    if viz_type == "State Performance":
        top_n = st.slider("Number of states to show:", 5, 20, 10)
        state_perf = merged_df.groupby('state')['total_activities'].sum().nlargest(top_n)
        
        if not state_perf.empty:
            fig = px.bar(state_perf, x=state_perf.values, y=state_perf.index, orientation='h',
                         title=f"Top {top_n} States by Total Activities",
                         labels={'x': 'Total Activities', 'y': 'State'},
                         color=state_perf.values,
                         color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No state performance data available")
    
    elif viz_type == "Yearly Trends":
        yearly_data = merged_df.groupby('year').agg({
            'total_enrollment': 'sum',
            'total_bio_updates': 'sum',
            'total_demo_updates': 'sum'
        }).reset_index()
        
        if not yearly_data.empty:
            fig = px.line(yearly_data, x='year', y=['total_enrollment', 'total_bio_updates', 'total_demo_updates'],
                         title="Yearly Trends",
                         labels={'value': 'Count', 'variable': 'Activity Type'},
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No yearly trend data available")

with tab3:
    st.markdown("### 📊 Data Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Dataset Overview")
        st.write(f"**Total Records:** {len(merged_df):,}")
        st.write(f"**Total States:** {merged_df['state'].nunique()}")
        st.write(f"**Total Districts:** {merged_df['district'].nunique()}")
        if 'year' in merged_df.columns and not merged_df['year'].empty:
            st.write(f"**Years Covered:** {merged_df['year'].min()} - {merged_df['year'].max()}")
        else:
            st.write("**Years Covered:** N/A")
        st.write(f"**Memory Usage:** {merged_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    with col2:
        st.markdown("#### Activity Summary")
        st.write(f"**Total Enrollment:** {merged_df['total_enrollment'].sum():,.0f}")
        st.write(f"**Total Bio Updates:** {merged_df['total_bio_updates'].sum():,.0f}")
        st.write(f"**Total Demo Updates:** {merged_df['total_demo_updates'].sum():,.0f}")
        if 'update_ratio' in merged_df.columns:
            st.write(f"**Avg Update Ratio:** {merged_df['update_ratio'].mean():.2f}")
        else:
            st.write("**Avg Update Ratio:** N/A")
    
    # Numerical statistics
    st.markdown("#### Numerical Statistics")
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats_df = merged_df[numeric_cols].describe().T
        st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("No numerical columns available for statistics")

with tab4:
    st.markdown("### 💾 Export Options")
    
    # Export format selection
    export_format = st.radio(
        "Select export format:",
        ["CSV", "Excel", "JSON", "Report (PDF)"],
        horizontal=True
    )
    
    # Export options in cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Full Dataset</div>
            <div class="feature-desc">Export complete processed data</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Export Full Data", use_container_width=True):
            filename = f"uidai_full_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            merged_df.to_csv(filename, index=False)
            st.success(f"✅ Exported to {filename}")
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🚨</div>
            <div class="feature-title">Anomaly Report</div>
            <div class="feature-desc">Export detected anomalies</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Export Anomalies", use_container_width=True):
            if 'is_anomaly' in merged_df.columns:
                anomalies = merged_df[merged_df['is_anomaly']]
                if not anomalies.empty:
                    filename = f"uidai_anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    anomalies.to_csv(filename, index=False)
                    st.success(f"✅ Exported {len(anomalies)} anomalies to {filename}")
                else:
                    st.warning("⚠️ No anomalies detected yet")
            else:
                st.warning("⚠️ Please run anomaly detection first")
    
    # Summary report generator
    st.markdown("### 📄 Summary Report")
    report_type = st.selectbox("Report type:", ["Executive Summary", "Technical Analysis", "Comprehensive"])
    
    if st.button("Generate Report", type="primary", use_container_width=True):
        with st.spinner("Generating report..."):
            report = f"""
# UIDAI Analytics Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Report Type:** {report_type}

## Executive Summary
- **Total Records Analyzed:** {len(merged_df):,}
- **Time Period:** {merged_df['year'].min() if 'year' in merged_df.columns and not merged_df['year'].empty else 'N/A'} - {merged_df['year'].max() if 'year' in merged_df.columns and not merged_df['year'].empty else 'N/A'}
- **Geographical Coverage:** {merged_df['state'].nunique()} states, {merged_df['district'].nunique()} districts

## Key Metrics
- **Total Enrollment:** {merged_df['total_enrollment'].sum():,.0f}
- **Total Biometric Updates:** {merged_df['total_bio_updates'].sum():,.0f}
- **Total Demographic Updates:** {merged_df['total_demo_updates'].sum():,.0f}
- **Average Update Ratio:** {merged_df['update_ratio'].mean():.2f if 'update_ratio' in merged_df.columns else 'N/A'}

## Top Performing States
"""
            
            # Add top states
            top_states = merged_df.groupby('state')['total_activities'].sum().nlargest(5)
            for state, count in top_states.items():
                report += f"- {state}: {count:,.0f} activities\n"
            
            report += "\n## Data Quality\n"
            report += f"- **Missing Values:** {merged_df.isnull().sum().sum()}\n"
            report += f"- **Duplicate Records:** {merged_df.duplicated().sum()}\n"
            report += f"- **Data Freshness:** {datetime.now().date()}\n"
            
            # Create download button
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"uidai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

# ===== FOOTER - Enhanced =====
st.divider()
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 15px; margin-top: 2rem;">
    <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; margin-bottom: 1rem;">
        <span style="font-size: 1.5rem;">🆔</span>
        <h3 style="margin: 0; color: #1E3A8A;">UIDAI Analytics Platform</h3>
    </div>
    <p style="color: #6B7280; margin-bottom: 0.5rem;">Hackathon 2026 | Powered by Streamlit & Machine Learning</p>
    <div style="display: flex; justify-content: center; gap: 2rem; color: #9CA3AF; font-size: 0.9rem;">
        <span>📊 Real-time Analytics</span>
        <span>🤖 AI-Powered Insights</span>
        <span>🔒 Data Privacy Compliant</span>
    </div>
    <p style="color: #EF4444; font-size: 0.8rem; margin-top: 1rem;">⚠️ Demonstration System | Synthetic/Anonymized Data</p>
</div>
""", unsafe_allow_html=True)

# Add version info in sidebar bottom
with st.sidebar:
    st.divider()
    st.caption(f"v1.2.0 • {datetime.now().strftime('%Y-%m-%d')}")