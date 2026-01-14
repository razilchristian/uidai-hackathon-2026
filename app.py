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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        margin: 10px 0;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ===== DATA LOADING FUNCTIONS (From your notebook) =====

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
                st.warning(f"Warning: {file} missing columns: {missing_cols}")
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
                st.info(f"Removed {initial_len - len(df)} duplicates from {os.path.basename(file)}")
            
            dfs.append(df)
            
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No data loaded successfully")
    
    return pd.concat(dfs, ignore_index=True)

def process_enrollment_data(enrollment_df):
    """Process and aggregate enrollment data."""
    # Extract year from date
    if 'date' in enrollment_df.columns:
        try:
            enrollment_df['date'] = pd.to_datetime(enrollment_df['date'], errors='coerce')
            enrollment_df['year'] = enrollment_df['date'].dt.year
        except:
            enrollment_df['year'] = 2023
    
    # Group by state, district, pincode, and year
    enrollment_agg = enrollment_df.groupby(['state', 'district', 'pincode', 'year']).agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum'
    }).reset_index()
    
    # Calculate total enrollment
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
    
    # Determine age columns based on prefix
    if prefix == 'demo':
        age_cols = ['demo_age_5_17', 'demo_age_17_']
    else:  # biometric
        age_cols = ['bio_age_5_17', 'bio_age_17_']
    
    # Group and aggregate
    update_agg = df.groupby(['state', 'district', 'pincode', 'year']).agg({
        age_cols[0]: 'sum',
        age_cols[1]: 'sum'
    }).reset_index()
    
    # Rename columns
    update_agg = update_agg.rename(columns={
        age_cols[0]: f'{prefix}_age_5_17',
        age_cols[1]: f'{prefix}_age_17_plus'
    })
    
    # Calculate total updates
    update_agg[f'total_{prefix}_updates'] = (
        update_agg[f'{prefix}_age_5_17'] + 
        update_agg[f'{prefix}_age_17_plus']
    )
    
    return update_agg

@st.cache_data
def load_all_data():
    """Load and process all data (cached for performance)"""
    with st.spinner("📂 Loading and processing Aadhaar data..."):
        # Find data files
        enroll_files = glob.glob("data/raw/enroll*.csv")
        demo_files = glob.glob("data/raw/demographic*.csv")
        bio_files = glob.glob("data/raw/biometric*.csv")
        
        # Load raw data
        enrollment = load_and_clean_improved(enroll_files)
        demographic = load_and_clean_improved(demo_files)
        biometric = load_and_clean_improved(bio_files)
        
        # Process data
        enrollment_agg = process_enrollment_data(enrollment)
        demo_agg = process_update_data(demographic, 'demo')
        bio_agg = process_update_data(biometric, 'bio')
        
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
        
        return merged, enrollment, demographic, biometric

# ===== ML FUNCTIONS =====

def save_model(model, filename):
    """Save model to models directory with proper directory creation"""
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{filename}')

def train_forecast_model(df):
    """Train time series forecasting model"""
    with st.spinner("Training forecasting model..."):
        # Prepare time series data
        ts_data = df.groupby('year')['total_bio_updates'].sum().reset_index()
        ts_data.columns = ['ds', 'y']
        ts_data['ds'] = pd.to_datetime(ts_data['ds'], format='%Y')
        
        # Check if we have enough data
        valid_data = ts_data[ts_data['y'].notna() & (ts_data['y'] > 0)]
        
        if len(valid_data) < 2:
            # Create synthetic data for demonstration
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
        future = model.make_future_dataframe(periods=3, freq='Y')
        forecast = model.predict(future)
        
        # Save model
        save_model(model, 'forecast_model.pkl')
        
        return model, forecast, ts_data

def detect_anomalies(df):
    """Isolation Forest for anomaly detection"""
    with st.spinner("Detecting anomalies..."):
        # Features for anomaly detection
        features = df[['total_enrollment', 'total_demo_updates', 'total_bio_updates',
                       'update_ratio']].fillna(0)
        
        # Check if we have enough data
        if len(features) < 5:
            st.warning("Insufficient data for anomaly detection. Creating synthetic data for demonstration.")
            # Create synthetic data for demonstration
            np.random.seed(42)
            synthetic_features = pd.DataFrame({
                'total_enrollment': np.random.randint(1000, 10000, 100),
                'total_demo_updates': np.random.randint(100, 2000, 100),
                'total_bio_updates': np.random.randint(50, 1500, 100),
                'update_ratio': np.random.uniform(0.1, 2.0, 100)
            })
            # Add some outliers
            synthetic_features.loc[95:99, 'update_ratio'] = [10, 15, 20, 25, 30]
            features = synthetic_features
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Train Isolation Forest
        contamination = min(0.1, 50/len(features))  # Adaptive contamination
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = iso_forest.fit_predict(scaled_features)
        
        # Add predictions to dataframe (only for original rows if we had synthetic)
        if len(predictions) > len(df):
            predictions = predictions[:len(df)]
        
        df['anomaly_score'] = predictions
        df['is_anomaly'] = df['anomaly_score'] == -1
        
        # Save models
        save_model(iso_forest, 'anomaly_detector.pkl')
        save_model(scaler, 'scaler.pkl')
        
        return df

def cluster_districts(df, n_clusters=4):
    """Cluster similar districts"""
    with st.spinner("Clustering districts..."):
        # Features for clustering
        cluster_features = df[[
            'total_enrollment', 
            'total_bio_updates',
            'update_ratio',
            'age_5_17'
        ]].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        scaled_cluster = scaler.fit_transform(cluster_features)
        
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(scaled_cluster)
        
        # Save model
        save_model(kmeans, 'cluster_model.pkl')
        
        return df

def optimize_resources(df):
    """Optimize resource allocation"""
    with st.spinner("Optimizing resources..."):
        # Simple optimization: Predict needed resources per district
        X = df[['total_enrollment', 'total_bio_updates', 'age_5_17']].fillna(0)
        y = df['total_bio_updates']
        
        # Train Random Forest for prediction
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_model.fit(X, y)
        
        # Predict needed resources
        df['predicted_updates'] = rf_model.predict(X)
        df['resource_gap'] = df['predicted_updates'] - df['total_bio_updates']
        
        # Priority score (0-100)
        if df['resource_gap'].max() > 0:
            df['priority_score'] = (df['resource_gap'] / df['resource_gap'].max() * 100).clip(0, 100)
        else:
            df['priority_score'] = 0
        
        # Save model
        save_model(rf_model, 'resource_optimizer.pkl')
        
        return df

# ===== MAIN APP =====

# Title
st.markdown('<h1 class="main-header">🆔 UIDAI Biometric Analytics Platform</h1>', unsafe_allow_html=True)
st.markdown("### Real-time Analysis of Aadhaar Operations | Hackathon 2026")

# ===== SIDEBAR =====
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/7/7e/Aadhaar_Logo.svg/1200px-Aadhaar_Logo.svg.png", 
             width=150)
    st.title("🔧 Controls")
    
    # Data Loading Section
    st.subheader("📂 Data Management")
    if st.button("🔄 Load & Process Data", use_container_width=True):
        st.session_state.data_loaded = False
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
                st.error(f"Error loading data: {e}")
    
    st.divider()
    
    # ML Model Selection
    st.subheader("🤖 AI Models")
    ml_option = st.radio(
        "Select Model",
        ["📊 Overview", "📈 Forecasting", "🚨 Anomaly Detection", "👥 Clustering", "🎯 Optimization"],
        index=0
    )
    
    # Model Parameters
    if ml_option == "📈 Forecasting":
        forecast_years = st.slider("Forecast Years", 1, 5, 3)
    
    elif ml_option == "👥 Clustering":
        n_clusters = st.slider("Number of Clusters", 2, 8, 4)
    
    st.divider()
    
    # Data Filters
    st.subheader("🔍 Filters")
    if 'merged_df' in st.session_state:
        states = ["All"] + list(st.session_state.merged_df['state'].unique())
        selected_state = st.selectbox("State", states)
    
    st.divider()
    
    # Actions
    if st.button("📥 Export Report", use_container_width=True):
        st.info("Report generation started!")

# ===== MAIN CONTENT =====

# Check if data is loaded
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if not st.session_state.data_loaded:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ## Welcome to UIDAI Analytics Platform
        
        **Dataset Information:**
        - Enrollment Data: enroll1.csv, enroll2.csv, ...
        - Demographic Updates: demographic1.csv, demographic2.csv, ...
        - Biometric Updates: biometric1.csv, biometric2.csv, ...
        
        **Click 'Load & Process Data' in the sidebar to begin!**
        """)
        
        # Quick stats if data exists
        if os.path.exists('data/raw/'):
            enroll_files = glob.glob("data/raw/enroll*.csv")
            demo_files = glob.glob("data/raw/demographic*.csv")
            bio_files = glob.glob("data/raw/biometric*.csv")
            
            st.info(f"""
            **Files Detected:**
            - Enrollment: {len(enroll_files)} files
            - Demographic: {len(demo_files)} files  
            - Biometric: {len(bio_files)} files
            """)
    
    st.stop()

# Data is loaded - show dashboard
merged_df = st.session_state.merged_df

# ===== KPI METRICS =====
st.markdown("## 📈 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_activities = merged_df['total_activities'].sum()
    st.metric("Total Activities", f"{total_activities:,.0f}")

with col2:
    bio_ratio = merged_df['total_bio_updates'].sum() / max(merged_df['total_enrollment'].sum(), 1)
    st.metric("Update/Enrollment Ratio", f"{bio_ratio:.1f}x")

with col3:
    top_state = merged_df.groupby('state')['total_bio_updates'].sum().idxmax()
    top_state_count = merged_df.groupby('state')['total_bio_updates'].sum().max()
    st.metric("Top State", top_state, f"{top_state_count:,.0f}")

with col4:
    anomaly_count = len(merged_df[merged_df['update_ratio'] > 100])
    st.metric("High Anomalies", f"{anomaly_count}", 
              delta=None, delta_color="off")

# ===== MAIN CONTENT BASED ON SELECTION =====

if ml_option == "📊 Overview":
    st.markdown("## 📊 System Overview")
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution
        age_totals = [merged_df['age_0_5'].sum(), 
                     merged_df['age_5_17'].sum(), 
                     merged_df['age_18_greater'].sum()]
        
        fig1 = px.pie(
            values=age_totals,
            names=['0-5 Years', '5-17 Years', '18+ Years'],
            title="Age Distribution of Enrollments",
            color_discrete_sequence=px.colors.sequential.Blues_r
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
                      color_discrete_sequence=['#1E3A8A', '#3B82F6', '#60A5FA'])
        st.plotly_chart(fig2, use_container_width=True)
    
    # Top states chart
    st.markdown("## 🏆 Top Performing States")
    top_states = merged_df.groupby('state')['total_bio_updates'].sum().nlargest(10)
    
    fig3 = go.Figure(go.Bar(
        x=top_states.values,
        y=top_states.index,
        orientation='h',
        marker_color='#1E3A8A',
        text=[f"{x:,.0f}" for x in top_states.values],
        textposition='outside'
    ))
    
    fig3.update_layout(
        title="Top 10 States by Biometric Updates",
        xaxis_title="Number of Updates",
        yaxis_title="State",
        height=400
    )
    st.plotly_chart(fig3, use_container_width=True)

elif ml_option == "📈 Forecasting":
    st.markdown("## 📈 Biometric Update Forecast")
    
    if st.button("Train Forecasting Model", type="primary"):
        try:
            model, forecast, ts_data = train_forecast_model(merged_df)
            
            # Plot forecast
            col1, col2 = st.columns(2)
            
            with col1:
                fig = model.plot(forecast)
                st.pyplot(fig)
            
            with col2:
                # Components plot
                fig_components = model.plot_components(forecast)
                st.pyplot(fig_components)
            
            # Show forecast table
            st.markdown("### Forecast Values")
            forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
            forecast_display.columns = ['Date', 'Predicted', 'Lower Bound', 'Upper Bound']
            forecast_display['Date'] = forecast_display['Date'].dt.year
            st.dataframe(forecast_display, use_container_width=True)
            
            # Show historical data
            st.markdown("### Historical Data")
            st.dataframe(ts_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in forecasting: {e}")

elif ml_option == "🚨 Anomaly Detection":
    st.markdown("## 🚨 Anomaly Detection")
    
    if st.button("Detect Anomalies", type="primary"):
        try:
            merged_df = detect_anomalies(merged_df)
            st.session_state.merged_df = merged_df
            
            anomalies = merged_df[merged_df['is_anomaly']]
            
            if not anomalies.empty:
                st.warning(f"⚠️ Detected {len(anomalies)} anomalies")
                
                # Show anomalies table
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(
                        anomalies[['state', 'district', 'total_enrollment', 
                                  'total_bio_updates', 'update_ratio']].head(20),
                        use_container_width=True
                    )
                
                with col2:
                    # Anomaly distribution by state
                    anomaly_by_state = anomalies['state'].value_counts().head(10)
                    fig = px.bar(
                        x=anomaly_by_state.index,
                        y=anomaly_by_state.values,
                        title="Anomalies by State",
                        labels={'x': 'State', 'y': 'Number of Anomalies'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot
                fig = px.scatter(
                    merged_df,
                    x='total_enrollment',
                    y='total_bio_updates',
                    color='is_anomaly',
                    hover_data=['state', 'district', 'update_ratio'],
                    title="Anomaly Detection Results",
                    color_discrete_map={True: 'red', False: 'blue'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No anomalies detected!")
        except Exception as e:
            st.error(f"Error in anomaly detection: {e}")

elif ml_option == "👥 Clustering":
    st.markdown("## 👥 District Clustering")
    
    if st.button("Cluster Districts", type="primary"):
        try:
            merged_df = cluster_districts(merged_df, n_clusters)
            st.session_state.merged_df = merged_df
            
            # Show cluster distribution
            cluster_counts = merged_df['cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=[f"Cluster {i}" for i in cluster_counts.index],
                    y=cluster_counts.values,
                    title=f"District Distribution Across {n_clusters} Clusters",
                    labels={'x': 'Cluster', 'y': 'Number of Districts'},
                    color=[f"Cluster {i}" for i in cluster_counts.index]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Pie chart of cluster distribution
                fig_pie = px.pie(
                    values=cluster_counts.values,
                    names=[f"Cluster {i}" for i in cluster_counts.index],
                    title="Cluster Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Show cluster characteristics
            st.markdown("### Cluster Characteristics")
            cluster_stats = merged_df.groupby('cluster').agg({
                'total_enrollment': 'mean',
                'total_bio_updates': 'mean',
                'update_ratio': 'mean',
                'state': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Mixed'
            }).round(2)
            
            st.dataframe(cluster_stats, use_container_width=True)
            
            # Show sample districts from each cluster
            st.markdown("### Sample Districts from Each Cluster")
            for cluster_id in sorted(merged_df['cluster'].unique()):
                with st.expander(f"Cluster {cluster_id}"):
                    sample_districts = merged_df[merged_df['cluster'] == cluster_id][['state', 'district', 'total_enrollment', 'total_bio_updates']].head(5)
                    st.dataframe(sample_districts, use_container_width=True)
        except Exception as e:
            st.error(f"Error in clustering: {e}")

elif ml_option == "🎯 Optimization":
    st.markdown("## 🎯 Resource Optimization")
    
    if st.button("Optimize Resources", type="primary"):
        try:
            merged_df = optimize_resources(merged_df)
            st.session_state.merged_df = merged_df
            
            # Show priority districts
            priority_districts = merged_df.nlargest(10, 'priority_score')
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    priority_districts,
                    x='district',
                    y='priority_score',
                    color='state',
                    title="Top 10 Priority Districts for Resource Allocation",
                    labels={'priority_score': 'Priority Score (0-100)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Resource gap visualization
                fig_gap = px.scatter(
                    merged_df.nlargest(20, 'resource_gap'),
                    x='district',
                    y='resource_gap',
                    size='priority_score',
                    color='state',
                    title="Resource Gap by District",
                    labels={'resource_gap': 'Resource Gap'}
                )
                st.plotly_chart(fig_gap, use_container_width=True)
            
            # Show state-level recommendations
            st.markdown("### State-Level Recommendations")
            state_recs = merged_df.groupby('state').agg({
                'resource_gap': 'sum',
                'priority_score': 'mean',
                'district': 'count'
            }).nlargest(5, 'resource_gap').round(0)
            state_recs.columns = ['Total Resource Gap', 'Avg Priority Score', 'Number of Districts']
            
            st.dataframe(state_recs, use_container_width=True)
            
            # Export recommendations
            if st.button("📋 Export Recommendations"):
                recommendations = merged_df[['state', 'district', 'priority_score', 'resource_gap']].sort_values('priority_score', ascending=False)
                recommendations.to_csv('data/processed/resource_recommendations.csv', index=False)
                st.success("✅ Recommendations exported to resource_recommendations.csv")
        except Exception as e:
            st.error(f"Error in optimization: {e}")

# ===== DATA EXPLORER =====
st.divider()
st.markdown("## 🔍 Data Explorer")

# Show raw data preview
tab1, tab2, tab3 = st.tabs(["📋 Processed Data", "📊 Raw Stats", "💾 Export"])

with tab1:
    st.dataframe(merged_df.head(100), use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dataset Shape:**", merged_df.shape)
        st.write("**States:**", merged_df['state'].nunique())
        st.write("**Districts:**", merged_df['district'].nunique())
        st.write("**Years Covered:**", f"{merged_df['year'].min()} - {merged_df['year'].max()}")
    with col2:
        st.write("**Total Enrollment:**", f"{merged_df['total_enrollment'].sum():,.0f}")
        st.write("**Total Bio Updates:**", f"{merged_df['total_bio_updates'].sum():,.0f}")
        st.write("**Total Demo Updates:**", f"{merged_df['total_demo_updates'].sum():,.0f}")
        st.write("**Avg Update Ratio:**", f"{merged_df['update_ratio'].mean():.1f}")

with tab3:
    st.markdown("### Export Options")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export to CSV", use_container_width=True):
            merged_df.to_csv('data/processed/uidai_analysis_export.csv', index=False)
            st.success("✅ Exported to uidai_analysis_export.csv")
    
    with col2:
        if st.button("Export Anomalies", use_container_width=True):
            if 'is_anomaly' in merged_df.columns:
                anomalies = merged_df[merged_df['is_anomaly']]
                if not anomalies.empty:
                    anomalies.to_csv('data/processed/anomalies.csv', index=False)
                    st.success(f"✅ Exported {len(anomalies)} anomalies to anomalies.csv")
                else:
                    st.warning("No anomalies detected yet")
            else:
                st.warning("Please run anomaly detection first")
    
    st.markdown("### Quick Reports")
    if st.button("Generate Summary Report"):
        report = f"""
        # UIDAI Analytics Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        ## Summary Statistics:
        - Total Records: {len(merged_df):,}
        - Total States: {merged_df['state'].nunique()}
        - Total Districts: {merged_df['district'].nunique()}
        - Total Enrollment: {merged_df['total_enrollment'].sum():,.0f}
        - Total Biometric Updates: {merged_df['total_bio_updates'].sum():,.0f}
        - Average Update Ratio: {merged_df['update_ratio'].mean():.2f}
        
        ## Top 5 States by Activity:
        {merged_df.groupby('state')['total_activities'].sum().nlargest(5).to_string()}
        """
        
        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name="uidai_report.txt",
            mime="text/plain"
        )

# ===== FOOTER =====
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 20px;">
    <p>🆔 UIDAI Analytics Platform | Hackathon 2026 | Powered by Streamlit</p>
    <p>⚠️ This is a demonstration system. All data is synthetic or anonymized.</p>
</div>
""", unsafe_allow_html=True)