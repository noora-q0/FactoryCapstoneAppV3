import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import requests
import joblib  # Make sure this is imported to load joblib files
import os
import requests
import copy




# Define cached functions
@st.cache  # Cache for downloading data
def download_file(url, dest_path):
    response = requests.get(url)
    with open(dest_path, 'wb') as f:
        f.write(response.content)
    return dest_path


@st.cache(allow_output_mutation=True)   # Cache for loading models
def load_model(path):
    return joblib.load(path)


@st.cache  # Cache for loading data
def load_data(path):
    return pd.read_csv(path)

# URLs for S3 files
file_urls = {
    "cleaned_data.csv": "https://factory-capstone-models.s3.us-east-1.amazonaws.com/cleaned_data.csv",
    "scaler.joblib": "https://factory-capstone-models.s3.us-east-1.amazonaws.com/scaler.joblib",
    "tuned_rf_model.joblib": "https://factory-capstone-models.s3.us-east-1.amazonaws.com/tuned_rf_model.joblib"
}

# Download and cache files only once
data_path = download_file(file_urls["cleaned_data.csv"], "cleaned_data.csv")
scaler_path = download_file(file_urls["scaler.joblib"], "scaler.joblib")
model_path = download_file(file_urls["tuned_rf_model.joblib"], "tuned_rf_model.joblib")

# Load data and models using cached paths
factory_df = load_data(data_path).copy()  # Create a copy to avoid mutating the cached version
scaler = load_model(scaler_path)
# Load the model directly without deepcopy, since we're allowing mutations in the cache
model = load_model(model_path)





# Extract columns for Stage 1 output measurements (both Actual and Setpoint)
stage1_actuals = [col for col in factory_df.columns if "Stage1.Output" in col and col.endswith(".Actual")]
stage1_setpoints = [col for col in factory_df.columns if "Stage1.Output" in col and col.endswith(".Setpoint")]



# Ensure error columns are calculated and accessible globally
if 'error_columns' not in globals():
    error_columns = {}
    for actual in stage1_actuals:
        setpoint = actual.replace("Actual", "Setpoint")
        if setpoint in factory_df.columns:
            error_col = f"{actual}_Error"
            factory_df[error_col] = factory_df[actual] - factory_df[setpoint]
            error_columns[actual] = error_col


# Filter out non-numeric columns for numerical analysis
numeric_df = factory_df.select_dtypes(include=[np.number])


# Sidebar navigation
st.sidebar.title("Capstone Project Navigation")
section = st.sidebar.radio("Select a Section:", 
                           ("Dataset Overview Dashboard", "Trend Analysis (Actual vs. Setpoint)", 
                            "EDA", "Clustering", "Predictions", "Feature Importance Analysis",
                            "Error Threshold Analysis", "Error Distribution Analysis", "Anomaly Detection",
		            "Cumulative Error Tracking", "Correlation Analysis"))





# Define input features
input_features = [
    'AmbientConditions.AmbientHumidity.U.Actual', 'AmbientConditions.AmbientTemperature.U.Actual',
    'Machine1.RawMaterial.Property1', 'Machine1.RawMaterial.Property2', 'Machine1.RawMaterial.Property3', 'Machine1.RawMaterial.Property4',
    'Machine1.RawMaterialFeederParameter.U.Actual', 'Machine1.Zone1Temperature.C.Actual', 'Machine1.Zone2Temperature.C.Actual',
    'Machine1.MotorAmperage.U.Actual', 'Machine1.MotorRPM.C.Actual', 'Machine1.MaterialPressure.U.Actual', 'Machine1.MaterialTemperature.U.Actual',
    'Machine1.ExitZoneTemperature.C.Actual', 'Machine2.RawMaterial.Property1', 'Machine2.RawMaterial.Property2',
    'Machine2.RawMaterial.Property3', 'Machine2.RawMaterial.Property4', 'Machine2.RawMaterialFeederParameter.U.Actual',
    'Machine2.Zone1Temperature.C.Actual', 'Machine2.Zone2Temperature.C.Actual', 'Machine2.MotorAmperage.U.Actual',
    'Machine2.MotorRPM.C.Actual', 'Machine2.MaterialPressure.U.Actual', 'Machine2.MaterialTemperature.U.Actual',
    'Machine2.ExitZoneTemperature.C.Actual', 'Machine3.RawMaterial.Property1', 'Machine3.RawMaterial.Property2',
    'Machine3.RawMaterial.Property3', 'Machine3.RawMaterial.Property4', 'Machine3.RawMaterialFeederParameter.U.Actual',
    'Machine3.Zone1Temperature.C.Actual', 'Machine3.Zone2Temperature.C.Actual', 'Machine3.MotorAmperage.U.Actual',
    'Machine3.MotorRPM.C.Actual', 'Machine3.MaterialPressure.U.Actual', 'Machine3.MaterialTemperature.U.Actual',
    'Machine3.ExitZoneTemperature.C.Actual', 'FirstStage.CombinerOperation.Temperature1.U.Actual',
    'FirstStage.CombinerOperation.Temperature2.U.Actual', 'FirstStage.CombinerOperation.Temperature3.C.Actual'
]



# Target Measurements
target_measurements = [
    'Stage1.Output.Measurement0.U.Actual', 'Stage1.Output.Measurement1.U.Actual', 
    'Stage1.Output.Measurement2.U.Actual', 'Stage1.Output.Measurement3.U.Actual', 
    'Stage1.Output.Measurement4.U.Actual', 'Stage1.Output.Measurement5.U.Actual',
    'Stage1.Output.Measurement6.U.Actual', 'Stage1.Output.Measurement7.U.Actual', 
    'Stage1.Output.Measurement8.U.Actual', 'Stage1.Output.Measurement9.U.Actual',
    'Stage1.Output.Measurement10.U.Actual', 'Stage1.Output.Measurement11.U.Actual',
    'Stage1.Output.Measurement12.U.Actual', 'Stage1.Output.Measurement13.U.Actual',
    'Stage1.Output.Measurement14.U.Actual'
]




# 1. Overview Section
if section == "Dataset Overview Dashboard":
    st.title("Dataset Overview")
    st.write("This section provides an overview of the dataset.")
    
    # Display the first few rows of the dataset
    st.subheader("Dataset Preview")
    st.write(factory_df.head())

    # Display summary statistics
    st.subheader("Summary Statistics")
    st.write(factory_df.describe())
    
    # Create tabs for different visualizations
    tabs = st.tabs(["Data Distribution", "Box Plots", "Correlation Heatmap", "Time Series Trends", "Data Quality Indicators"])

    # Tab 1: Data Distribution Visuals
    


    # Multi-select widget for selecting variables
    st.sidebar.subheader("Select Features to Visualize")
    selected_features = st.sidebar.multiselect(
        "Choose features to plot:",
        options=input_features,
        default=input_features[:5]  # Set an initial default selection
    )

    # If no feature is selected, display a warning
    if not selected_features:
        st.warning("Please select at least one feature to visualize.")
    else:
        # Filter the numeric DataFrame based on selected features
        selected_df = factory_df[selected_features]

        # Tab 1: Data Distribution Visuals
        with tabs[0]:
            st.header("Data Distribution")
            for col in selected_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=selected_df[col], nbinsx=30))
                fig.update_layout(
                    title=f"Distribution of {col}",
                    xaxis_title=col,
                    yaxis_title="Frequency",
                    height=300
                )
                st.plotly_chart(fig)

        # Tab 2: Box Plots for Key Features
        with tabs[1]:
            st.header("Box Plots")
            for col in selected_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Box(y=selected_df[col], name=col))
                fig.update_layout(
                    title=f"Box Plot for {col}",
                    yaxis_title=col,
                    height=300
                )
                st.plotly_chart(fig)




        # Tab 3: Correlation Heatmap
        with tabs[2]:
            st.header("Correlation Heatmap")

            if len(selected_features) < 3:
                st.write("Please select at least three features to display a meaningful correlation heatmap.")
            else:
                correlation = selected_df.corr()
                fig = go.Figure(data=go.Heatmap(
                    z=correlation.values,
                    x=correlation.columns,
                    y=correlation.columns,
                    colorscale="Viridis"
                ))
                fig.update_layout(
                    title="Correlation Heatmap",
                    height=600
                )
                st.plotly_chart(fig)






        # Tab 4: Time Series Trends
        with tabs[3]:
            st.header("Time Series Trends")
            for col in selected_features:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=factory_df['time_stamp'], y=factory_df[col], mode='lines', name=col))
                fig.update_layout(
                    title=f"Trend of {col} Over Time",
                    xaxis_title="Time",
                    yaxis_title=col,
                    height=400
                )
                st.plotly_chart(fig)




    # Tab 5: Data Quality Indicators
    with tabs[4]:
        st.header("Data Quality Indicators")
        # Show missing values percentage
        missing_values = factory_df.isnull().sum() / len(factory_df) * 100
        st.bar_chart(missing_values[missing_values > 0])  # Show only features with missing data
        st.write("Outliers and anomalies can be highlighted here based on thresholds.")
        # Further analysis for outliers can be added here



# 2. Trend Analysis (Actual vs. Setpoint) Section
elif section == "Trend Analysis (Actual vs. Setpoint)":
    st.title("Trend Analysis: Actual vs. Setpoint")

    # Section Introduction
    st.write("""
        This section allows you to compare actual values with their setpoints for each measurement. 
        Select a specific measurement or view all measurements to analyze how closely actual outputs follow setpoints.
        
        - **Blue line**: Represents actual measurement values.
        - **Red dashed line**: Represents setpoint values, providing a benchmark for comparison.
    """)

    # Extract columns for Stage 1 output measurements
    stage1_actuals = [col for col in factory_df.columns if "Stage1.Output" in col and col.endswith(".Actual")]
    stage1_setpoints = [col for col in factory_df.columns if "Stage1.Output" in col and col.endswith(".Setpoint")]

    # Sidebar: Select measurement
    options = ["All Measurements"] + stage1_actuals
    selected_measurement_actual = st.sidebar.selectbox("Choose a Stage 1 output measurement (Actual) to display:", options)

    # Convert time_stamp to string for compatibility with Plotly
    factory_df['time_stamp'] = factory_df['time_stamp'].astype(str)

    # Check if "All Measurements" is selected
    if selected_measurement_actual == "All Measurements":
        # Status message for viewing all measurements
        st.info("Displaying trend analysis for all measurements.")

        # Define grid layout for all measurements
        rows, cols = 5, 3  # Adjust rows and cols as needed
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=stage1_actuals)

        # Plot each measurement in the grid
        for i, actual in enumerate(stage1_actuals):
            row, col = i // cols + 1, i % cols + 1
            setpoint = actual.replace("Actual", "Setpoint")

            fig.add_trace(go.Scatter(x=factory_df['time_stamp'], y=factory_df[actual],
                                     mode='lines', name=f'{actual} (Actual)', line=dict(color='blue')), row=row, col=col)
            if setpoint in factory_df.columns:
                fig.add_trace(go.Scatter(x=factory_df['time_stamp'], y=factory_df[setpoint],
                                         mode='lines', name=f'{setpoint} (Setpoint)', line=dict(color='red', dash='dash')), row=row, col=col)

        fig.update_layout(title='All Measurements: Actual vs. Setpoint Over Time', showlegend=False, height=1500, width=1000)
        st.plotly_chart(fig)
    else:
        # Status message for viewing a single measurement
        st.info(f"Displaying trend analysis for {selected_measurement_actual}.")

        # Plot the selected measurement and its setpoint
        setpoint = selected_measurement_actual.replace("Actual", "Setpoint")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=factory_df['time_stamp'], y=factory_df[selected_measurement_actual],
                                 mode='lines', name=f'{selected_measurement_actual} (Actual)', line=dict(color='blue')))
        if setpoint in factory_df.columns:
            fig.add_trace(go.Scatter(x=factory_df['time_stamp'], y=factory_df[setpoint],
                                     mode='lines', name=f'{setpoint} (Setpoint)', line=dict(color='red', dash='dash')))
        fig.update_layout(title=f'{selected_measurement_actual.split(".")[-2]} - Actual vs. Setpoint Over Time',
                          xaxis_title="Time", yaxis_title="Measurement Value", height=600, width=1000)
        st.plotly_chart(fig)


# 3. EDA Section
elif section == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    st.write("This section provides additional insights into the dataset.")
    
    # Select and display a histogram for any measurement
    measurement = st.selectbox("Choose a measurement to analyze:", stage1_actuals)
    fig = px.histogram(factory_df, x=measurement, nbins=30, title=f"Distribution of {measurement}")
    st.plotly_chart(fig)





# 4. Clustering Section
elif section == "Clustering":
    st.title("Clustering Analysis")

    st.write("""
        This section performs K-Means clustering on the selected features to group similar instances. 
        Clustering helps identify patterns and operational states, allowing for better monitoring and analysis.
    """)

    # Define selected features for clustering
    selected_features = [
        'AmbientConditions.AmbientHumidity.U.Actual', 'AmbientConditions.AmbientTemperature.U.Actual',
        'Machine1.RawMaterial.Property1', 'Machine1.RawMaterial.Property2', 'Machine1.RawMaterial.Property3', 'Machine1.RawMaterial.Property4',
        'Machine1.RawMaterialFeederParameter.U.Actual', 'Machine1.Zone1Temperature.C.Actual', 'Machine1.Zone2Temperature.C.Actual',
        'Machine1.MotorAmperage.U.Actual', 'Machine1.MotorRPM.C.Actual', 'Machine1.MaterialPressure.U.Actual', 'Machine1.MaterialTemperature.U.Actual',
        'Machine1.ExitZoneTemperature.C.Actual', 'Machine2.RawMaterial.Property1', 'Machine2.RawMaterial.Property2',
        'Machine2.RawMaterial.Property3', 'Machine2.RawMaterial.Property4', 'Machine2.RawMaterialFeederParameter.U.Actual',
        'Machine2.Zone1Temperature.C.Actual', 'Machine2.Zone2Temperature.C.Actual', 'Machine2.MotorAmperage.U.Actual',
        'Machine2.MotorRPM.C.Actual', 'Machine2.MaterialPressure.U.Actual', 'Machine2.MaterialTemperature.U.Actual',
        'Machine2.ExitZoneTemperature.C.Actual', 'Machine3.RawMaterial.Property1', 'Machine3.RawMaterial.Property2',
        'Machine3.RawMaterial.Property3', 'Machine3.RawMaterial.Property4', 'Machine3.RawMaterialFeederParameter.U.Actual',
        'Machine3.Zone1Temperature.C.Actual', 'Machine3.Zone2Temperature.C.Actual', 'Machine3.MotorAmperage.U.Actual',
        'Machine3.MotorRPM.C.Actual', 'Machine3.MaterialPressure.U.Actual', 'Machine3.MaterialTemperature.U.Actual',
        'Machine3.ExitZoneTemperature.C.Actual', 'FirstStage.CombinerOperation.Temperature1.U.Actual',
        'FirstStage.CombinerOperation.Temperature2.U.Actual', 'FirstStage.CombinerOperation.Temperature3.C.Actual'
    ]

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(factory_df[selected_features])

    # Elbow Method for Optimal k
    st.subheader("Optimal Number of Clusters (k) - Elbow Method")
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features_scaled)
        inertia.append(kmeans.inertia_)
    
    # Plot Elbow Method
    fig, ax = plt.subplots()
    ax.plot(k_range, inertia, 'bo-')
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for Optimal k")
    st.pyplot(fig)

    # Select optimal k based on elbow (default k=3 for demo)
    optimal_k = st.slider("Select the number of clusters:", 2, 10, 3)
    
    # Apply K-Means with the selected optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    factory_df['Cluster'] = clusters

    # PCA for 2D Visualization
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)
    factory_df['PCA1'] = features_pca[:, 0]
    factory_df['PCA2'] = features_pca[:, 1]


    # Scatter plot of clusters
    st.subheader("Cluster Visualization (PCA 2D Projection)")
    fig = px.scatter(factory_df, x='PCA1', y='PCA2', color=factory_df['Cluster'].astype(str),
                     title=f"K-Means Clusters (k={optimal_k}) in 2D PCA Projection",
                     labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'})
    st.plotly_chart(fig)

    # Cluster Profiles
    st.subheader("Cluster Profiles")
    cluster_summary = factory_df.groupby('Cluster')[selected_features].mean()
    st.write("Average values for each cluster:")
    st.dataframe(cluster_summary)

    # Cluster Profile Visualization (Bar and Radar Charts)
    st.subheader("Cluster Profile Visualization")
    plot_type = st.radio("Select a plot type for cluster profiles:", ("Bar Plot", "Radar Chart"))

    if plot_type == "Bar Plot":
        fig = go.Figure()
        for cluster in cluster_summary.index:
            fig.add_trace(go.Bar(
                x=cluster_summary.columns,
                y=cluster_summary.loc[cluster],
                name=f'Cluster {cluster}'
            ))
        fig.update_layout(
            title="Cluster Profile Comparison (Average Values)",
            xaxis_title="Features",
            yaxis_title="Average Value",
            barmode="group",
            height=600,
            width=1000
        )
        st.plotly_chart(fig)

    elif plot_type == "Radar Chart":
        fig = go.Figure()
        for cluster in cluster_summary.index:
            fig.add_trace(go.Scatterpolar(
                r=cluster_summary.loc[cluster],
                theta=cluster_summary.columns,
                fill='toself',
                name=f'Cluster {cluster}'
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title="Cluster Profile Comparison (Radar Chart)",
            height=600,
            width=800
        )
        st.plotly_chart(fig)



    # Link Clusters to Error Analysis
    st.subheader("Cluster Error Analysis")

    # Filter only the columns that contain error data
    actual_error_columns = [col for col in factory_df.columns if col.endswith('_Error')]
    
    # Remove NaN values from error columns for averaging
    factory_df[actual_error_columns] = factory_df[actual_error_columns].fillna(0)

    if actual_error_columns:
        cluster_errors = factory_df.groupby('Cluster')[actual_error_columns].mean()

        # Display error analysis table
        st.write("Average Error per Measurement for Each Cluster:")
        st.dataframe(cluster_errors)

        # Identify the best-performing cluster based on overall average error
        cluster_errors['Avg_Error'] = cluster_errors.mean(axis=1)
        best_cluster = cluster_errors['Avg_Error'].idxmin()
        min_error_value = cluster_errors['Avg_Error'].min()

        # Display best-performing cluster
        st.markdown(f"### Best-Performing Cluster")
        st.write(f"Cluster {best_cluster} has the lowest average error (Avg. Error = {min_error_value:.2f}). This cluster can be considered the best-performing cluster based on error rates.")

        # Visualize average error per measurement across clusters
        fig = go.Figure()
        for cluster in cluster_errors.index:
            fig.add_trace(go.Bar(
                x=actual_error_columns,
                y=cluster_errors.loc[cluster, actual_error_columns],
                name=f'Cluster {cluster}'
            ))
        fig.update_layout(
            title="Average Error per Measurement Across Clusters",
            xaxis_title="Measurement",
            yaxis_title="Average Error",
            barmode="group",
            height=600,
            width=1000
        )
        st.plotly_chart(fig)
    else:
        st.write("No error columns found in the dataset.")









    # Anomaly Detection Based on Cluster Distance
    st.subheader("Anomaly Detection Based on Cluster Distance")

    # Calculate Euclidean distance to cluster center for each instance
    distances = []
    for i, center in enumerate(kmeans.cluster_centers_):
        cluster_data = features_scaled[clusters == i]
        cluster_distances = np.linalg.norm(cluster_data - center, axis=1)
        distances.extend(cluster_distances)

    # Add distances to the dataframe
    factory_df['DistanceToCenter'] = distances

    # Set a threshold for anomalies (e.g., 95th percentile of distances for each cluster)
    threshold = factory_df.groupby('Cluster')['DistanceToCenter'].transform(lambda x: x.quantile(0.95))
    factory_df['Anomaly'] = factory_df['DistanceToCenter'] > threshold

    # Display summary of anomalies per cluster
    anomaly_summary = factory_df.groupby('Cluster')['Anomaly'].sum().reset_index()
    anomaly_summary.columns = ['Cluster', 'Number of Anomalies']
    st.write("Anomaly Summary by Cluster")
    st.dataframe(anomaly_summary)

    # Visualize Anomalies in 2D PCA plot
    fig = px.scatter(factory_df, x='PCA1', y='PCA2', color='Cluster',
                     symbol='Anomaly', title="Anomalies Based on Cluster Distance (PCA 2D Projection)",
                     labels={'PCA1': 'Principal Component 1', 'PCA2': 'Principal Component 2'},
                     color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_traces(marker=dict(size=6))



    # Update layout of the PCA scatter plot with improved color bar for anomalies
    fig.update_layout(
        title="Anomalies Based on Cluster Distance (PCA 2D Projection)",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        coloraxis_colorbar=dict(
            title="Anomaly",
            tickvals=[0, 1],  # Shows only 0 and 1 for clarity
            ticktext=["False", "True"],
            len=0.5,          # Adjust length to fit better
            thickness=15,     # Adjust thickness for better visibility
            title_font=dict(size=12),  # Adjust font size for title
            tickfont=dict(size=10)     # Adjust font size for tick labels
        )
    )

    st.plotly_chart(fig)

    # Insights section
    st.markdown("""
    ### Insights on Anomalies
    - **Anomalies**: Instances marked as anomalies have a significantly high distance from their respective cluster centers.
    - **Threshold**: Anomalies are defined based on the 95th percentile distance within each cluster, highlighting the most unusual instances in each operational state.
    - **Practical Use**: Managers can investigate these anomalies to understand operational deviations, helping in process improvement and stability monitoring.
    """)









# 5. Predictions Section
if section == "Predictions":
    st.title("Predictions and Forecasting")
    st.write("Use this section to input new feature values and predict the measurements.")

    # Input feature values (using sidebar)
    user_inputs = [st.sidebar.number_input(f"Input value for {feature}", value=0.0) for feature in input_features]
    
    # Scale the user inputs
    user_inputs_scaled = scaler.transform([user_inputs])

    # Predict using the trained model
    predictions = model.predict(user_inputs_scaled)

    # Ensure predictions has the correct shape
    if predictions.ndim == 1 and predictions.shape[0] == len(target_measurements):
        predictions = predictions.reshape(1, -1)  # Reshape to match target measurements
    elif predictions.ndim == 1:
        st.error("The model is returning only one prediction value. Check that the model is trained to predict all target measurements.")
        st.stop()

    # Create DataFrame with predictions
    prediction_df = pd.DataFrame(predictions, columns=target_measurements)



    # NEW: Prediction Ranges or Confidence Intervals Section
    st.subheader("Predicted Measurements with Confidence Intervals")
    st.write("This chart shows each measurement's predicted value along with the 95% confidence interval.")

    # Calculate prediction intervals
    def calculate_prediction_intervals(model, X, confidence=0.95):
        # Get predictions from each tree in the forest
        all_tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
        
        # Calculate mean and standard deviation across trees
        mean_predictions = np.mean(all_tree_predictions, axis=0)
        std_dev = np.std(all_tree_predictions, axis=0)
        
        # Calculate the confidence interval
        z_score = 1.96  # for 95% confidence
        lower_bound = mean_predictions - z_score * std_dev
        upper_bound = mean_predictions + z_score * std_dev
        
        return mean_predictions, lower_bound, upper_bound

    # Calculate prediction intervals for user inputs
    mean_predictions, lower_bound, upper_bound = calculate_prediction_intervals(model, user_inputs_scaled)

    # Create a DataFrame with predictions and intervals
    interval_df = pd.DataFrame({
        'Measurement': target_measurements,
        'Mean Prediction': mean_predictions.flatten(),
        'Lower Bound': lower_bound.flatten(),
        'Upper Bound': upper_bound.flatten()
    })


    # Plot predictions with confidence intervals
    fig = go.Figure()

    # Add mean predictions as bars
    fig.add_trace(go.Bar(
        y=interval_df['Measurement'],
        x=interval_df['Mean Prediction'],
        text=interval_df['Mean Prediction'].round(2),  # Rounded predictions for cleaner display
        orientation='h',
        name="Mean Prediction",
        textposition='outside',  # Position text outside for clarity
        textfont=dict(size=12, color="red"),  # Adjust font size and color
        marker=dict(color="blue"),  # Adjust color if needed
    ))

    # Add confidence intervals as improved, darker grey error bars
    for i, measurement in enumerate(interval_df['Measurement']):
        fig.add_shape(type="line",
                      x0=interval_df['Lower Bound'][i], x1=interval_df['Upper Bound'][i],
                      y0=i, y1=i,
                      line=dict(color="grey", width=2, dash="dash"))  # Darker grey and thicker lines for clarity

    # Update layout to improve readability
    fig.update_layout(
        title="Predicted Measurements with 95% Confidence Intervals",
        xaxis_title="Predicted Value",
        yaxis_title="Measurement",
        height=800,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    st.plotly_chart(fig)

    # Feature Importance Analysis Section (after displaying the plot)
    st.subheader("Management Insights")

    # Adding a description of what the confidence intervals mean for management
    st.write("""
    This chart provides management with key insights into prediction reliability across various measurements:
    - **Narrower intervals** indicate more reliable predictions, suggesting stability in these measurements. Management can be confident in these predictions as they reflect lower variability.
    - **Wider intervals** highlight measurements with greater prediction uncertainty. These are areas where conditions might be more variable, or where the model is less certain in its output. Management should consider prioritizing monitoring or further analysis for these measurements.
  
    **Actionable Recommendations**:
    - Focus on measurements with **wider intervals** as they may benefit from further quality checks or adjustments in operational processes.
    - **Consistent adjustments** on such measurements may improve model accuracy over time, leading to more reliable forecasting and operational planning.
    """)





# 6. Feature Importance Analysis Section
if section == "Feature Importance Analysis":
    st.title("Feature Importance Analysis")
    st.write("This section displays the importance of each feature used in the prediction model.")
    
    final_rf_model = model  # Reference to the loaded model for feature importance

    # Extract feature importances
    feature_importances = final_rf_model.feature_importances_

    # Create a DataFrame to display feature names and their importance
    feature_importance_df = pd.DataFrame({
        'Feature': input_features,
        'Importance': feature_importances
    })

    # Sort features by importance
    sorted_features = feature_importance_df.sort_values(by='Importance', ascending=False)


    # Define brief insights for each feature based on its role
    feature_insights = {
        'Machine3.MaterialTemperature.U.Actual': 'Crucial for maintaining product quality and stability.',
        'Machine3.MotorRPM.C.Actual': 'Impacts machine speed, directly affecting production efficiency.',
        'FirstStage.CombinerOperation.Temperature1.U.Actual': 'Essential for optimal combining temperature, affects yield.',
        'AmbientConditions.AmbientHumidity.U.Actual': 'Environmental factor, can impact material properties.',
        'Machine1.MotorRPM.C.Actual': 'Controls speed, affecting operational consistency.',
        'Machine2.MotorAmperage.U.Actual': 'Indicates motor load, important for machine health.',
        # Add more annotations as needed or generalize where specific insights aren’t available.
    }

    # Annotate insights based on importance scores
    # Filter out the annotations for only the defined key insights
    sorted_features['Insight'] = sorted_features['Feature'].apply(lambda x: feature_insights.get(x, ''))

    # Create an enhanced bar plot with color coding and annotations
    fig = px.bar(
        sorted_features,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=px.colors.sequential.Blues,
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'}
    )

    # Add annotations as text beside each bar
    for index, row in sorted_features.iterrows():
        fig.add_annotation(
            x=row['Importance'],
            y=row['Feature'],
            text=row['Insight'],
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=10)
        )

    # Layout adjustments
    fig.update_layout(
        title="Feature Importance with Management Insights",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=800,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    st.plotly_chart(fig)

    # Management Summary and Recommendations
    st.subheader("Management Summary and Recommendations")
    st.write("""
    The **Feature Importance Analysis** reveals the critical features that significantly impact the predictive accuracy of our model.
    This analysis suggests that the following areas require focused attention:

    1. **Machine3.MaterialTemperature.U.Actual**: Since temperature control in Machine 3 has the highest importance, it’s crucial to maintain optimal temperature levels to ensure product quality and process stability.
    2. **Machine3.MotorRPM.C.Actual**: Motor speed in Machine 3 is also a key factor. Regular checks on RPM to avoid fluctuations can improve operational efficiency.
    3. **FirstStage.CombinerOperation.Temperature1.U.Actual**: This temperature setting in the combining stage impacts yield. Precise control here could reduce wastage and enhance output.
    4. **AmbientConditions.AmbientHumidity.U.Actual**: Ambient conditions, such as humidity, may indirectly impact raw material properties. Monitoring this can help in adapting to varying environmental conditions, especially during seasonal changes.

    ### Recommendations
    - **Prioritize Monitoring**: Regularly monitor high-importance features for any deviations. Automated alerts or dashboards can help track these key factors in real-time.
    - **Optimize Parameters**: Consider optimization efforts around high-importance features to maximize production quality and efficiency.
    - **Regular Maintenance**: Since some features are related to machine health, schedule routine maintenance to prevent breakdowns and maintain smooth operations.
    - **Environmental Controls**: Implement environmental control measures where feasible, especially around ambient conditions.

    These insights provide a roadmap for prioritizing efforts in areas that will likely yield the highest impact on production quality and efficiency.
    """)






# 6. Error Analysis Section
elif section == "Error Threshold Analysis":
    st.title("Error Analysis: Underperformance and Acceptable Range")
    
    # Section Introduction
    st.write("""
        In this section, we analyze measurements where actual values fall below setpoints, indicating underperformance.
        Use the **Acceptable Negative Error Range** slider in the sidebar to set the tolerance level for negative errors.
        
        - **Purple line**: Represents the actual error level for each measurement.
        - **Red dashed line**: Shows the user-defined acceptable threshold. Any error below this threshold is flagged as underperforming.
    """)

    # Sidebar slider for acceptable negative error threshold
    st.sidebar.header("Set Acceptable Negative Error Range")
    acceptable_error_threshold = st.sidebar.slider(
        "Select the acceptable negative error threshold:", 
        min_value=-20.0, max_value=0.0, value=-2.0, step=0.5
    )

    # Calculate errors (Actual - Setpoint) and store in a dictionary for easy access
    error_columns = {}
    for actual in stage1_actuals:
        setpoint = actual.replace("Actual", "Setpoint")
        if setpoint in factory_df.columns:
            error_col = f"{actual}_Error"
            factory_df[error_col] = factory_df[actual] - factory_df[setpoint]
            error_columns[actual] = error_col

    # Filter measurements exceeding the acceptable range
    underperforming_measurements = [
        col for col, error_col in error_columns.items()
        if (factory_df[error_col] < acceptable_error_threshold).any()
    ]

    # Display filtered measurements based on the threshold
    if underperforming_measurements:
        st.subheader(f"Measurements with Errors Below Acceptable Threshold of {acceptable_error_threshold}")
        
        # Show a message explaining the current view
        st.info(f"{len(underperforming_measurements)} measurement(s) exceed the acceptable error threshold.")

        # Plot error over time for each underperforming measurement
        for measurement in underperforming_measurements:
            error_col = error_columns[measurement]

            # Plot the error with indication of acceptable range
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=factory_df['time_stamp'], y=factory_df[error_col],
                mode='lines', name=f'{measurement} Error',
                line=dict(color='purple')
            ))

            # Add acceptable threshold line
            fig.add_hline(y=acceptable_error_threshold, line=dict(color='red', dash='dash'),
                          annotation_text="Acceptable Threshold", annotation_position="bottom right")

            # Update layout for readability
            fig.update_layout(
                title=f'{measurement} Error Over Time (Acceptable Threshold: {acceptable_error_threshold})',
                xaxis_title="Time",
                yaxis_title="Error Value",
                height=400,
                width=700,
            )

            # Display the Plotly chart in Streamlit
            st.plotly_chart(fig)

    else:
        # If no measurements exceed the threshold, show a message
        st.subheader("All measurements are within the acceptable error range.")
        st.success("No measurements are currently underperforming based on the selected threshold.")




# Error Distribution Analysis Section
elif section == "Error Distribution Analysis":
    st.title("Error Distribution Analysis")
    
    # Section Introduction
    st.write("""
        This section provides insights into the distribution of errors for each measurement. 
        Toggle between viewing the **Histogram with KDE** to assess frequency distributions and 
        the **Violin Plot** to visualize the spread and density of errors across multiple measurements.
        
        - **Histogram with KDE**: Shows the frequency and distribution shape for error values.
        - **Violin Plot**: Highlights variability and outliers in error values across measurements.
    """)

    # Sidebar: Choose plot type
    plot_type = st.radio("Select Plot Type:", ("Histogram with KDE", "Violin Plot"))

    # Calculate errors (Actual - Setpoint) for each measurement with a setpoint
    error_data = {}
    for actual in stage1_actuals:
        setpoint = actual.replace("Actual", "Setpoint")
        if setpoint in factory_df.columns:
            error_col = f"{actual}_Error"
            factory_df[error_col] = factory_df[actual] - factory_df[setpoint]
            error_data[actual] = error_col

    # Select a specific measurement for detailed view
    selected_error = st.selectbox("Choose a measurement to analyze:", list(error_data.values()))

    # Plot based on the selected plot type
    if plot_type == "Histogram with KDE":
        # Histogram with KDE using Plotly
        fig = px.histogram(factory_df, x=selected_error, nbins=30, marginal="violin", 
                           title=f"Histogram with KDE for {selected_error}", 
                           labels={selected_error: "Error Value"}, opacity=0.6)
        fig.update_traces(marker=dict(color='blue'), selector=dict(type="histogram"))
        st.plotly_chart(fig)

    elif plot_type == "Violin Plot":
        # Violin Plot using Plotly
        fig = go.Figure()
        fig.add_trace(go.Violin(
            y=factory_df[selected_error],
            box_visible=True,
            meanline_visible=True,
            line_color="blue",
            fillcolor="lightblue",
            opacity=0.6,
            points="all",  # Shows all data points for a detailed view
            jitter=0.2,
            name=selected_error,
            hoveron="violins+points+kde"     # Show hover only for the violin body, not individual points

        ))


        fig.update_layout(title=f"Violin Plot of {selected_error}", 
                          yaxis_title="Error Value", height=600, width=700)
        st.plotly_chart(fig)


# Anomaly Detection Section
elif section == "Anomaly Detection":
    st.title("Anomaly Detection")

    # Section Introduction
    st.write("""
        This section detects anomalies in measurements based on deviations from setpoints.
        Anomalies indicate points that fall significantly outside the expected range,
        helping managers quickly identify issues requiring immediate attention.
        
        - **Blue line**: Actual measurement values.
        - **Red dots**: Detected anomalies based on user-defined sensitivity.
    """)

    # Sidebar slider for sensitivity setting
    st.sidebar.header("Set Anomaly Sensitivity")
    sensitivity = st.sidebar.slider(
        "Select sensitivity level (IQR multiplier):",
        min_value=1.0, max_value=3.0, value=1.5, step=0.5
    )

    # Choose measurement to analyze for anomalies
    selected_measurement_actual = st.selectbox("Choose a measurement to analyze:", stage1_actuals)
    selected_measurement_setpoint = selected_measurement_actual.replace("Actual", "Setpoint")

    # Calculate IQR and identify anomalies for the selected measurement
    q1 = factory_df[selected_measurement_actual].quantile(0.25)
    q3 = factory_df[selected_measurement_actual].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - sensitivity * iqr
    upper_bound = q3 + sensitivity * iqr

    # Identify anomaly points
    anomalies = factory_df[
        (factory_df[selected_measurement_actual] < lower_bound) | 
        (factory_df[selected_measurement_actual] > upper_bound)
    ]

    # Plot actual values with anomalies highlighted
    fig = go.Figure()

    # Add actual values trace
    fig.add_trace(go.Scatter(
        x=factory_df['time_stamp'],
        y=factory_df[selected_measurement_actual],
        mode='lines',
        name=f'{selected_measurement_actual} (Actual)',
        line=dict(color='blue')
    ))

    # Add setpoint trace if it exists
    if selected_measurement_setpoint in factory_df.columns:
        fig.add_trace(go.Scatter(
            x=factory_df['time_stamp'],
            y=factory_df[selected_measurement_setpoint],
            mode='lines',
            name=f'{selected_measurement_setpoint} (Setpoint)',
            line=dict(color='gray', dash='dash')
        ))

    # Highlight anomalies as red points
    fig.add_trace(go.Scatter(
        x=anomalies['time_stamp'],
        y=anomalies[selected_measurement_actual],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=8, symbol='circle')
    ))

    # Update layout with title and axis labels
    fig.update_layout(
        title=f"Anomaly Detection for {selected_measurement_actual} (Sensitivity: {sensitivity}x IQR)",
        xaxis_title="Time",
        yaxis_title="Measurement Value",
        height=600,
        width=1000
    )

    # Display the plot
    st.plotly_chart(fig)

    # Display summary of anomalies
    st.subheader("Anomaly Summary")
    st.write(f"Detected {len(anomalies)} anomalies in {selected_measurement_actual} with sensitivity {sensitivity}x IQR.")
    st.dataframe(anomalies[['time_stamp', selected_measurement_actual]])



# Cumulative Error Tracking Section
elif section == "Cumulative Error Tracking":
    st.title("Cumulative Error Tracking")

    # Section Introduction
    st.write("""
        This section shows cumulative error for each measurement over time. Cumulative error tracks the accumulation of deviations 
        from the setpoint, helping identify consistent overperformance or underperformance.
        
        - **Positive cumulative error**: Indicates the measurement tends to exceed the setpoint.
        - **Negative cumulative error**: Indicates the measurement tends to fall below the setpoint.
    """)

    # Select measurement to analyze for cumulative error
    selected_measurement_actual = st.selectbox("Choose a measurement to track cumulative error:", stage1_actuals)
    selected_measurement_setpoint = selected_measurement_actual.replace("Actual", "Setpoint")

    # Convert time_stamp to datetime if not already
    factory_df['time_stamp'] = pd.to_datetime(factory_df['time_stamp'])

    # Timestamp picker for selecting the start point
    start_timestamp = st.select_slider(
        "Select the start timestamp for cumulative error tracking:",
        options=factory_df['time_stamp'].unique(),
        value=factory_df['time_stamp'].min()
    )

    # Filter the dataframe based on the selected start timestamp
    df_filtered = factory_df[factory_df['time_stamp'] >= pd.Timestamp(start_timestamp)]

    # Calculate cumulative error starting from the selected timestamp
    if selected_measurement_setpoint in df_filtered.columns:
        # Calculate error (Actual - Setpoint) and cumulative sum of errors starting from selected timestamp
        df_filtered['Error'] = df_filtered[selected_measurement_actual] - df_filtered[selected_measurement_setpoint]
        df_filtered['Cumulative_Error'] = df_filtered['Error'].cumsum()

        # Plot cumulative error over time
        fig = go.Figure()

        # Cumulative error trace
        fig.add_trace(go.Scatter(
            x=df_filtered['time_stamp'],
            y=df_filtered['Cumulative_Error'],
            mode='lines',
            name='Cumulative Error',
            line=dict(color='purple')
        ))

        # Zero reference line to indicate balanced performance
        fig.add_hline(y=0, line=dict(color='gray', dash='dash'),
                      annotation_text="Target Setpoint", annotation_position="bottom right")

        # Update layout with titles and labels
        fig.update_layout(
            title=f"Cumulative Error Tracking for {selected_measurement_actual} (From {start_timestamp})",
            xaxis_title="Time",
            yaxis_title="Cumulative Error",
            height=600,
            width=1000
        )

        # Display the plot
        st.plotly_chart(fig)

        # Show current cumulative error
        st.subheader("Current Cumulative Error Summary")
        st.write(f"The cumulative error for {selected_measurement_actual} starting from {start_timestamp} is: {df_filtered['Cumulative_Error'].iloc[-1]:.2f}")
    else:
        st.write("Setpoint data not available for the selected measurement.")






# Correlation Analysis Section
elif section == "Correlation Analysis":
    st.title("Correlation Analysis")

    # Section Introduction
    st.write("""
        This section provides a correlation analysis between selected machine parameters, ambient conditions, 
        and the selected error value (Actual - Setpoint). The correlation heatmap helps identify relationships that might influence errors, 
        enabling targeted adjustments to improve performance.
        
        - **Positive correlation**: Variables increase or decrease together.
        - **Negative correlation**: As one variable increases, the other decreases.
    """)

    # Define selected features (ambient conditions, machine parameters, etc.)
    selected_features = [
        'AmbientConditions.AmbientHumidity.U.Actual', 'AmbientConditions.AmbientTemperature.U.Actual',
        'Machine1.RawMaterial.Property1', 'Machine1.RawMaterial.Property2', 'Machine1.RawMaterial.Property3', 'Machine1.RawMaterial.Property4',
        'Machine1.RawMaterialFeederParameter.U.Actual', 'Machine1.Zone1Temperature.C.Actual', 'Machine1.Zone2Temperature.C.Actual',
        'Machine1.MotorAmperage.U.Actual', 'Machine1.MotorRPM.C.Actual', 'Machine1.MaterialPressure.U.Actual', 'Machine1.MaterialTemperature.U.Actual',
        'Machine1.ExitZoneTemperature.C.Actual', 'Machine2.RawMaterial.Property1', 'Machine2.RawMaterial.Property2',
        'Machine2.RawMaterial.Property3', 'Machine2.RawMaterial.Property4', 'Machine2.RawMaterialFeederParameter.U.Actual',
        'Machine2.Zone1Temperature.C.Actual', 'Machine2.Zone2Temperature.C.Actual', 'Machine2.MotorAmperage.U.Actual',
        'Machine2.MotorRPM.C.Actual', 'Machine2.MaterialPressure.U.Actual', 'Machine2.MaterialTemperature.U.Actual',
        'Machine2.ExitZoneTemperature.C.Actual', 'Machine3.RawMaterial.Property1', 'Machine3.RawMaterial.Property2',
        'Machine3.RawMaterial.Property3', 'Machine3.RawMaterial.Property4', 'Machine3.RawMaterialFeederParameter.U.Actual',
        'Machine3.Zone1Temperature.C.Actual', 'Machine3.Zone2Temperature.C.Actual', 'Machine3.MotorAmperage.U.Actual',
        'Machine3.MotorRPM.C.Actual', 'Machine3.MaterialPressure.U.Actual', 'Machine3.MaterialTemperature.U.Actual',
        'Machine3.ExitZoneTemperature.C.Actual', 'FirstStage.CombinerOperation.Temperature1.U.Actual',
        'FirstStage.CombinerOperation.Temperature2.U.Actual', 'FirstStage.CombinerOperation.Temperature3.C.Actual'
    ]

    # Calculate error columns for relevant measurements
    error_columns = {}
    for col in factory_df.columns:
        if col.endswith(".Actual") and col.replace("Actual", "Setpoint") in factory_df.columns:
            error_col = col.replace(".Actual", ".Error")
            factory_df[error_col] = factory_df[col] - factory_df[col.replace("Actual", "Setpoint")]
            error_columns[col] = error_col

    # Dropdown to select specific error for analysis
    selected_error = st.selectbox("Select a measurement error to analyze correlations:", options=list(error_columns.values()))

    # Subset dataframe for correlation calculation
    correlation_features = selected_features + [selected_error]
    df_corr = factory_df[correlation_features].corr()

    # Isolate the selected error's correlations with other variables
    df_corr_selected = df_corr[[selected_error]].drop(index=[selected_error])

    # Plot correlation heatmap for the selected error
    fig = go.Figure(data=go.Heatmap(
        z=df_corr_selected[selected_error].values.reshape(-1, 1),
        x=[selected_error],
        y=df_corr_selected.index,
        colorscale="Viridis",
        colorbar=dict(title="Correlation Coefficient"),
        zmin=-1,  # Set minimum for color scale to -1
        zmax=1    # Set maximum for color scale to 1
    ))

    # Customize layout
    fig.update_layout(
        title=f"Correlation Heatmap for {selected_error} with Selected Variables",
        xaxis_nticks=1,  # Only show the selected error on the x-axis
        height=800,
        width=600
    )

    # Display heatmap in Streamlit
    st.plotly_chart(fig)

    # Insights section
    st.markdown(
        """
        ### Insights
        - **Strong Positive/Negative Correlations**: Look for variables with high correlation with the selected error, indicating potential areas for adjustment.
        - **Zero or Low Correlations**: Variables with low correlations might have minimal impact on process errors.
        """
    )



