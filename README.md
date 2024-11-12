# Capstone Project: Continuous Factory Process Analysis

This capstone project is a web-based data analysis tool built using Streamlit to explore, analyze, and monitor data from a continuous factory process. The app provides various dashboards and analytical tools to gain insights into key metrics, trends, and anomalies within the dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Sections](#sections)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Project Overview

This application is designed to help users understand and analyze the continuous factory process dataset through interactive visualizations and predictive models. The project includes data exploration, error analysis, anomaly detection, and other advanced analytical methods to provide a comprehensive view of the process.

## Features

- **Interactive Dashboards**: Explore the dataset with various visualizations and filter options.
- **Trend Analysis**: Compare actual measurements to setpoints to identify deviations.
- **Clustering and Predictions**: Use machine learning to identify patterns and predict future measurements.
- **Error Analysis and Anomaly Detection**: Gain insights into errors and detect unusual patterns.
- **Cumulative Tracking and Correlation Analysis**: Monitor cumulative trends and correlations.

## Sections

### 1. Dataset Overview Dashboard

## Overview Tab

The Overview Tab provides foundational insights into the dataset, offering a variety of visualizations to help understand data structure, variability, and feature relationships. Each sub-tab includes specific analyses:

### 1. Data Distribution
- **Purpose**: Displays the spread of values for each feature, helping to identify typical ranges, trends, and any significant outliers.
- **Displays**: Histograms and density plots for each main feature, highlighting common values and any unusual data points.

### 2. Box Plots
- **Purpose**: Highlights the variability and identifies outliers for each feature, crucial for assessing stability in each process variable.
- **Displays**: Box plots with median, quartile ranges, and outliers, offering insights into each feature’s consistency and range.

### 3. Correlation Heatmap
- **Purpose**: Visualizes relationships between features, enabling identification of highly correlated pairs, which may indicate dependencies or interactions.
- **Displays**: A heatmap showing correlation coefficients for feature pairs, with color gradients reflecting the strength and direction of relationships.

### 4. Time Series Trends
- **Purpose**: Tracks changes in features over time, useful for detecting patterns, seasonal effects, or shifts that might impact output measurements.
- **Displays**: Line charts for time-based data, providing a view of feature trends across different time intervals.


*(Add additional tabs and descriptions as necessary.)*

### 2. Trend Analysis (Actual vs. Setpoint)

Analyze trends by comparing actual measurements against setpoints to identify potential deviations in the manufacturing process.

### 3. EDA (Exploratory Data Analysis)

Conduct exploratory data analysis on the dataset, providing insights into distributions, correlations, and basic statistical summaries.

### 4. Clustering

Apply clustering techniques to group similar patterns within the dataset, revealing potential patterns or distinct groups in the data.

### 5. Predictions

Generate predictions for key metrics and display them alongside confidence intervals, allowing users to anticipate future values.

### 6. Feature Importance Analysis

Display feature importance rankings to understand which factors contribute the most to predictive models.

### 7. Error Threshold Analysis

Identify when measurements exceed predefined error thresholds, helping to monitor performance consistency.

### 8. Error Distribution Analysis

Visualize the distribution of errors to understand how often deviations occur and their magnitudes.

### 9. Anomaly Detection

Detect and highlight anomalies in the data, which may indicate issues in the factory process that require attention.

### 10. Cumulative Error Tracking

Track cumulative errors over time to monitor long-term trends and potential areas of improvement.

### 11. Correlation Analysis

Analyze correlations between different variables to understand relationships and potential dependencies within the data.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/noora-q0/FactoryCapstoneAppV3.git

