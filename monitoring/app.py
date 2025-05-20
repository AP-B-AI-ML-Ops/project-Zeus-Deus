import os
import sys
import json
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric
import logging

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set MLflow tracking URI from environment variable or use default
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_data(data_path):
    """Load data from a JSON file."""
    logger.info(f"Loading data from {data_path}")
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """Preprocess the data for monitoring."""
    logger.info("Preprocessing data for monitoring")
    
    # Create an empty list to hold the flattened records
    flattened_records = []
    
    # Process each record
    for record in data:
        # Skip records with missing weather or vegetation data
        if record['weather'] is None or record['vegetation'] is None:
            continue
            
        # Create a flattened dictionary
        flat_record = {
            # Event features
            'lat': float(record['event']['lat']),
            'lon': float(record['event']['lon']),
            'brightness': float(record['event']['brightness']),
            'scan': float(record['event']['scan']),
            'track': float(record['event']['track']),
            'confidence': float(record['event']['confidence']),
            'bright_t31': float(record['event']['bright_t31']),
            'frp': float(record['event']['frp']),
            'daynight': 1 if record['event']['daynight'] == "D" else 0,  # Day=1, Night=0
            
            # Weather features
            'tavg': record['weather'].get('tavg'),
            'tmin': record['weather'].get('tmin'),
            'tmax': record['weather'].get('tmax'),
            'prcp': record['weather'].get('prcp'),
            'wspd': record['weather'].get('wspd'),
            'pres': record['weather'].get('pres'),
            
            # Vegetation features
            'ndvi': record['vegetation'].get('ndvi'),
            'evi': record['vegetation'].get('evi'),
            
            # Target: high severity fire (confidence > 80 and frp > 20)
            'high_severity': 1 if (float(record['event']['confidence']) > 80 and 
                                float(record['event']['frp']) > 20) else 0
        }
        
        flattened_records.append(flat_record)
    
    # Convert to DataFrame
    df = pd.DataFrame(flattened_records)
    
    # Replace NaN values with median for each column
    for col in df.columns:
        if col != 'high_severity' and df[col].dtype in [np.float64, np.int64]:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    logger.info(f"Preprocessed data shape: {df.shape}")
    return df

def generate_data_drift_report(reference_data, current_data):
    """Generate a data drift report using Evidently."""
    logger.info("Generating data drift report")
    
    # For evidently 0.2.8, we need a simpler column mapping without 'embeddings'
    column_mapping = {
        'target': 'high_severity',
        'prediction': None,
        'numerical_features': [
            'brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp',
            'tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres', 'ndvi', 'evi'
        ],
        'categorical_features': ['daynight']
    }
    
    # Create a data drift report
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])
    
    # Calculate the report
    try:
        data_drift_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Save the report to HTML
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/data_drift_{timestamp}.html"
        data_drift_report.save_html(report_path)
        
        return data_drift_report, report_path
    except Exception as e:
        logger.error(f"Error generating data drift report: {e}")
        return None, None

def generate_target_drift_report(reference_data, current_data):
    """Generate a target drift report using Evidently."""
    logger.info("Generating target drift report")
    
    # For evidently 0.2.8, we need a simpler column mapping without 'embeddings'
    column_mapping = {
        'target': 'high_severity',
        'prediction': None,
        'numerical_features': [
            'brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp',
            'tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres', 'ndvi', 'evi'
        ],
        'categorical_features': ['daynight']
    }
    
    # Create a target drift report
    target_drift_report = Report(metrics=[
        TargetDriftPreset()
    ])
    
    # Calculate the report
    try:
        target_drift_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Save the report to HTML
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/target_drift_{timestamp}.html"
        target_drift_report.save_html(report_path)
        
        return target_drift_report, report_path
    except Exception as e:
        logger.error(f"Error generating target drift report: {e}")
        return None, None

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Define the layout
app.layout = html.Div([
    html.H1("Wildfire Prediction Model Monitoring"),
    
    html.Div([
        html.H2("Data Selection"),
        html.Div([
            html.Label("Reference Dataset:"),
            dcc.Dropdown(
                id='reference-dataset',
                options=[
                    {'label': 'Training Data', 'value': 'training'},
                    {'label': 'Validation Data', 'value': 'validation'}
                ],
                value='training'
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Current Dataset:"),
            dcc.Dropdown(
                id='current-dataset',
                options=[
                    {'label': 'Test Data', 'value': 'test'},
                    {'label': 'Production Data', 'value': 'production'}
                ],
                value='test'
            ),
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    
    html.Div([
        html.H2("Data Drift Analysis"),
        html.Button('Generate Reports', id='generate-button'),
        html.Div(id='data-drift-output'),
        html.Div(id='target-drift-output')
    ]),
    
    html.Div([
        html.H2("Model Performance Over Time"),
        dcc.Graph(id='model-performance-graph')
    ])
])

@app.callback(
    [Output('data-drift-output', 'children'),
     Output('target-drift-output', 'children'),
     Output('model-performance-graph', 'figure')],
    [Input('generate-button', 'n_clicks')],
    [dash.dependencies.State('reference-dataset', 'value'),
     dash.dependencies.State('current-dataset', 'value')]
)
def update_output(n_clicks, reference_dataset, current_dataset):
    if n_clicks is None:
        # Initial load, return empty outputs
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data selected")
        return "Select datasets and click 'Generate Reports'", "", empty_fig
    
    try:
        # Load reference data
        if reference_dataset == 'training':
            reference_data_path = "data/processed/training_data.json"
        else:
            reference_data_path = "data/processed/validation_data.json"
        
        # Load current data
        if current_dataset == 'test':
            current_data_path = "data/processed/test_data.json"
        else:
            current_data_path = "data/processed/production_data.json"
        
        # Check if files exist
        if not os.path.exists(reference_data_path):
            reference_data_path = "data/merged_complete.json"
        
        if not os.path.exists(current_data_path):
            current_data_path = "data/merged_complete.json"
        
        # Load and preprocess data
        reference_raw = load_data(reference_data_path)
        current_raw = load_data(current_data_path)
        
        if reference_raw is None or current_raw is None:
            return "Error loading data. Check if the data files exist.", "", go.Figure()
        
        reference_df = preprocess_data(reference_raw)
        current_df = preprocess_data(current_raw)
        
        # Generate reports
        data_drift_report, data_drift_path = generate_data_drift_report(reference_df, current_df)
        target_drift_report, target_drift_path = generate_target_drift_report(reference_df, current_df)
        
        if data_drift_path is None:
            data_drift_summary = html.Div([
                html.H3("Data Drift Analysis Error"),
                html.P("Could not generate data drift report. Check logs for details.")
            ])
        else:
            data_drift_summary = html.Div([
                html.H3("Data Drift Summary"),
                html.P("Data drift analysis completed."),
                html.A("View Data Drift Report", href=f"/{data_drift_path}", target="_blank")
            ])
        
        if target_drift_path is None:
            target_drift_summary = html.Div([
                html.H3("Target Drift Analysis Error"),
                html.P("Could not generate target drift report. Check logs for details.")
            ])
        else:
            target_drift_summary = html.Div([
                html.H3("Target Drift Summary"),
                html.P("Target drift analysis completed."),
                html.A("View Target Drift Report", href=f"/{target_drift_path}", target="_blank")
            ])
        
        # Get model performance metrics from MLflow
        client = MlflowClient()
        
        try:
            experiment = mlflow.get_experiment_by_name("wildfire-prediction")
            
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                
                if len(runs) > 0:
                    # Extract metrics and timestamps
                    timestamps = pd.to_datetime(runs['start_time']).dt.strftime('%Y-%m-%d')
                    
                    # Create performance graph
                    performance_fig = go.Figure()
                    
                    if 'metrics.accuracy' in runs.columns:
                        accuracy = runs['metrics.accuracy']
                        performance_fig.add_trace(go.Scatter(x=timestamps, y=accuracy, mode='lines+markers', name='Accuracy'))
                    
                    # Add precision and recall if available
                    if 'metrics.precision' in runs.columns:
                        precision = runs['metrics.precision']
                        performance_fig.add_trace(go.Scatter(x=timestamps, y=precision, mode='lines+markers', name='Precision'))
                    
                    if 'metrics.recall' in runs.columns:
                        recall = runs['metrics.recall']
                        performance_fig.add_trace(go.Scatter(x=timestamps, y=recall, mode='lines+markers', name='Recall'))
                    
                    performance_fig.update_layout(
                        title="Model Performance Metrics Over Time",
                        xaxis_title="Date",
                        yaxis_title="Metric Value",
                        legend_title="Metrics"
                    )
                else:
                    performance_fig = go.Figure()
                    performance_fig.update_layout(title="No model runs found in MLflow")
            else:
                performance_fig = go.Figure()
                performance_fig.update_layout(title="MLflow experiment not found")
        except Exception as mlflow_error:
            logger.error(f"Error accessing MLflow: {mlflow_error}")
            performance_fig = go.Figure()
            performance_fig.update_layout(title="Error accessing MLflow data")
        
        return data_drift_summary, target_drift_summary, performance_fig
    
    except Exception as e:
        logger.error(f"Error generating reports: {e}")
        return f"Error generating reports: {str(e)}", "", go.Figure()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050) 