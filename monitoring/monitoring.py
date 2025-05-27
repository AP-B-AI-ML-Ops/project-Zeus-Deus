from evidently import Report
import pandas as pd
from evidently.presets import DataDriftPreset
from datetime import datetime, timezone
from prefect import task, Flow

@task
@task
def load_batch_data():
    reference = pd.read_csv("/data/output/batch_predictions_v1.csv")
    current = pd.read_csv("/data/output/batch_predictions_v2.csv")
    return reference, current

@task
def generate_report(reference, current):
    """Generate an Evidently report comparing reference and current data"""
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    return snapshot

@task
def extract_metrics(snapshot):
    """Extract metrics from the Evidently report snapshot"""
    json_data = snapshot.dict()
    result_data = []
    report_time = datetime.now(timezone.utc)
    for metric in json_data["metrics"]:
        metric_id = metric["metric_id"]
        value = metric["value"]
        result_data.append(
            {"run_time": report_time, "metric_name": metric_id, "value": value}
        )
    return pd.DataFrame(result_data)

@task
def save_metrics(metrics_df):
    """Save metrics to a CSV file"""
    metrics_df.to_csv("/reports/metrics.csv", index=False)

@Flow
def main():
    reference, current = load_batch_data()
    snapshot = generate_report(reference, current)
    metrics_df = extract_metrics(snapshot)
    save_metrics(metrics_df)
    
if __name__ == "__main__":
    main()
    