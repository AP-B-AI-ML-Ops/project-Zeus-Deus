import pandas as pd
import matplotlib.pyplot as plt
import json

# Read the CSV
df = pd.read_csv('/reports/metrics.csv')

# Convert value column to numeric where possible
def parse_value(val):
    try:
        # Try to parse as JSON if it's a dictionary string
        if isinstance(val, str) and val.startswith('{'):
            return float(json.loads(val.replace("'", '"')).get('share', 0))
        # Otherwise try to convert to float
        return float(val)
    except (ValueError, AttributeError):
        return None

# Apply the parsing
df['numeric_value'] = df['value'].apply(parse_value)

# Pivot the data to have metrics as columns
pivot_df = df.pivot(index='run_time', columns='metric_name', values='numeric_value')

# Plot
pivot_df.plot(figsize=(12, 6))
plt.title('Model Metrics Over Time')
plt.ylabel('Metric Value')
plt.tight_layout()
plt.savefig('/reports/metrics_plot.png')
print("Plot saved to /reports/metrics_plot.png")
print(df.head())

# Check for numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
print("Numeric columns found:", numeric_cols.tolist())

if not numeric_cols.empty:
    # Plot only numeric columns
    df[numeric_cols].plot(figsize=(10, 6))
    plt.title('Model Metrics Over Time')
    plt.tight_layout()
    plt.savefig('/reports/metrics_plot.png')
    print("Plot saved to /reports/metrics_plot.png")
else:
    print("Error: No numeric data found to plot. Data types:")
    print(df.dtypes)