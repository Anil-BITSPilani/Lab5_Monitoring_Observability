import pandas as pd
import json
import logging
from datetime import datetime, timedelta

# --- Configure Structured Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ml_monitoring')

# --- 1. Load Logs and Training Stats ---
def load_data(log_path, stats_path):
    logger.info(f"Loading production logs from {log_path}")
    df = pd.read_csv(log_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    logger.info(f"Loaded {len(df)} log entries")
    
    logger.info(f"Loading training statistics from {stats_path}")
    with open(stats_path, "r") as f:
        training_stats = json.load(f)
    return df, training_stats

# --- 2. Filter to a Recent Time Window ---
def filter_to_window(df, days=7):
    # Use the most recent timestamp in the data as reference
    max_timestamp = df["timestamp"].max()
    cutoff_date = max_timestamp - timedelta(days=days)
    recent_window = df[df["timestamp"] >= cutoff_date]
    logger.info(f"Filtered to {len(recent_window)} entries from last {days} days (since {cutoff_date})")
    return recent_window

# --- 3. Compute System Metrics ---
def compute_system_metrics(df):
    logger.info("Computing system performance metrics")
    latency_mean = df["latency_ms"].mean()
    latency_p95 = df["latency_ms"].quantile(0.95)
    latency_p99 = df["latency_ms"].quantile(0.99)
    
    total_requests = len(df)
    error_requests = len(df[df["status_code"] != 200])
    error_rate = (error_requests / total_requests) * 100 if total_requests > 0 else 0
    
    system_metrics = {
        "Total Requests (last 7 days)": total_requests,
        "Error Rate (%)": f"{error_rate:.2f}",
        "Mean Latency (ms)": f"{latency_mean:.2f}",
        "p95 Latency (ms)": f"{latency_p95:.2f}",
        "p99 Latency (ms)": f"{latency_p99:.2f}",
    }
    
    # Log structured metrics
    logger.info("System metrics computed", extra={
        "total_requests": total_requests,
        "error_rate": error_rate,
        "latency_p95": latency_p95,
        "latency_p99": latency_p99
    })
    
    return system_metrics

# --- 4. Compute and Compare Data Metrics ---
def compute_data_metrics(df, training_stats):
    logger.info("Computing data quality and drift metrics")
    comparison_table = []
    
    # Numeric feature comparison
    for feature in ["feature_1", "feature_2"]:
        prod_mean = df[feature].mean()
        prod_std = df[feature].std()
        train_mean = training_stats[feature]["mean"]
        train_std = training_stats[feature]["std"]
        
        # Check for data quality issues
        null_count = df[feature].isnull().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count} null values in {feature}")
        
        comparison_table.append({
            "feature": feature,
            "train_mean": train_mean,
            "prod_mean": f"{prod_mean:.2f}",
            "diff": f"{prod_mean - train_mean:.2f}",
            "train_std": train_std,
            "prod_std": f"{prod_std:.2f}",
        })
        
    # Categorical feature comparison
    prod_counts = df["feature_3"].value_counts(normalize=True)
    train_counts = pd.Series(training_stats["feature_3"]["value_counts"])
    train_counts = train_counts / train_counts.sum()
    
    logger.info("Categorical feature distribution comparison:")
    print("\nCategorical Feature: feature_3 Distribution")
    print("--- Training ---")
    print(train_counts)
    print("--- Production (last 7 days) ---")
    print(prod_counts)
    
    return pd.DataFrame(comparison_table)

# --- Main Execution ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LAB PART 1: System & Data Metrics")
    print("="*60)
    
    print("\n📚 LEARNING OBJECTIVE:")
    print("   Learn to analyze production logs and compute key monitoring metrics.")
    print("   You'll calculate system health metrics (latency, errors) and compare")
    print("   data distributions between training and production.")
    
    print("\n📊 DATA SOURCE:")
    print("   - data/logs.csv: Simulated production logs (17 requests)")
    print("   - data/training_stats.json: Baseline statistics from training data")
    
    print("\n💡 WHAT YOU'LL SEE:")
    print("   1. System metrics: request count, error rate, latency percentiles")
    print("   2. Data drift: feature statistics comparison (training vs production)")
    print("   3. Structured logs: monitoring.log for aggregation/alerting")
    print("\n" + "="*60 + "\n")
    
    logger.info("="*60)
    logger.info("Starting ML Monitoring - System & Data Metrics Analysis")
    logger.info("="*60)
    
    LOG_PATH = "data/logs.csv"
    STATS_PATH = "data/training_stats.json"
    
    logs_df, training_stats = load_data(LOG_PATH, STATS_PATH)
    recent_logs = filter_to_window(logs_df, days=7)
    
    print("\n--- System Metrics ---")
    system_metrics = compute_system_metrics(recent_logs)
    for key, value in system_metrics.items():
        print(f"{key}: {value}")
        
    print("\n--- Data Drift Comparison (Numeric) ---")
    comparison_df = compute_data_metrics(recent_logs, training_stats)
    print(comparison_df.to_string(index=False))
    
    logger.info("Analysis complete - check monitoring.log for structured logs")
