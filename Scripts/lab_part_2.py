import pandas as pd
import numpy as np
import json
import yaml
import logging
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_monitoring_alerts')

# --- Alert Class ---
class Alert:
    """Represents a monitoring alert"""
    def __init__(self, metric_name, value, threshold, severity, message):
        self.metric_name = metric_name
        self.value = value
        self.threshold = threshold
        self.severity = severity
        self.message = message
        self.timestamp = datetime.now()
    
    def __repr__(self):
        return f"[{self.severity.upper()}] {self.message}: {self.metric_name}={self.value:.4f} (threshold={self.threshold})"
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity,
            "metric": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message
        }

# --- Alert Manager ---
class AlertManager:
    """Manages alert evaluation and triggering"""
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.alerts = []
    
    def evaluate_system_metrics(self, metrics):
        """Evaluate system metrics against thresholds"""
        # Error rate check
        error_rate = float(metrics["Error Rate (%)"])
        if error_rate > self.config['system_metrics']['error_rate']['threshold']:
            alert = Alert(
                "error_rate",
                error_rate,
                self.config['system_metrics']['error_rate']['threshold'],
                self.config['system_metrics']['error_rate']['severity'],
                self.config['system_metrics']['error_rate']['message']
            )
            self.alerts.append(alert)
            logger.warning(str(alert))
        
        # Latency checks
        latency_p95 = float(metrics["p95 Latency (ms)"])
        if latency_p95 > self.config['system_metrics']['latency_p95']['threshold']:
            alert = Alert(
                "latency_p95",
                latency_p95,
                self.config['system_metrics']['latency_p95']['threshold'],
                self.config['system_metrics']['latency_p95']['severity'],
                self.config['system_metrics']['latency_p95']['message']
            )
            self.alerts.append(alert)
            logger.warning(str(alert))
    
    def evaluate_drift(self, psi_scores):
        """Evaluate drift metrics against thresholds"""
        for feature, psi in psi_scores.items():
            # Critical threshold
            if psi > self.config['data_drift']['psi']['threshold']:
                alert = Alert(
                    f"{feature}_psi",
                    psi,
                    self.config['data_drift']['psi']['threshold'],
                    self.config['data_drift']['psi']['severity'],
                    f"Critical drift in {feature}"
                )
                self.alerts.append(alert)
                logger.error(str(alert))
            # Warning threshold
            elif psi > self.config['data_drift']['psi_warning']['threshold']:
                alert = Alert(
                    f"{feature}_psi",
                    psi,
                    self.config['data_drift']['psi_warning']['threshold'],
                    self.config['data_drift']['psi_warning']['severity'],
                    f"Warning: drift detected in {feature}"
                )
                self.alerts.append(alert)
                logger.warning(str(alert))
    
    def evaluate_model_performance(self, perf_metrics, training_stats):
        """Evaluate model performance metrics"""
        # AUC check
        current_auc = perf_metrics['auc']
        baseline_auc = training_stats['model_performance']['auc']
        
        if current_auc < self.config['model_performance']['auc']['min_threshold']:
            alert = Alert(
                "auc",
                current_auc,
                self.config['model_performance']['auc']['min_threshold'],
                self.config['model_performance']['auc']['severity'],
                self.config['model_performance']['auc']['message']
            )
            self.alerts.append(alert)
            logger.error(str(alert))
        
        # AUC degradation check
        degradation_percent = ((baseline_auc - current_auc) / baseline_auc) * 100
        if degradation_percent > self.config['model_performance']['auc_degradation']['percent_drop']:
            alert = Alert(
                "auc_degradation",
                degradation_percent,
                self.config['model_performance']['auc_degradation']['percent_drop'],
                self.config['model_performance']['auc_degradation']['severity'],
                f"AUC degraded by {degradation_percent:.1f}% from baseline"
            )
            self.alerts.append(alert)
            logger.warning(str(alert))
        
        # Recall check
        if perf_metrics['recall'] < self.config['model_performance']['recall']['min_threshold']:
            alert = Alert(
                "recall",
                perf_metrics['recall'],
                self.config['model_performance']['recall']['min_threshold'],
                self.config['model_performance']['recall']['severity'],
                self.config['model_performance']['recall']['message']
            )
            self.alerts.append(alert)
            logger.warning(str(alert))
    
    def get_alerts_summary(self):
        """Get summary of all triggered alerts"""
        if not self.alerts:
            return "✅ No alerts triggered - all metrics within acceptable ranges"
        
        summary = f"\n{'='*60}\n⚠️  ALERT SUMMARY: {len(self.alerts)} Alert(s) Triggered\n{'='*60}\n"
        
        critical = [a for a in self.alerts if a.severity == 'critical']
        warnings = [a for a in self.alerts if a.severity == 'warning']
        
        if critical:
            summary += f"\n🚨 CRITICAL ({len(critical)}):\n"
            for alert in critical:
                summary += f"  - {alert}\n"
        
        if warnings:
            summary += f"\n⚠️  WARNING ({len(warnings)}):\n"
            for alert in warnings:
                summary += f"  - {alert}\n"
        
        summary += f"\n{'='*60}\n"
        return summary
    
    def export_alerts(self, filepath="alerts.json"):
        """Export alerts to JSON file"""
        with open(filepath, 'w') as f:
            json.dump([a.to_dict() for a in self.alerts], f, indent=2)
        logger.info(f"Alerts exported to {filepath}")

# --- Load Data (Reusing from Part 1) ---
def load_data(log_path, stats_path):
    df = pd.read_csv(log_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    with open(stats_path, "r") as f:
        training_stats = json.load(f)
    return df, training_stats

def filter_to_window(df, days=7):
    # Use the most recent timestamp in the data as reference
    max_timestamp = df["timestamp"].max()
    cutoff_date = max_timestamp - timedelta(days=days)
    recent_window = df[df["timestamp"] >= cutoff_date]
    return recent_window

# --- 1. Compute a Simple Drift Score (PSI) ---
def calculate_psi(expected, actual, buckets=10):
    """Calculate the Population Stability Index (PSI) for a single feature."""
    # Define bins based on the expected (training) distribution
    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    breakpoints = np.percentile(expected, breakpoints)
    
    # Bin the data
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
    
    # Replace 0s to avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # Calculate PSI
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    return np.sum(psi_values)

# --- 2. Compute Performance Metrics ---
def compute_performance_metrics(df):
    # For simplicity, we assume ground_truth is available in logs
    # We also assume a binary classification task where prediction > 25 is the positive class
    y_true = (df["ground_truth"] > 25).astype(int)
    y_pred = (df["prediction"] > 25).astype(int)
    y_score = df["prediction"] # Use raw prediction for AUC
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
    }
    return metrics

# --- 3. Build the Monitoring Summary Table ---
def build_summary_table(system_metrics, data_metrics, drift_scores, perf_metrics, training_stats):
    summary = {
        "Metric": [],
        "Value": [],
        "Notes": []
    }
    
    # System Metrics
    summary["Metric"].append("Requests (7d)")
    summary["Value"].append(system_metrics["Total Requests (last 7 days)"])
    summary["Notes"].append("")
    summary["Metric"].append("p95 Latency (ms)")
    summary["Value"].append(system_metrics["p95 Latency (ms)"])
    summary["Notes"].append("")
    
    # Data Metrics
    feature1_stats = data_metrics[data_metrics["feature"] == "feature_1"].iloc[0]
    summary["Metric"].append("Feature1 Mean (Train vs Prod)")
    train_mean_val = feature1_stats["train_mean"]
    prod_mean_val = feature1_stats["prod_mean"]
    diff_val = feature1_stats["diff"]
    summary["Value"].append(f"{train_mean_val} / {prod_mean_val}")
    summary["Notes"].append(f"Δ = {diff_val}")
    
    # Drift Metrics
    psi_score = drift_scores["feature_1_psi"]
    summary["Metric"].append("Feature1 PSI")
    summary["Value"].append(f"{psi_score:.4f}")
    summary["Notes"].append("Flag if > 0.2")
    
    # Performance Metrics
    summary["Metric"].append("AUC (recent)")
    auc_val = perf_metrics["auc"]
    baseline_auc = training_stats["model_performance"]["auc"]
    summary["Value"].append(f"{auc_val:.4f}")
    summary["Notes"].append(f"Baseline AUC = {baseline_auc}")
    summary["Metric"].append("Recall (recent)")
    recall_val = perf_metrics["recall"]
    summary["Value"].append(f"{recall_val:.4f}")
    summary["Notes"].append("")
    
    return pd.DataFrame(summary)

# --- Main Execution ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LAB PART 2: Drift Detection & Automated Alerting")
    print("="*60)
    
    print("\n📚 LEARNING OBJECTIVE:")
    print("   Move from basic statistics to ACTIONABLE monitoring. We'll calculate")
    print("   a quantitative drift score called PSI, measure live model performance,")
    print("   and most importantly, implement AUTOMATED ALERTING.")
    
    print("\n📊 WHAT WE'RE SIMULATING:")
    print("   We're using the same production logs from Part 1, but now we'll compare")
    print("   them against simulated training data to calculate drift. For the PSI")
    print("   calculation, we generate a synthetic training distribution that matches")
    print("   our baseline statistics. In a real system, you'd save your actual")
    print("   training data distribution when you train the model, then load it here")
    print("   for comparison.")
    
    print("\n🎯 THIS IS WHERE MONITORING BECOMES OPERATIONAL:")
    print("   We're not just calculating metrics - we're triggering alerts when things")
    print("   go wrong. This separates monitoring from manual dashboard checking.")
    
    print("\n📊 DATA & CONFIG:")
    print("   - data/logs.csv: Same production logs from Part 1")
    print("   - data/training_stats.json: Baseline statistics")
    print("   - configs/alert_config.yaml: Alert thresholds (you can customize!)")
    
    print("\n💡 PSI (POPULATION STABILITY INDEX):")
    print("   This is a single number that quantifies how much a feature has drifted")
    print("   from its training distribution. Rule of thumb:")
    print("   - PSI < 0.1: No significant shift ✅")
    print("   - PSI 0.1-0.2: Minor shift - investigate ⚠️")
    print("   - PSI > 0.2: Major shift - likely need to retrain ❌")
    
    print("\n💡 WHAT YOU'LL SEE:")
    print("   1. PSI scores for each feature")
    print("   2. Model performance metrics (AUC, precision, recall)")
    print("   3. Triggered alerts with severity levels (critical/warning)")
    print("   4. Alert export to data/alerts.json")
    
    print("\n⚙️  TIP: Edit configs/alert_config.yaml to change thresholds!")
    print("   Configuration is separated from code - ops teams can adjust without")
    print("   touching Python. This is a best practice for production systems.")
    print("="*60 + "\n")
    
    logger.info("="*60)
    logger.info("Starting ML Monitoring - Drift Detection & Alerting")
    logger.info("="*60)
    
    LOG_PATH = "data/logs.csv"
    STATS_PATH = "data/training_stats.json"
    CONFIG_PATH = "configs/alert_config.yaml"
    
    # Initialize Alert Manager
    alert_manager = AlertManager(CONFIG_PATH)
    logger.info(f"Loaded alert configuration from {CONFIG_PATH}")
    
    logs_df, training_stats = load_data(LOG_PATH, STATS_PATH)
    recent_logs = filter_to_window(logs_df, days=7)
    
    # --- Part 1 Metrics (re-calculated for the summary) ---
    from lab_part_1 import compute_system_metrics, compute_data_metrics
    system_metrics = compute_system_metrics(recent_logs)
    data_metrics_df = compute_data_metrics(recent_logs, training_stats)
    
    # Evaluate system metrics for alerts
    logger.info("Evaluating system metrics against thresholds")
    alert_manager.evaluate_system_metrics(system_metrics)
    
    # --- Part 2 Metrics ---
    # For PSI, we need a "training" distribution. We'll simulate it here.
    # In a real scenario, you would load this from your training dataset.
    np.random.seed(42)
    training_feature_1 = np.random.normal(loc=10.0, scale=2.5, size=1000)
    training_feature_2 = np.random.normal(loc=1.5, scale=0.8, size=1000)
    
    feature_1_psi = calculate_psi(training_feature_1, recent_logs["feature_1"])
    feature_2_psi = calculate_psi(training_feature_2, recent_logs["feature_2"])
    drift_scores = {
        "feature_1_psi": feature_1_psi,
        "feature_2_psi": feature_2_psi
    }
    
    # Evaluate drift for alerts
    logger.info("Evaluating data drift metrics")
    alert_manager.evaluate_drift(drift_scores)
    
    perf_metrics = compute_performance_metrics(recent_logs)
    
    # Evaluate model performance for alerts
    logger.info("Evaluating model performance metrics")
    alert_manager.evaluate_model_performance(perf_metrics, training_stats)
    
    # --- Build and Display Summary ---
    summary_df = build_summary_table(system_metrics, data_metrics_df, drift_scores, perf_metrics, training_stats)
    
    print("\n--- Monitoring Summary Table ---")
    print(summary_df.to_string(index=False))
    
    # Display alerts
    print(alert_manager.get_alerts_summary())
    
    # Export alerts to file
    if alert_manager.alerts:
        alert_manager.export_alerts("data/alerts.json")
    
    print("\n--- Drift Scores (PSI) ---")
    for feature, psi in drift_scores.items():
        status = "✅" if psi < 0.1 else ("⚠️" if psi < 0.2 else "❌")
        print(f"{feature}: {psi:.4f} {status}")
    
    print("\n--- Reflection Prompt ---")
    print("Given this table and the alerts, would you be worried about:")
    print("  - System performance?")
    print("  - Data drift?")
    print("  - Model performance or fairness?")
    print("Write 3–4 bullet points of interpretation.")
    print("  - Model performance or fairness?")
    print("Write 3–4 bullet points of interpretation.")
