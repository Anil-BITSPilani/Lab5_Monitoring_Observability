"""
Lab Part 3: Observability Stack - Logs, Metrics, Traces & Dashboards
This script demonstrates building a complete observability stack for ML monitoring.
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configure Structured JSON Logging ---
class JsonFormatter(logging.Formatter):
    """Custom formatter to output logs in JSON format"""
    def format(self, record):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'metric_name'):
            log_data['metric_name'] = record.metric_name
        if hasattr(record, 'metric_value'):
            log_data['metric_value'] = record.metric_value
        if hasattr(record, 'trace_id'):
            log_data['trace_id'] = record.trace_id
            
        return json.dumps(log_data)

# Setup structured logging
json_handler = logging.FileHandler('observability_logs.json')
json_handler.setFormatter(JsonFormatter())

logger = logging.getLogger('ml_observability')
logger.setLevel(logging.INFO)
logger.addHandler(json_handler)
logger.addHandler(logging.StreamHandler())

# --- Metrics Collector (Prometheus-style) ---
class MetricsCollector:
    """Collects and aggregates metrics over time"""
    def __init__(self):
        self.metrics = defaultdict(list)
        self.timestamps = defaultdict(list)
    
    def record(self, metric_name, value, timestamp=None):
        """Record a metric value"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.metrics[metric_name].append(value)
        self.timestamps[metric_name].append(timestamp)
        
        logger.info(
            f"Recorded metric: {metric_name}={value}",
            extra={'metric_name': metric_name, 'metric_value': value}
        )
    
    def get_timeseries(self, metric_name):
        """Get timeseries data for a metric"""
        return {
            'timestamps': self.timestamps[metric_name],
            'values': self.metrics[metric_name]
        }
    
    def get_stats(self, metric_name):
        """Get statistical summary of a metric"""
        values = self.metrics[metric_name]
        if not values:
            return None
        
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    def export_prometheus_format(self, filepath="metrics.prom"):
        """Export metrics in Prometheus text format"""
        with open(filepath, 'w') as f:
            for metric_name, values in self.metrics.items():
                if values:
                    current_value = values[-1]
                    f.write(f"# HELP {metric_name} ML monitoring metric\n")
                    f.write(f"# TYPE {metric_name} gauge\n")
                    f.write(f"{metric_name} {current_value}\n")
        logger.info(f"Metrics exported to {filepath}")

# --- Trace Collector (OpenTelemetry-style) ---
class Trace:
    """Represents a distributed trace"""
    def __init__(self, trace_id, operation):
        self.trace_id = trace_id
        self.operation = operation
        self.spans = []
        self.start_time = datetime.now()
        self.end_time = None
    
    def add_span(self, span_name, duration_ms, metadata=None):
        """Add a span to the trace"""
        span = {
            'span_name': span_name,
            'duration_ms': duration_ms,
            'metadata': metadata or {}
        }
        self.spans.append(span)
    
    def finish(self):
        """Mark trace as complete"""
        self.end_time = datetime.now()
        total_duration = (self.end_time - self.start_time).total_seconds() * 1000
        
        logger.info(
            f"Trace completed: {self.operation}",
            extra={'trace_id': self.trace_id}
        )
        
        return total_duration
    
    def to_dict(self):
        """Convert trace to dictionary"""
        return {
            'trace_id': self.trace_id,
            'operation': self.operation,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'spans': self.spans
        }

class TraceCollector:
    """Collects and manages traces"""
    def __init__(self):
        self.traces = []
    
    def create_trace(self, operation):
        """Create a new trace"""
        trace_id = f"trace-{len(self.traces)+1:04d}"
        trace = Trace(trace_id, operation)
        self.traces.append(trace)
        return trace
    
    def export_traces(self, filepath="traces.json"):
        """Export all traces to JSON"""
        with open(filepath, 'w') as f:
            json.dump([t.to_dict() for t in self.traces], f, indent=2)
        logger.info(f"Traces exported to {filepath}")

# --- Dashboard Builder ---
class MonitoringDashboard:
    """Creates interactive visualizations for monitoring data"""
    
    def __init__(self, metrics_collector):
        self.metrics = metrics_collector
    
    def create_latency_dashboard(self, output_file="dashboard_latency.html"):
        """Create a latency monitoring dashboard"""
        latency_data = self.metrics.get_timeseries('latency_ms')
        
        if not latency_data['values']:
            print("No latency data available")
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Latency Over Time', 'Latency Distribution',
                          'Latency Percentiles', 'Request Rate'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Latency time series
        fig.add_trace(
            go.Scatter(
                x=latency_data['timestamps'],
                y=latency_data['values'],
                mode='lines+markers',
                name='Latency',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Latency distribution
        fig.add_trace(
            go.Histogram(
                x=latency_data['values'],
                name='Distribution',
                marker=dict(color='green')
            ),
            row=1, col=2
        )
        
        # Percentiles
        stats = self.metrics.get_stats('latency_ms')
        percentiles = ['p50', 'p95', 'p99']
        values = [stats['median'], stats['p95'], stats['p99']]
        
        fig.add_trace(
            go.Bar(
                x=percentiles,
                y=values,
                name='Percentiles',
                marker=dict(color='orange')
            ),
            row=2, col=1
        )
        
        # Request rate (requests per time window)
        # Aggregate requests into time buckets
        timestamps = pd.Series(latency_data['timestamps'])
        request_counts = timestamps.dt.floor('1h').value_counts().sort_index()
        
        fig.add_trace(
            go.Scatter(
                x=request_counts.index,
                y=request_counts.values,
                mode='lines+markers',
                name='Requests/Hour',
                line=dict(color='purple')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="ML Model Latency Dashboard",
            showlegend=True,
            height=800
        )
        
        fig.write_html(output_file)
        logger.info(f"Latency dashboard saved to {output_file}")
        print(f"📊 Dashboard created: {output_file}")
    
    def create_drift_dashboard(self, output_file="dashboard_drift.html"):
        """Create a data drift monitoring dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature 1 Distribution', 'Feature 2 Distribution',
                          'PSI Scores Over Time', 'Model Performance'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Feature 1 distribution
        if 'feature_1' in self.metrics.metrics:
            fig.add_trace(
                go.Histogram(
                    x=self.metrics.metrics['feature_1'],
                    name='Feature 1',
                    marker=dict(color='blue', opacity=0.7)
                ),
                row=1, col=1
            )
        
        # Feature 2 distribution
        if 'feature_2' in self.metrics.metrics:
            fig.add_trace(
                go.Histogram(
                    x=self.metrics.metrics['feature_2'],
                    name='Feature 2',
                    marker=dict(color='green', opacity=0.7)
                ),
                row=1, col=2
            )
        
        # PSI scores
        if 'psi_feature_1' in self.metrics.metrics:
            psi_data = self.metrics.get_timeseries('psi_feature_1')
            fig.add_trace(
                go.Scatter(
                    x=psi_data['timestamps'],
                    y=psi_data['values'],
                    mode='lines+markers',
                    name='PSI Score',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
            # Add threshold lines
            fig.add_hline(y=0.1, line_dash="dash", line_color="orange",
                         annotation_text="Warning", row=2, col=1)
            fig.add_hline(y=0.2, line_dash="dash", line_color="red",
                         annotation_text="Critical", row=2, col=1)
        
        # Model performance (AUC)
        if 'auc' in self.metrics.metrics:
            auc_data = self.metrics.get_timeseries('auc')
            fig.add_trace(
                go.Scatter(
                    x=auc_data['timestamps'],
                    y=auc_data['values'],
                    mode='lines+markers',
                    name='AUC',
                    line=dict(color='purple')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="ML Data Drift & Performance Dashboard",
            showlegend=True,
            height=800
        )
        
        fig.write_html(output_file)
        logger.info(f"Drift dashboard saved to {output_file}")
        print(f"📊 Dashboard created: {output_file}")

# --- Simulate ML Inference with Observability ---
def simulate_ml_inference_with_traces(trace_collector, metrics_collector, num_requests=20):
    """Simulate ML model inference requests with full observability"""
    print("\n" + "="*60)
    print("Simulating ML Inference Requests with Full Observability")
    print("="*60)
    
    for i in range(num_requests):
        # Create a trace for this request
        trace = trace_collector.create_trace(f"ml_prediction_request_{i+1}")
        
        # Simulate preprocessing
        preprocess_time = np.random.uniform(5, 15)
        trace.add_span("preprocess", preprocess_time, {"features_extracted": 3})
        
        # Simulate model inference
        inference_time = np.random.uniform(20, 80)
        trace.add_span("model_inference", inference_time, {"model": "fraud_detector_v2"})
        
        # Simulate postprocessing
        postprocess_time = np.random.uniform(2, 8)
        trace.add_span("postprocess", postprocess_time, {"output_formatted": True})
        
        # Finish trace and record total latency
        total_latency = trace.finish()
        
        # Record metrics
        metrics_collector.record('latency_ms', total_latency)
        metrics_collector.record('feature_1', np.random.normal(12, 3))
        metrics_collector.record('feature_2', np.random.normal(1.8, 0.9))
        
        # Simulate PSI calculation periodically
        if i % 5 == 0:
            psi_score = np.random.uniform(0.05, 0.25)
            metrics_collector.record('psi_feature_1', psi_score)
        
        # Simulate model performance
        if i % 3 == 0:
            auc = np.random.uniform(0.75, 0.93)
            metrics_collector.record('auc', auc)
    
    print(f"✅ Simulated {num_requests} requests with full tracing")

# --- Main Execution ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LAB PART 3: Observability Stack")
    print("Logs, Metrics, Traces & Dashboards")
    print("="*60)
    
    print("\n📚 LEARNING OBJECTIVE:")
    print("   Build a complete observability stack with the 4 pillars:")
    print("   Logs, Metrics, Traces, and Dashboards. This is the foundation")
    print("   for production ML monitoring systems.")
    
    print("\n🔬 SIMULATION:")
    print("   This script will SIMULATE 20 ML inference requests to demonstrate")
    print("   how to collect observability data. In production, this data would")
    print("   come from your actual ML service.")
    
    print("\n💡 WHAT YOU'LL GET:")
    print("   1. LOGS: Structured JSON logs (observability_logs.json)")
    print("   2. METRICS: Prometheus format (data/metrics.prom)")
    print("   3. TRACES: Distributed traces showing latency breakdown (data/traces.json)")
    print("   4. DASHBOARDS: Interactive HTML visualizations (data/dashboard_*.html)")
    
    print("\n🎯 AFTER RUNNING: Open the HTML dashboards in your browser!")
    print("="*60 + "\n")
    
    # Initialize observability components
    metrics_collector = MetricsCollector()
    trace_collector = TraceCollector()
    
    logger.info("Observability stack initialized")
    
    # Simulate ML inference with full observability
    simulate_ml_inference_with_traces(trace_collector, metrics_collector, num_requests=20)
    
    # Display metrics statistics
    print("\n--- Latency Statistics ---")
    latency_stats = metrics_collector.get_stats('latency_ms')
    for stat, value in latency_stats.items():
        print(f"{stat}: {value:.2f}")
    
    # Export metrics in Prometheus format
    print("\n--- Exporting Metrics ---")
    metrics_collector.export_prometheus_format("data/metrics.prom")
    print("✅ Metrics exported to data/metrics.prom")
    
    # Export traces
    print("\n--- Exporting Traces ---")
    trace_collector.export_traces("data/traces.json")
    print("✅ Traces exported to data/traces.json")
    
    # Create interactive dashboards
    print("\n--- Creating Dashboards ---")
    dashboard = MonitoringDashboard(metrics_collector)
    dashboard.create_latency_dashboard("data/dashboard_latency.html")
    dashboard.create_drift_dashboard("data/dashboard_drift.html")
    
    print("\n" + "="*60)
    print("🎯 OBSERVABILITY STACK SUMMARY")
    print("="*60)
    print("✅ Logs: Structured JSON logs → observability_logs.json")
    print("✅ Metrics: Prometheus format → data/metrics.prom")
    print("✅ Traces: Distributed traces → data/traces.json")
    print("✅ Dashboards: Interactive HTML → data/dashboard_*.html")
    print("="*60)
    
    print("\n📝 Key Takeaways:")
    print("1. Logs provide detailed event information for debugging")
    print("2. Metrics enable aggregation and alerting on KPIs")
    print("3. Traces help diagnose latency issues across components")
    print("4. Dashboards visualize trends and patterns over time")
    print()
