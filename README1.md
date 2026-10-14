# M5 Monitoring Lab: Observability & Drift Detection

This repository contains comprehensive lab materials for Module 5, covering all aspects of ML monitoring including:
- **Why ML Monitoring is Different** from traditional monitoring
- **System, Data & Model Metrics** tracking and analysis  
- **Drift Detection, Data Quality & Alerting** with automated thresholds
- **Observability Stack** with logs, metrics, traces and interactive dashboards

## Repository Structure

```
Lab_5/
├── data/
│   ├── logs.csv                    # Sample production logs
│   ├── training_stats.json         # Training data statistics for comparison
│   ├── alerts.json                 # Generated alerts (created by Part 2)
│   ├── metrics.prom                # Prometheus-format metrics (created by Part 3)
│   ├── traces.json                 # Distributed traces (created by Part 3)
│   ├── dashboard_latency.html      # Interactive latency dashboard (created by Part 3)
│   └── dashboard_drift.html        # Interactive drift dashboard (created by Part 3)
├── configs/
│   └── alert_config.yaml           # Alert threshold configuration
├── scripts/
│   ├── lab_part_0.py               # Why ML Monitoring is Different
│   ├── lab_part_1.py               # System & Data Metrics (with structured logging)
│   ├── lab_part_2.py               # Drift Detection & Alerting
│   └── lab_part_3.py               # Observability Stack (Logs, Metrics, Traces, Dashboards)
├── m5_lab_details.md               # Complete trainer's script and lab guide
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Getting Started

### 1. Install Dependencies

```bash
cd Lab_5
pip install -r requirements.txt
```

Or using uv:
```bash
uv pip install -r requirements.txt
```

### 2. Run the Labs in Sequence

#### Part 0: Why ML Monitoring is Different (~10-15 min)
Understand ML-specific monitoring challenges through concrete scenarios.

```bash
python scripts/lab_part_0.py
```

**Topics Covered:**
- Silent model degradation
- Data distribution shift (covariate drift)
- Feedback loop degradation
- Label shift (target drift)
- Traditional vs ML monitoring comparison

---

#### Part 1: System & Data Metrics (~10-15 min)
Learn to analyze production logs and compute core monitoring metrics.

```bash
python scripts/lab_part_1.py
```

**Topics Covered:**
- Loading and filtering production logs
- System metrics: latency (mean, p95, p99), error rate
- Data metrics: feature statistics, distribution comparison
- Structured logging with Python logging module
- Data quality checks

**Outputs:**
- System metrics summary
- Data drift comparison table
- Structured logs in `monitoring.log`

---

#### Part 2: Drift & Alerting (~15-20 min)
Implement drift detection and automated alerting.

```bash
python scripts/lab_part_2.py
```

**Topics Covered:**
- Population Stability Index (PSI) calculation
- Model performance metrics (accuracy, precision, recall, AUC)
- Threshold-based alerting
- Alert severity levels (critical, warning, info)
- Automated alert triggering and routing

**Outputs:**
- Monitoring summary table
- PSI drift scores
- Triggered alerts with severity
- Alert export to `data/alerts.json`

---

#### Part 3: Observability Stack (~15-20 min)
Build a complete observability stack with logs, metrics, traces, and dashboards.

```bash
python scripts/lab_part_3.py
```

**Topics Covered:**
- Structured JSON logging for ML events
- Metrics collection (Prometheus-style)
- Distributed tracing (OpenTelemetry-style)
- Interactive dashboard creation with Plotly
- End-to-end request observability

**Outputs:**
- `observability_logs.json` - Structured logs
- `data/metrics.prom` - Prometheus-format metrics
- `data/traces.json` - Distributed traces
- `data/dashboard_latency.html` - Interactive latency dashboard
- `data/dashboard_drift.html` - Interactive drift dashboard

---

## Lab Guide

For the complete trainer's script with detailed narration, timing, and teaching notes, see [m5_lab_details.md](m5_lab_details.md).

## Learning Objectives

By the end of this lab, students will be able to:

1. **Explain** why ML systems require specialized monitoring beyond traditional system metrics
2. **Calculate** and interpret system metrics (latency percentiles, error rates)
3. **Detect** data drift using statistical measures like PSI
4. **Implement** automated alerting with configurable thresholds
5. **Build** a complete observability stack with logs, metrics, and traces
6. **Create** interactive dashboards for monitoring visualization
7. **Design** comprehensive monitoring strategies for production ML systems

## Configuration

Edit `configs/alert_config.yaml` to customize alert thresholds:

```yaml
system_metrics:
  error_rate:
    threshold: 1.0  # Alert if error rate > 1%
    severity: "critical"
  
data_drift:
  psi:
    threshold: 0.2  # Alert if PSI > 0.2 (major drift)
    severity: "critical"

model_performance:
  auc:
    min_threshold: 0.75  # Alert if AUC < 0.75
    severity: "critical"
```

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'plotly'`  
**Solution:** Run `pip install -r requirements.txt`

**Issue:** No data in dashboards  
**Solution:** Ensure you've run Part 3 which generates the simulated data

**Issue:** Alert thresholds not triggering  
**Solution:** Check `configs/alert_config.yaml` and adjust threshold values

## Next Steps

After completing this lab:
- Integrate monitoring into your ML pipelines
- Set up real alerting channels (Slack, PagerDuty, email)
- Deploy dashboards to monitoring platforms (Grafana, Datadog)
- Implement continuous drift monitoring in production

## References

- [Evidently AI - ML Monitoring](https://www.evidentlyai.com/)
- [Prometheus Monitoring](https://prometheus.io/)
- [OpenTelemetry](https://opentelemetry.io/)
- [PSI Calculation](https://mwburke.github.io/data%20science/2018/04/29/population-stability-index.html)
