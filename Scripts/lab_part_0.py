"""
Lab Part 0: Why ML Monitoring is Different
This script demonstrates scenarios where traditional monitoring fails to detect ML-specific issues.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Scenario 1: Silent Model Degradation ---
def scenario_1_silent_degradation():
    """
    Traditional metrics (latency, error rate) look healthy,
    but model accuracy is degrading due to concept drift.
    """
    print("=" * 60)
    print("SCENARIO 1: Silent Model Degradation")
    print("=" * 60)
    
    # Week 1: Healthy system and model
    week1_requests = 10000
    week1_latency = 45  # ms
    week1_error_rate = 0.1  # %
    week1_accuracy = 0.92
    
    # Week 4: System still healthy, but model degraded
    week4_requests = 10000
    week4_latency = 43  # ms (even better!)
    week4_error_rate = 0.1  # %
    week4_accuracy = 0.67  # DEGRADED!
    
    print("\n📊 Traditional System Metrics:")
    print(f"{'Metric':<20} {'Week 1':<15} {'Week 4':<15} {'Status'}")
    print("-" * 60)
    print(f"{'Requests':<20} {week1_requests:<15} {week4_requests:<15} ✅ Stable")
    print(f"{'Avg Latency (ms)':<20} {week1_latency:<15} {week4_latency:<15} ✅ Stable")
    print(f"{'Error Rate (%)':<20} {week1_error_rate:<15} {week4_error_rate:<15} ✅ Stable")
    
    print("\n🤖 ML-Specific Metrics:")
    print(f"{'Metric':<20} {'Week 1':<15} {'Week 4':<15} {'Status'}")
    print("-" * 60)
    print(f"{'Model Accuracy':<20} {week1_accuracy:<15} {week4_accuracy:<15} ❌ DEGRADED!")
    
    print("\n💡 Key Insight:")
    print("   Traditional monitoring shows everything is fine, but the model")
    print("   is making increasingly poor predictions. This is a SILENT FAILURE.")
    print("   Users are getting fast, error-free responses... that are WRONG.")
    

# --- Scenario 2: Data Distribution Shift ---
def scenario_2_data_shift():
    """
    Input data distribution changes (e.g., new user demographics),
    causing model performance to drop even though code is unchanged.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 2: Data Distribution Shift (Covariate Drift)")
    print("=" * 60)
    
    # Training: Users from Region A
    np.random.seed(42)
    train_age = np.random.normal(35, 10, 1000)  # Mean age: 35
    train_income = np.random.normal(60000, 15000, 1000)  # Mean income: $60k
    
    # Production Week 1: Similar to training
    prod_week1_age = np.random.normal(36, 10, 500)
    prod_week1_income = np.random.normal(62000, 15000, 500)
    
    # Production Week 4: Expansion to Region B (younger, lower income)
    prod_week4_age = np.random.normal(25, 8, 500)  # Younger users!
    prod_week4_income = np.random.normal(35000, 10000, 500)  # Lower income!
    
    print("\n📊 Feature Distribution Comparison:")
    print(f"{'Dataset':<20} {'Mean Age':<15} {'Mean Income':<15}")
    print("-" * 60)
    print(f"{'Training Data':<20} {train_age.mean():<15.1f} ${train_income.mean():<14,.0f}")
    print(f"{'Production Week 1':<20} {prod_week1_age.mean():<15.1f} ${prod_week1_income.mean():<14,.0f}")
    print(f"{'Production Week 4':<20} {prod_week4_age.mean():<15.1f} ${prod_week4_income.mean():<14,.0f}")
    
    print("\n💡 Key Insight:")
    print("   The model was trained on one user population but is now being")
    print("   used on a DIFFERENT population. Traditional monitoring can't")
    print("   detect this - you need to track FEATURE DISTRIBUTIONS.")
    

# --- Scenario 3: Feedback Loop Degradation ---
def scenario_3_feedback_loop():
    """
    Model predictions influence user behavior, which changes the data,
    which degrades the model further (a vicious cycle).
    """
    print("\n" + "=" * 60)
    print("SCENARIO 3: Feedback Loop Degradation")
    print("=" * 60)
    
    print("\n📖 Story: A recommendation system")
    print("-" * 60)
    print("Week 1: Model recommends diverse content")
    print("        → Users engage with variety of topics")
    print("        → Training data remains diverse")
    print("        → Model performance: GOOD ✅")
    
    print("\nWeek 4: Model starts over-recommending popular items")
    print("        → Users only see popular content")
    print("        → Niche content gets no exposure")
    print("        → Training data becomes BIASED toward popular items")
    print("        → Model reinforces the bias further")
    print("        → Model performance: DEGRADED ❌")
    
    # Simulate recommendation diversity
    week1_diversity = 0.85  # 85% of catalog getting recommended
    week4_diversity = 0.42  # Only 42% of catalog now recommended
    
    print("\n📊 Recommendation Diversity Metric:")
    print(f"{'Metric':<30} {'Week 1':<15} {'Week 4':<15}")
    print("-" * 60)
    print(f"{'Catalog Coverage (%)':<30} {week1_diversity*100:<15.0f} {week4_diversity*100:<15.0f}")
    
    print("\n💡 Key Insight:")
    print("   ML models can create FEEDBACK LOOPS where predictions influence")
    print("   future data. Traditional monitoring won't catch this - you need")
    print("   to track diversity, fairness, and distribution shifts over time.")
    

# --- Scenario 4: Label Shift (Target Drift) ---
def scenario_4_label_shift():
    """
    The distribution of the target variable changes, even though features are similar.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 4: Label Shift (Target Drift)")
    print("=" * 60)
    
    print("\n📖 Story: Fraud detection model")
    print("-" * 60)
    
    # Training: 1% fraud rate
    train_fraud_rate = 0.01
    train_legit_rate = 0.99
    
    # Production Week 4: 5% fraud rate (new attack pattern)
    prod_fraud_rate = 0.05
    prod_legit_rate = 0.95
    
    print("\n📊 Class Distribution:")
    print(f"{'Dataset':<20} {'Fraud %':<15} {'Legitimate %':<15}")
    print("-" * 60)
    print(f"{'Training':<20} {train_fraud_rate*100:<15.1f} {train_legit_rate*100:<15.1f}")
    print(f"{'Production Week 4':<20} {prod_fraud_rate*100:<15.1f} {prod_legit_rate*100:<15.1f}")
    
    # Model trained on 1% fraud will have poor precision at 5% fraud rate
    train_precision = 0.80
    prod_precision = 0.45  # Degrades significantly
    
    print("\n📊 Model Performance:")
    print(f"{'Dataset':<20} {'Precision':<15}")
    print("-" * 60)
    print(f"{'Training':<20} {train_precision:<15.2f}")
    print(f"{'Production Week 4':<20} {prod_precision:<15.2f}")
    
    print("\n💡 Key Insight:")
    print("   The TARGET distribution has shifted. The model was calibrated")
    print("   for 1% fraud but now faces 5% fraud. Traditional monitoring")
    print("   can't detect this - you need to track PREDICTION distributions")
    print("   and compare them to training baselines.")
    

# --- Summary Comparison ---
def print_summary():
    """
    Compare what traditional vs ML monitoring can detect.
    """
    print("\n" + "=" * 60)
    print("SUMMARY: Traditional vs ML Monitoring")
    print("=" * 60)
    
    scenarios = [
        ("Silent Model Degradation", "❌ Cannot detect", "✅ Tracks accuracy/AUC"),
        ("Data Distribution Shift", "❌ Cannot detect", "✅ Tracks feature drift (PSI)"),
        ("Feedback Loop Degradation", "❌ Cannot detect", "✅ Tracks diversity/fairness"),
        ("Label Shift", "❌ Cannot detect", "✅ Tracks prediction distribution"),
        ("System Errors (500s)", "✅ Can detect", "✅ Can also detect"),
        ("High Latency", "✅ Can detect", "✅ Can also detect"),
    ]
    
    print(f"\n{'Scenario':<30} {'Traditional':<25} {'ML Monitoring'}")
    print("-" * 80)
    for scenario, trad, ml in scenarios:
        print(f"{scenario:<30} {trad:<25} {ml}")
    
    print("\n" + "=" * 60)
    print("🎯 CONCLUSION")
    print("=" * 60)
    print("Traditional system monitoring is NECESSARY but NOT SUFFICIENT.")
    print("You need ML-specific monitoring to track:")
    print("  1. Model performance (accuracy, AUC, precision, recall)")
    print("  2. Data drift (feature distributions, PSI, KL divergence)")
    print("  3. Prediction drift (target distribution changes)")
    print("  4. Fairness & diversity metrics")
    print("=" * 60)


# --- Main Execution ---
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LAB PART 0: Why ML Monitoring is Different")
    print("Exploring scenarios where traditional monitoring fails")
    print("=" * 60)    
    print("\n📚 LEARNING OBJECTIVE:")
    print("   Understand why traditional system monitoring (latency, errors) is")
    print("   NOT SUFFICIENT for ML systems. We'll explore 4 simulated scenarios")
    print("   where everything looks healthy but the ML model is actually failing.")
    print("\n💡 NOTE: All scenarios below are SIMULATED for educational purposes.")
    print("   They demonstrate real-world ML monitoring challenges.")
    print("\n   Press Enter to continue...")
    input()    
    scenario_1_silent_degradation()
    scenario_2_data_shift()
    scenario_3_feedback_loop()
    scenario_4_label_shift()
    print_summary()
    
    print("\n📝 Reflection Questions:")
    print("1. Can you think of other ML-specific failure modes?")
    print("2. Which metrics would you track for YOUR ML model?")
    print("3. How would you automate detection of these issues?")
    print()
