"""
FactoryGuard AI - Model Logic Validation & Physical Expectations
Krutika - Week 3, Days 4-5

This script validates model behavior against physical expectations for
predictive maintenance in manufacturing environments.
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime

# ============================================================================
# PHYSICAL EXPECTATIONS DEFINITION
# ============================================================================

PHYSICAL_EXPECTATIONS = {
    "temperature": {
        "expected_direction": "positive",
        "rationale": "Higher temperatures indicate thermal stress, increased wear, and potential component degradation",
        "threshold_concern": "> 80°C typically critical for industrial machinery"
    },
    "temperature_rolling_means": {
        "expected_direction": "positive",
        "rationale": "Sustained high temperatures over time (rolling averages) indicate persistent thermal stress",
        "threshold_concern": "Consistent elevation more concerning than spikes"
    },
    "vibration": {
        "expected_direction": "positive",
        "rationale": "Excessive vibration indicates mechanical imbalance, bearing wear, or misalignment",
        "threshold_concern": "Vibration amplitude increases precede mechanical failures"
    },
    "vibration_rolling_means": {
        "expected_direction": "positive",
        "rationale": "Sustained vibration patterns indicate progressive mechanical degradation",
        "threshold_concern": "Trend more important than absolute values"
    },
    "pressure": {
        "expected_direction": "mixed",
        "rationale": "Both high and low pressure can indicate problems (leaks, blockages, pump issues)",
        "threshold_concern": "Deviations from nominal operating pressure"
    },
    "runtime_hours": {
        "expected_direction": "positive",
        "rationale": "Extended runtime increases cumulative wear and fatigue",
        "threshold_concern": "Approaching maintenance intervals"
    },
    "lag_features": {
        "expected_direction": "positive",
        "rationale": "Recent sensor values (lags) capture immediate pre-failure conditions",
        "threshold_concern": "Short-term trends critical for 24h prediction"
    }
}


def load_shap_analysis():
    """
    Load SHAP values and feature importance from Akshada's work
    
    Returns:
        shap_data: Dictionary containing SHAP values and metadata (or None if unavailable)
        importance_df: Feature importance rankings
    """
    print("=" * 70)
    print("LOADING SHAP ANALYSIS RESULTS")
    print("=" * 70)
    
    # Load feature importance (primary source)
    importance_df = pd.read_csv('outputs/feature_importance_analysis.csv')
    print("\n✓ Feature importance analysis loaded")
    
    # Try to load SHAP values (optional)
    shap_data = None
    if os.path.exists('outputs/shap_values.pkl'):
        try:
            with open('outputs/shap_values.pkl', 'rb') as f:
                shap_data = pickle.load(f)
            print("✓ SHAP values loaded")
            print(f"\nDataset: {shap_data['shap_values'].shape[0]} samples")
            print(f"Features: {len(shap_data['feature_names'])}")
        except Exception as e:
            print(f"⚠ Could not load SHAP pickle file (using CSV only): {str(e)[:50]}")
            shap_data = None
    else:
        print("⚠ SHAP pickle file not found (using CSV only)")
    
    print(f"\nTotal features in analysis: {len(importance_df)}")
    
    return shap_data, importance_df


def categorize_features(feature_names):
    """
    Categorize features by type for structured analysis
    
    Args:
        feature_names: List of feature names
        
    Returns:
        categories: Dictionary mapping category to feature list
    """
    categories = {
        "temperature_base": [],
        "temperature_rolling": [],
        "temperature_lag": [],
        "vibration_base": [],
        "vibration_rolling": [],
        "vibration_lag": [],
        "pressure_base": [],
        "pressure_rolling": [],
        "pressure_lag": [],
        "temporal": [],
        "other": []
    }
    
    for feature in feature_names:
        feature_lower = feature.lower()
        
        if "temperature" in feature_lower:
            if "roll" in feature_lower:
                categories["temperature_rolling"].append(feature)
            elif "lag" in feature_lower:
                categories["temperature_lag"].append(feature)
            else:
                categories["temperature_base"].append(feature)
        elif "vibration" in feature_lower:
            if "roll" in feature_lower:
                categories["vibration_rolling"].append(feature)
            elif "lag" in feature_lower:
                categories["vibration_lag"].append(feature)
            else:
                categories["vibration_base"].append(feature)
        elif "pressure" in feature_lower:
            if "roll" in feature_lower:
                categories["pressure_rolling"].append(feature)
            elif "lag" in feature_lower:
                categories["pressure_lag"].append(feature)
            else:
                categories["pressure_base"].append(feature)
        elif feature_lower in ["hour", "day", "month", "day_of_week"]:
            categories["temporal"].append(feature)
        else:
            categories["other"].append(feature)
    
    return categories


def validate_feature_impact(feature, mean_shap, expected_direction):
    """
    Validate if feature impact matches physical expectations
    
    Args:
        feature: Feature name
        mean_shap: Mean SHAP value for the feature
        expected_direction: Expected direction ("positive", "negative", "mixed")
        
    Returns:
        validation_result: Dictionary with validation status
    """
    actual_direction = "positive" if mean_shap > 0 else "negative"
    
    if expected_direction == "mixed":
        status = "ACCEPTABLE"
        explanation = "Both directions physically plausible"
    elif expected_direction == actual_direction:
        status = "CONSISTENT"
        explanation = f"Impact direction matches expectation ({expected_direction})"
    else:
        status = "INCONSISTENT"
        explanation = f"Expected {expected_direction}, but model shows {actual_direction}"
    
    return {
        "feature": feature,
        "mean_shap": mean_shap,
        "actual_direction": actual_direction,
        "expected_direction": expected_direction,
        "status": status,
        "explanation": explanation
    }


def analyze_physical_consistency(importance_df):
    """
    Analyze SHAP results against physical expectations
    
    Args:
        importance_df: Feature importance DataFrame from Akshada's analysis
        
    Returns:
        validation_results: List of validation results for each feature
        summary_stats: Summary statistics of validation
    """
    print("\n" + "=" * 70)
    print("VALIDATING AGAINST PHYSICAL EXPECTATIONS")
    print("=" * 70)
    
    validation_results = []
    
    for _, row in importance_df.iterrows():
        feature = row['feature']
        mean_shap = row['mean_shap']
        
        # Determine expected direction based on feature type
        if "temperature" in feature.lower() and "roll" in feature.lower():
            expected = "positive"
        elif "temperature" in feature.lower():
            expected = "positive"
        elif "vibration" in feature.lower():
            expected = "positive"
        elif "pressure" in feature.lower():
            expected = "mixed"
        elif feature.lower() in ["hour", "day", "month", "day_of_week"]:
            expected = "mixed"
        else:
            expected = "mixed"
        
        result = validate_feature_impact(feature, mean_shap, expected)
        validation_results.append(result)
    
    # Calculate summary statistics
    total = len(validation_results)
    consistent = sum(1 for r in validation_results if r['status'] == 'CONSISTENT')
    inconsistent = sum(1 for r in validation_results if r['status'] == 'INCONSISTENT')
    acceptable = sum(1 for r in validation_results if r['status'] == 'ACCEPTABLE')
    
    summary_stats = {
        "total_features": total,
        "consistent": consistent,
        "inconsistent": inconsistent,
        "acceptable": acceptable,
        "consistency_rate": (consistent / total) * 100 if total > 0 else 0
    }
    
    return validation_results, summary_stats


def identify_anomalies(validation_results, importance_df):
    """
    Identify counterintuitive or anomalous model behaviors
    
    Args:
        validation_results: List of validation results
        importance_df: Feature importance DataFrame
        
    Returns:
        anomalies: List of identified anomalies with explanations
    """
    print("\n" + "=" * 70)
    print("IDENTIFYING ANOMALIES AND RED FLAGS")
    print("=" * 70)
    
    anomalies = []
    
    # Check for inconsistent high-importance features
    for result in validation_results:
        if result['status'] == 'INCONSISTENT':
            feature_importance = importance_df[
                importance_df['feature'] == result['feature']
            ]['mean_abs_shap'].values[0]
            
            if feature_importance > 0.5:  # High importance threshold
                anomalies.append({
                    "type": "HIGH_IMPORTANCE_INCONSISTENCY",
                    "feature": result['feature'],
                    "severity": "HIGH",
                    "description": f"Important feature shows unexpected direction",
                    "mean_shap": result['mean_shap'],
                    "expected": result['expected_direction'],
                    "actual": result['actual_direction'],
                    "recommendation": "Investigate data quality or feature engineering"
                })
    
    # Check for temperature rolling mean anomalies
    temp_rolling = importance_df[
        importance_df['feature'].str.contains('temperature_roll', case=False, na=False)
    ]
    
    for _, row in temp_rolling.iterrows():
        if row['mean_shap'] < 0:  # Negative impact unexpected
            anomalies.append({
                "type": "TEMPERATURE_ROLLING_NEGATIVE",
                "feature": row['feature'],
                "severity": "MEDIUM",
                "description": "Temperature rolling mean shows negative impact on failure",
                "mean_shap": row['mean_shap'],
                "expected": "positive",
                "actual": "negative",
                "recommendation": "Check for data normalization issues or multicollinearity"
            })
    
    # Check for vibration anomalies
    vibration_features = importance_df[
        importance_df['feature'].str.contains('vibration', case=False, na=False)
    ]
    
    for _, row in vibration_features.iterrows():
        if row['mean_shap'] < 0:
            anomalies.append({
                "type": "VIBRATION_NEGATIVE",
                "feature": row['feature'],
                "severity": "MEDIUM",
                "description": "Vibration shows negative impact (unexpected)",
                "mean_shap": row['mean_shap'],
                "expected": "positive",
                "actual": "negative",
                "recommendation": "Verify sensor calibration and data preprocessing"
            })
    
    print(f"\n✓ Identified {len(anomalies)} anomalies")
    
    return anomalies


def generate_recommendations(validation_results, anomalies, summary_stats):
    """
    Generate recommendations for model improvement
    
    Args:
        validation_results: Validation results
        anomalies: Identified anomalies
        summary_stats: Summary statistics
        
    Returns:
        recommendations: List of actionable recommendations
    """
    recommendations = []
    
    # Based on consistency rate
    if summary_stats['consistency_rate'] < 70:
        recommendations.append({
            "priority": "HIGH",
            "category": "Model Architecture",
            "recommendation": "Consider feature engineering review - low consistency with physical expectations",
            "rationale": f"Only {summary_stats['consistency_rate']:.1f}% of features show expected behavior"
        })
    
    # Based on anomalies
    if len(anomalies) > 0:
        recommendations.append({
            "priority": "HIGH",
            "category": "Data Quality",
            "recommendation": "Investigate data quality for features showing counterintuitive behavior",
            "rationale": f"{len(anomalies)} anomalies detected in feature impacts"
        })
    
    # Temperature-specific
    temp_inconsistent = [r for r in validation_results 
                        if 'temperature' in r['feature'].lower() 
                        and r['status'] == 'INCONSISTENT']
    if len(temp_inconsistent) > 2:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "Feature Engineering",
            "recommendation": "Review temperature feature engineering and scaling",
            "rationale": "Multiple temperature features show unexpected impact direction"
        })
    
    # General recommendations
    recommendations.append({
        "priority": "MEDIUM",
        "category": "Model Interpretability",
        "recommendation": "Consider domain expert review of feature impacts",
        "rationale": "Validate model logic with maintenance engineers"
    })
    
    recommendations.append({
        "priority": "LOW",
        "category": "Model Enhancement",
        "recommendation": "Explore non-linear models to capture complex physical relationships",
        "rationale": "Linear model may not capture all physical dynamics"
    })
    
    return recommendations


def save_validation_results(validation_results, anomalies, recommendations, summary_stats):
    """
    Save validation results to JSON for programmatic access
    """
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "summary_statistics": summary_stats,
        "validation_results": validation_results,
        "anomalies": anomalies,
        "recommendations": recommendations
    }
    
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/validation_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n✓ Validation results saved to outputs/validation_results.json")


def main():
    """
    Main execution function for Krutika's validation task
    """
    print("\n" + "=" * 70)
    print("KRUTIKA'S MODEL LOGIC VALIDATION & PHYSICAL EXPECTATIONS")
    print("FactoryGuard AI - Week 3")
    print("=" * 70 + "\n")
    
    # Step 1: Load SHAP analysis
    shap_data, importance_df = load_shap_analysis()
    
    # Step 2: Categorize features
    feature_names = shap_data['feature_names'] if shap_data else importance_df['feature'].tolist()
    categories = categorize_features(feature_names)
    
    # Step 3: Validate against physical expectations
    validation_results, summary_stats = analyze_physical_consistency(importance_df)
    
    # Step 4: Identify anomalies
    anomalies = identify_anomalies(validation_results, importance_df)
    
    # Step 5: Generate recommendations
    recommendations = generate_recommendations(validation_results, anomalies, summary_stats)
    
    # Step 6: Save results
    save_validation_results(validation_results, anomalies, recommendations, summary_stats)
    
    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\nTotal Features Analyzed: {summary_stats['total_features']}")
    print(f"Consistent with Expectations: {summary_stats['consistent']} ({summary_stats['consistency_rate']:.1f}%)")
    print(f"Inconsistent: {summary_stats['inconsistent']}")
    print(f"Acceptable (Mixed): {summary_stats['acceptable']}")
    print(f"\nAnomalies Detected: {len(anomalies)}")
    print(f"Recommendations Generated: {len(recommendations)}")
    
    print("\n" + "=" * 70)
    print("SUCCESS: VALIDATION ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nNext Step: Review outputs/physical_validation_report.md")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
