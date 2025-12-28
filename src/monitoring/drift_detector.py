"""
Data Drift Detection for SageMaker Model Monitor

This script detects drift in input data distributions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data drift in model inputs."""

    def __init__(
        self,
        baseline_stats_path: str,
        current_stats_path: str,
        threshold: float = 0.05,
    ):
        """
        Initialize drift detector.

        Args:
            baseline_stats_path: Path to baseline statistics
            current_stats_path: Path to current statistics
            threshold: P-value threshold for drift detection
        """
        self.baseline_stats = self._load_stats(baseline_stats_path)
        self.current_stats = self._load_stats(current_stats_path)
        self.threshold = threshold

    def _load_stats(self, path: str) -> Dict[str, Any]:
        """Load statistics from JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    def detect_numerical_drift(
        self, feature_name: str
    ) -> Dict[str, Any]:
        """
        Detect drift in numerical features using KS test.

        Args:
            feature_name: Name of the feature

        Returns:
            Drift detection result
        """
        baseline_values = self.baseline_stats.get(feature_name, {}).get("values", [])
        current_values = self.current_stats.get(feature_name, {}).get("values", [])

        if not baseline_values or not current_values:
            return {
                "feature": feature_name,
                "drift_detected": False,
                "reason": "Insufficient data",
            }

        # Kolmogorov-Smirnov test
        statistic, p_value = ks_2samp(baseline_values, current_values)

        drift_detected = p_value < self.threshold

        return {
            "feature": feature_name,
            "drift_detected": drift_detected,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "threshold": self.threshold,
            "baseline_mean": float(np.mean(baseline_values)),
            "current_mean": float(np.mean(current_values)),
            "baseline_std": float(np.std(baseline_values)),
            "current_std": float(np.std(current_values)),
        }

    def detect_categorical_drift(
        self, feature_name: str
    ) -> Dict[str, Any]:
        """
        Detect drift in categorical features using chi-square test.

        Args:
            feature_name: Name of the feature

        Returns:
            Drift detection result
        """
        baseline_dist = self.baseline_stats.get(feature_name, {}).get("distribution", {})
        current_dist = self.current_stats.get(feature_name, {}).get("distribution", {})

        if not baseline_dist or not current_dist:
            return {
                "feature": feature_name,
                "drift_detected": False,
                "reason": "Insufficient data",
            }

        # Align categories
        all_categories = set(baseline_dist.keys()) | set(current_dist.keys())
        baseline_counts = [baseline_dist.get(cat, 0) for cat in all_categories]
        current_counts = [current_dist.get(cat, 0) for cat in all_categories]

        # Chi-square test
        contingency_table = np.array([baseline_counts, current_counts])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        drift_detected = p_value < self.threshold

        return {
            "feature": feature_name,
            "drift_detected": drift_detected,
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "threshold": self.threshold,
            "baseline_distribution": baseline_dist,
            "current_distribution": current_dist,
        }

    def detect_all_drift(self) -> Dict[str, Any]:
        """
        Detect drift across all features.

        Returns:
            Complete drift report
        """
        results = {
            "drift_detected": False,
            "features_with_drift": [],
            "feature_reports": {},
        }

        # Check all features in baseline
        for feature_name, feature_stats in self.baseline_stats.items():
            feature_type = feature_stats.get("type", "numerical")

            if feature_type == "numerical":
                drift_result = self.detect_numerical_drift(feature_name)
            else:
                drift_result = self.detect_categorical_drift(feature_name)

            results["feature_reports"][feature_name] = drift_result

            if drift_result.get("drift_detected", False):
                results["drift_detected"] = True
                results["features_with_drift"].append(feature_name)

        return results


class ModelQualityMonitor:
    """Monitor model quality metrics."""

    def __init__(
        self,
        baseline_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        threshold_degradation: float = 0.05,
    ):
        """
        Initialize model quality monitor.

        Args:
            baseline_metrics: Baseline model metrics
            current_metrics: Current model metrics
            threshold_degradation: Threshold for metric degradation (e.g., 0.05 = 5%)
        """
        self.baseline_metrics = baseline_metrics
        self.current_metrics = current_metrics
        self.threshold_degradation = threshold_degradation

    def check_metric_degradation(self, metric_name: str) -> Dict[str, Any]:
        """
        Check if a specific metric has degraded.

        Args:
            metric_name: Name of the metric to check

        Returns:
            Degradation check result
        """
        baseline_value = self.baseline_metrics.get(metric_name)
        current_value = self.current_metrics.get(metric_name)

        if baseline_value is None or current_value is None:
            return {
                "metric": metric_name,
                "degraded": False,
                "reason": "Metric not available",
            }

        # Calculate relative change
        relative_change = (current_value - baseline_value) / baseline_value

        degraded = relative_change < -self.threshold_degradation

        return {
            "metric": metric_name,
            "degraded": degraded,
            "baseline_value": float(baseline_value),
            "current_value": float(current_value),
            "relative_change": float(relative_change),
            "threshold": self.threshold_degradation,
        }

    def check_all_metrics(self) -> Dict[str, Any]:
        """
        Check all metrics for degradation.

        Returns:
            Complete quality check report
        """
        results = {
            "quality_degraded": False,
            "degraded_metrics": [],
            "metric_reports": {},
        }

        for metric_name in self.baseline_metrics.keys():
            check_result = self.check_metric_degradation(metric_name)
            results["metric_reports"][metric_name] = check_result

            if check_result.get("degraded", False):
                results["quality_degraded"] = True
                results["degraded_metrics"].append(metric_name)

        return results


def trigger_retraining(
    state_machine_arn: str, region: str = "us-east-1"
) -> Dict[str, Any]:
    """
    Trigger retraining Step Functions workflow.

    Args:
        state_machine_arn: ARN of the Step Functions state machine
        region: AWS region

    Returns:
        Execution response
    """
    client = boto3.client("stepfunctions", region_name=region)

    response = client.start_execution(
        stateMachineArn=state_machine_arn,
        input=json.dumps({"trigger": "drift_detected"}),
    )

    logger.info(f"Started retraining workflow: {response['executionArn']}")

    return response


def main():
    """Main function for drift detection."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-stats", type=str, required=True)
    parser.add_argument("--current-stats", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--trigger-retraining", action="store_true")
    parser.add_argument("--state-machine-arn", type=str, default="")

    args = parser.parse_args()

    # Detect drift
    detector = DriftDetector(
        baseline_stats_path=args.baseline_stats,
        current_stats_path=args.current_stats,
        threshold=args.threshold,
    )

    drift_report = detector.detect_all_drift()

    # Save report
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    report_file = output_path / "drift_report.json"
    with open(report_file, "w") as f:
        json.dump(drift_report, f, indent=2)

    logger.info(f"Drift report saved to {report_file}")

    # Log results
    if drift_report["drift_detected"]:
        logger.warning("DATA DRIFT DETECTED!")
        logger.warning(f"Features with drift: {drift_report['features_with_drift']}")

        if args.trigger_retraining and args.state_machine_arn:
            logger.info("Triggering automated retraining...")
            trigger_retraining(args.state_machine_arn)
    else:
        logger.info("No data drift detected")


if __name__ == "__main__":
    main()
