"""Generate comparison reports from OCR results with two-tier evaluation."""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.evaluation.compare import (
    compare_all,
    aggregate_by_effect,
    aggregate_by_format,
    get_worst_documents,
    get_best_documents,
)
from src.reporting.html_report import generate_html_report

# MLflow integration (graceful degradation if unavailable or server unreachable)
try:
    import mlflow
    from mlflow import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# MLflow configuration
MLFLOW_SERVER_URI = "http://10.174.93.17:5000"
MLFLOW_EXPERIMENT_NAME = "OCR_Research"


def sanitize_metric_name(name: str) -> str:
    """Sanitize a string for use as an MLflow metric name.
    
    MLflow metric names must match: ^[a-zA-Z0-9_\\-\\.]+$
    """
    sanitized = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name)
    sanitized = re.sub(r'_+', '_', sanitized)
    return sanitized.strip('_-.')


def log_to_mlflow(report: dict, report_path: Path, html_path: Path = None) -> None:
    """Log evaluation results to MLflow tracking server.
    
    Logs parameters, metrics, and artifacts for a model evaluation run.
    Fails silently if MLflow is unavailable or the server is unreachable.
    
    Args:
        report: The full report dictionary
        report_path: Path to the saved JSON report (logged as artifact)
        html_path: Path to the saved HTML report (logged as artifact, optional)
    """
    if not MLFLOW_AVAILABLE:
        print("MLflow not installed, skipping logging.")
        return
    
    try:
        mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
        
        # Create or get experiment
        client = MlflowClient(tracking_uri=MLFLOW_SERVER_URI)
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = client.create_experiment(MLFLOW_EXPERIMENT_NAME)
        else:
            experiment_id = experiment.experiment_id
        
        model_name = report.get("model_name", "unknown")
        summary = report.get("summary", {})
        perf = report.get("performance", {})
        
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            # Parameters
            mlflow.log_params({
                "model_name": model_name,
                "evaluation_type": report.get("evaluation_type", "two_tier"),
                "total_documents": str(summary.get("total_documents", 0)),
                "generated_at": report.get("generated_at", ""),
            })
            
            # Summary metrics
            mlflow.log_metrics({
                "extraction_accuracy": summary.get("extraction_accuracy", 0),
                "schema_accuracy": summary.get("schema_accuracy", 0),
                "overall_accuracy": summary.get("overall_accuracy", 0),
            })
            
            # Performance metrics
            mlflow.log_metrics({
                "avg_processing_time_s": perf.get("avg_processing_time_seconds", 0),
                "throughput_docs_per_min": perf.get("throughput_docs_per_min", 0),
                "total_cost_usd": perf.get("total_cost_usd", 0),
                "cost_per_page_usd": perf.get("cost_per_page_usd", 0),
                "avg_confidence": perf.get("avg_confidence", 0),
            })
            
            # Per-format extraction accuracy
            for fmt, metrics in report.get("by_format", {}).items():
                key = sanitize_metric_name(f"extraction_accuracy.format.{fmt}")
                mlflow.log_metric(key, metrics.get("avg_extraction_accuracy", 0))
            
            # Per-effect extraction accuracy
            for effect, metrics in report.get("by_effect_type", {}).items():
                key = sanitize_metric_name(f"extraction_accuracy.effect.{effect}")
                mlflow.log_metric(key, metrics.get("avg_extraction_accuracy", 0))
            
            # Tags
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("run_type", "evaluation")
            
            # Artifacts
            if report_path.exists():
                mlflow.log_artifact(str(report_path))
            if html_path and html_path.exists():
                mlflow.log_artifact(str(html_path))
        
        print(f"MLflow: Logged run '{run_name}' to experiment '{MLFLOW_EXPERIMENT_NAME}'")
        print(f"MLflow: View at {MLFLOW_SERVER_URI}/#/experiments/{experiment_id}")
    
    except Exception as e:
        print(f"MLflow: Logging failed (server may be unreachable): {e}")


def round_floats(obj, decimals: int = 3):
    """Recursively round all float values in a data structure.
    
    Args:
        obj: Any data structure (dict, list, or primitive)
        decimals: Number of decimal places to round to
        
    Returns:
        Same structure with floats rounded
    """
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, decimals) for item in obj]
    return obj


def calculate_summary_metrics(comparisons: list) -> dict:
    """Calculate summary metrics across all documents.
    
    Args:
        comparisons: List of DocumentComparison objects
        
    Returns:
        Dictionary with extraction, schema, and overall accuracy
    """
    if not comparisons:
        return {
            "extraction_accuracy": 0.0,
            "schema_accuracy": 0.0,
            "overall_accuracy": 0.0,
        }
    
    extraction_accs = [c.extraction_accuracy for c in comparisons]
    schema_accs = [c.schema_accuracy for c in comparisons]
    overall_accs = [c.overall_accuracy for c in comparisons]
    
    return {
        "extraction_accuracy": sum(extraction_accs) / len(extraction_accs),
        "schema_accuracy": sum(schema_accs) / len(schema_accs),
        "overall_accuracy": sum(overall_accs) / len(overall_accs),
    }


def calculate_performance_summary(comparisons: list) -> dict:
    """Calculate aggregate performance metrics across all documents.
    
    Args:
        comparisons: List of DocumentComparison objects
        
    Returns:
        Dictionary with performance metrics for the report
    """
    if not comparisons:
        return {
            "total_processing_time_seconds": 0.0,
            "avg_processing_time_seconds": 0.0,
            "min_processing_time_seconds": 0.0,
            "max_processing_time_seconds": 0.0,
            "throughput_docs_per_min": 0.0,
            "total_pages": 0,
            "total_cost_usd": 0.0,
            "cost_per_page_usd": 0.0,
            "avg_confidence": 0.0,
        }
    
    times = [c.processing_time_seconds for c in comparisons]
    total_time = sum(times)
    total_pages = sum(c.page_count for c in comparisons)
    total_cost = sum(c.estimated_cost_usd for c in comparisons)
    confidences = [c.avg_confidence for c in comparisons if c.avg_confidence > 0]
    
    return {
        "total_processing_time_seconds": total_time,
        "avg_processing_time_seconds": total_time / len(comparisons),
        "min_processing_time_seconds": min(times) if times else 0.0,
        "max_processing_time_seconds": max(times) if times else 0.0,
        "throughput_docs_per_min": (len(comparisons) / total_time * 60) if total_time > 0 else 0.0,
        "total_pages": total_pages,
        "total_cost_usd": total_cost,
        "cost_per_page_usd": total_cost / total_pages if total_pages > 0 else 0.0,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
    }


def save_json_report(report: dict, config, model_name: str) -> Path:
    """Save detailed JSON report.
    
    Args:
        report: Report dictionary
        config: Configuration object
        model_name: Name of the OCR model
        
    Returns:
        Path to saved report
    """
    report_file = config.reports_path / f"{model_name}_comparison_report.json"
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report_file


def print_summary_table(report: dict, model_name: str) -> None:
    """Print formatted summary table to console with two-tier metrics.
    
    Args:
        report: Report dictionary
        model_name: Name of the OCR model
    """
    summary = report["summary"]
    by_effect = report["by_effect_type"]
    by_format = report["by_format"]
    worst = report["worst_performing"]
    
    print("\n" + "=" * 75)
    print("OCR Model Comparison Report (Two-Tier Evaluation)")
    print("=" * 75)
    print(f"Model: {model_name}")
    print(f"Generated: {report['generated_at']}")
    print(f"Total Documents: {summary['total_documents']}")
    
    # Two-tier summary
    print("\n" + "-" * 75)
    print("ACCURACY SUMMARY:")
    print(f"  Extraction Accuracy (content found anywhere):    {summary['extraction_accuracy'] * 100:>6.1f}%")
    print(f"  Schema Accuracy (content in expected fields):    {summary['schema_accuracy'] * 100:>6.1f}%")
    print(f"  Overall Accuracy (weighted combination):         {summary['overall_accuracy'] * 100:>6.1f}%")
    
    # By Effect Type
    print("\n" + "-" * 75)
    print("By Effect Type:")
    print(f"  {'Effect':<20} | {'Count':>5} | {'Extraction':>10} | {'Schema':>10} | {'Overall':>10}")
    print(f"  {'-'*20} | {'-'*5} | {'-'*10} | {'-'*10} | {'-'*10}")
    
    # Sort by extraction accuracy descending
    sorted_effects = sorted(
        by_effect.items(),
        key=lambda x: x[1].get("avg_extraction_accuracy", x[1].get("avg_accuracy", 0)),
        reverse=True
    )
    
    for effect, metrics in sorted_effects:
        ext_acc = metrics.get("avg_extraction_accuracy", metrics.get("avg_accuracy", 0))
        schema_acc = metrics.get("avg_schema_accuracy", metrics.get("avg_field_accuracy", 0))
        overall_acc = metrics.get("avg_overall_accuracy", metrics.get("avg_accuracy", 0))
        
        print(
            f"  {effect:<20} | {metrics['count']:>5} | "
            f"{ext_acc * 100:>9.1f}% | "
            f"{schema_acc * 100:>9.1f}% | "
            f"{overall_acc * 100:>9.1f}%"
        )
    
    # By Format
    print("\n" + "-" * 75)
    print("By Format:")
    print(f"  {'Format':<10} | {'Count':>5} | {'Extraction':>10} | {'Schema':>10} | {'Overall':>10}")
    print(f"  {'-'*10} | {'-'*5} | {'-'*10} | {'-'*10} | {'-'*10}")
    
    sorted_formats = sorted(
        by_format.items(),
        key=lambda x: x[1].get("avg_extraction_accuracy", x[1].get("avg_accuracy", 0)),
        reverse=True
    )
    
    for fmt, metrics in sorted_formats:
        ext_acc = metrics.get("avg_extraction_accuracy", metrics.get("avg_accuracy", 0))
        schema_acc = metrics.get("avg_schema_accuracy", metrics.get("avg_field_accuracy", 0))
        overall_acc = metrics.get("avg_overall_accuracy", metrics.get("avg_accuracy", 0))
        
        print(
            f"  {fmt.upper():<10} | {metrics['count']:>5} | "
            f"{ext_acc * 100:>9.1f}% | "
            f"{schema_acc * 100:>9.1f}% | "
            f"{overall_acc * 100:>9.1f}%"
        )
    
    # Worst Performing (by extraction accuracy - the fair metric)
    print("\n" + "-" * 75)
    print("Worst Performing Documents (by Extraction Accuracy):")
    
    for i, doc in enumerate(worst[:10], 1):
        ext_acc = doc.get('extraction_accuracy', doc.get('overall_accuracy', 0))
        schema_acc = doc.get('schema_accuracy', 0)
        
        print(
            f"  {i:>2}. {doc['filename']:<35} "
            f"Ext: {ext_acc * 100:>5.1f}% | "
            f"Schema: {schema_acc * 100:>5.1f}% "
            f"({doc['effect_type']})"
        )
    
    # Extraction details for first document (if available)
    per_doc = report.get("per_document", [])
    if per_doc and per_doc[0].get("extraction_details"):
        print("\n" + "-" * 75)
        print(f"Field Extraction Details (Sample: {per_doc[0]['filename']}):")
        print(f"  {'Field':<25} | {'Found':>5} | {'In Field':<25}")
        print(f"  {'-'*25} | {'-'*5} | {'-'*25}")
        
        for field_name, details in per_doc[0]["extraction_details"].items():
            found = "YES" if details.get("found") else "NO"
            in_field = details.get("found_in_field", "-") or "-"
            print(f"  {field_name:<25} | {found:>5} | {in_field:<25}")
    
    print("\n" + "=" * 75)
    print("\nNote: Extraction Accuracy measures if the OCR found the content anywhere.")
    print("      Schema Accuracy measures if it was placed in the expected field.")
    print("      For fair model comparison, use Extraction Accuracy.")
    print("=" * 75)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comparison report from OCR results (two-tier evaluation)"
    )
    parser.add_argument(
        "--model",
        default="document_intelligence",
        help="OCR model name (default: document_intelligence)"
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: reports/<model>_comparison_report.json)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Don't print summary to console"
    )
    args = parser.parse_args()
    
    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config()
        
        # Run comparison
        print(f"Comparing OCR results for model: {args.model}")
        comparisons = compare_all(config, args.model)
        
        if not comparisons:
            print("ERROR: No comparison results. Have you run the OCR processing?")
            print(f"Run: poetry run python scripts/run_ocr.py")
            sys.exit(1)
        
        print(f"Compared {len(comparisons)} documents")
        
        # Calculate summary metrics
        summary_metrics = calculate_summary_metrics(comparisons)
        performance_metrics = calculate_performance_summary(comparisons)
        
        # Generate report
        detailed_report = {
            "generated_at": datetime.now().isoformat(),
            "model_name": args.model,
            "evaluation_type": "two_tier",
            "summary": {
                "total_documents": len(comparisons),
                **summary_metrics,
            },
            "performance": performance_metrics,
            "by_effect_type": aggregate_by_effect(comparisons),
            "by_format": aggregate_by_format(comparisons),
            "worst_performing": get_worst_documents(comparisons, n=20, by_metric="extraction"),
            "best_performing": get_best_documents(comparisons, n=10, by_metric="extraction"),
            "per_document": [c.to_dict() for c in comparisons],
        }
        
        # Round all float values to 3 decimal places
        detailed_report = round_floats(detailed_report, decimals=3)
        
        # Save report
        if args.output:
            report_path = Path(args.output)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(detailed_report, f, indent=2, ensure_ascii=False)
        else:
            report_path = save_json_report(detailed_report, config, args.model)
        
        print(f"JSON report saved to: {report_path}")
        
        # Generate HTML report
        html_report_path = report_path.with_suffix(".html")
        try:
            generate_html_report(detailed_report, html_report_path)
            print(f"HTML report saved to: {html_report_path}")
        except Exception as html_err:
            print(f"WARNING: HTML report generation failed: {html_err}")
        
        # Log to MLflow
        log_to_mlflow(detailed_report, report_path, html_report_path)
        
        # Print summary
        if not args.quiet:
            print_summary_table(detailed_report, args.model)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
