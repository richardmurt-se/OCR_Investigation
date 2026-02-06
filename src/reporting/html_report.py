"""HTML report generation for OCR comparison results."""

from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from jinja2 import Environment, FileSystemLoader, select_autoescape


def get_color_class(value: float, thresholds: tuple = (0.9, 0.7)) -> str:
    """Get CSS color class based on value thresholds.
    
    Args:
        value: Value between 0 and 1
        thresholds: (high, medium) thresholds for color coding
        
    Returns:
        CSS class name: 'success', 'warning', or 'danger'
    """
    high, medium = thresholds
    if value >= high:
        return "success"
    elif value >= medium:
        return "warning"
    else:
        return "danger"


def prepare_summary_cards(report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prepare summary card data for the template.
    
    Args:
        report_data: Full report data dictionary
        
    Returns:
        List of card dictionaries with label, value, color_class, and detail
    """
    summary = report_data.get("summary", {})
    performance = report_data.get("performance", {})
    
    cards = [
        {
            "label": "Extraction Accuracy",
            "value": f"{summary.get('extraction_accuracy', 0) * 100:.1f}%",
            "color_class": get_color_class(summary.get('extraction_accuracy', 0)),
            "detail": "Content found anywhere",
        },
        {
            "label": "Schema Accuracy",
            "value": f"{summary.get('schema_accuracy', 0) * 100:.1f}%",
            "color_class": get_color_class(summary.get('schema_accuracy', 0)),
            "detail": "Content in expected fields",
        },
        {
            "label": "Overall Accuracy",
            "value": f"{summary.get('overall_accuracy', 0) * 100:.1f}%",
            "color_class": get_color_class(summary.get('overall_accuracy', 0)),
            "detail": "Weighted combination",
        },
        {
            "label": "Avg Processing Time",
            "value": f"{performance.get('avg_processing_time_seconds', 0):.2f}s",
            "color_class": "info",
            "detail": f"{performance.get('throughput_docs_per_min', 0):.1f} docs/min",
        },
        {
            "label": "Total Cost",
            "value": f"${performance.get('total_cost_usd', 0):.4f}",
            "color_class": "info",
            "detail": f"{performance.get('total_pages', 0)} pages processed",
        },
        {
            "label": "Avg Confidence",
            "value": f"{performance.get('avg_confidence', 0) * 100:.1f}%",
            "color_class": get_color_class(performance.get('avg_confidence', 0)),
            "detail": "Field-level average",
        },
    ]
    
    return cards


def prepare_format_chart_data(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare chart data for accuracy by file format.
    
    Args:
        report_data: Full report data dictionary
        
    Returns:
        Dictionary with labels and data arrays for Chart.js
    """
    by_format = report_data.get("by_format", {})
    
    labels = []
    extraction = []
    schema = []
    overall = []
    
    for format_name, data in by_format.items():
        labels.append(format_name.upper())
        extraction.append(round(data.get("avg_extraction_accuracy", 0) * 100, 1))
        schema.append(round(data.get("avg_schema_accuracy", 0) * 100, 1))
        overall.append(round(data.get("avg_overall_accuracy", 0) * 100, 1))
    
    return {
        "labels": labels,
        "extraction": extraction,
        "schema": schema,
        "overall": overall,
    }


def prepare_effect_chart_data(report_data: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare chart data for accuracy by effect type.
    
    Args:
        report_data: Full report data dictionary
        
    Returns:
        Dictionary with labels and data arrays for Chart.js
    """
    by_effect = report_data.get("by_effect_type", {})
    
    labels = []
    extraction = []
    schema = []
    overall = []
    
    for effect_name, data in by_effect.items():
        labels.append(effect_name.title())
        extraction.append(round(data.get("avg_extraction_accuracy", 0) * 100, 1))
        schema.append(round(data.get("avg_schema_accuracy", 0) * 100, 1))
        overall.append(round(data.get("avg_overall_accuracy", 0) * 100, 1))
    
    return {
        "labels": labels,
        "extraction": extraction,
        "schema": schema,
        "overall": overall,
    }


def format_timestamp(iso_timestamp: str) -> str:
    """Format ISO timestamp for display.
    
    Args:
        iso_timestamp: ISO format timestamp string
        
    Returns:
        Human-readable date/time string
    """
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return iso_timestamp


def generate_html_report(
    report_data: Dict[str, Any],
    output_path: Path
) -> None:
    """Generate HTML report from JSON report data.
    
    Args:
        report_data: Dictionary containing the full comparison report
        output_path: Path to save the HTML file
    """
    # Get template directory
    template_dir = Path(__file__).parent.parent / "templates"
    
    # Set up Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(['html', 'xml'])
    )
    
    # Load template
    template = env.get_template("report.html")
    
    # Prepare template data
    template_data = {
        "model_name": report_data.get("model_name", "Unknown"),
        "generated_at": format_timestamp(report_data.get("generated_at", "")),
        "evaluation_type": report_data.get("evaluation_type", "two_tier"),
        "summary": report_data.get("summary", {}),
        "performance": report_data.get("performance", {}),
        "by_effect_type": report_data.get("by_effect_type", {}),
        "by_format": report_data.get("by_format", {}),
        "per_document": report_data.get("per_document", []),
        "summary_cards": prepare_summary_cards(report_data),
        "format_chart_data": prepare_format_chart_data(report_data),
        "effect_chart_data": prepare_effect_chart_data(report_data),
    }
    
    # Render template
    html_content = template.render(**template_data)
    
    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
