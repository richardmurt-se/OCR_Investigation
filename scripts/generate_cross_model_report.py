"""Generate cross-model comparison report from individual model reports."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from jinja2 import Environment, FileSystemLoader, select_autoescape


# Fixed colors for each model (consistent across all charts)
MODEL_COLORS = {
    "document_intelligence": "#3b82f6",  # blue
    "paddle_ocr": "#22c55e",             # green
    "glm-ocr": "#f97316",               # orange
    "deepseek-ocr-2": "#8b5cf6",        # purple
    "nemotron-parse": "#ec4899",         # pink
    "chandra": "#14b8a6",               # teal
}

# Fallback palette for unknown models
FALLBACK_COLORS = ["#6366f1", "#f43f5e", "#0ea5e9", "#84cc16", "#a855f7", "#d946ef"]


def get_model_color(model_name: str, idx: int = 0) -> str:
    """Get a consistent color for a model."""
    return MODEL_COLORS.get(model_name, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def get_display_name(model_name: str) -> str:
    """Convert model_name to a human-readable display name."""
    names = {
        "document_intelligence": "Azure Document Intelligence",
        "paddle_ocr": "PaddleOCR",
        "glm-ocr": "GLM-OCR",
        "deepseek-ocr-2": "DeepSeek-OCR-2",
        "nemotron-parse": "Nemotron-Parse",
        "chandra": "Chandra",
    }
    return names.get(model_name, model_name.replace("_", " ").replace("-", " ").title())


def discover_reports(reports_dir: Path) -> list:
    """Auto-discover all per-model comparison report JSONs."""
    reports = []
    for f in sorted(reports_dir.glob("*_comparison_report.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            reports.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"WARNING: Skipping {f.name}: {e}")
    return reports


def build_comparison_data(reports: list) -> dict:
    """Build the template context from multiple model reports."""
    models = []
    for i, report in enumerate(reports):
        name = report.get("model_name", f"model_{i}")
        summary = report.get("summary", {})
        perf = report.get("performance", {})
        by_effect = report.get("by_effect_type", {})
        by_format = report.get("by_format", {})

        models.append({
            "name": name,
            "display_name": get_display_name(name),
            "color": get_model_color(name, i),
            "total_documents": summary.get("total_documents", 0),
            "extraction_accuracy": summary.get("extraction_accuracy", 0),
            "avg_time": perf.get("avg_processing_time_seconds", 0),
            "throughput": perf.get("throughput_docs_per_min", 0),
            "total_cost": perf.get("total_cost_usd", 0),
            "cost_per_page": perf.get("cost_per_page_usd", 0),
            "avg_confidence": perf.get("avg_confidence", 0),
            "by_effect": by_effect,
            "by_format": by_format,
        })

    # Find the best model for extraction accuracy
    if models:
        best_extraction = max(models, key=lambda m: m["extraction_accuracy"])
        for m in models:
            m["is_best_extraction"] = (m["name"] == best_extraction["name"])
    
    # Collect all effect types across all models
    all_effects = set()
    for m in models:
        all_effects.update(m["by_effect"].keys())
    # Sort effects by average extraction accuracy across models (hardest last)
    effect_list = sorted(all_effects, key=lambda e: -sum(
        m["by_effect"].get(e, {}).get("avg_extraction_accuracy", 0) for m in models
    ) / max(len(models), 1))

    # Collect all formats
    all_formats = set()
    for m in models:
        all_formats.update(m["by_format"].keys())
    format_list = sorted(all_formats)

    # Build chart data for effect types
    effect_chart = {
        "labels": [e.replace("_", " ").title() for e in effect_list],
        "datasets": [],
    }
    for m in models:
        effect_chart["datasets"].append({
            "label": m["display_name"],
            "data": [
                round(m["by_effect"].get(e, {}).get("avg_extraction_accuracy", 0) * 100, 1)
                for e in effect_list
            ],
            "color": m["color"],
        })

    # Build chart data for formats
    format_chart = {
        "labels": [f.upper() for f in format_list],
        "datasets": [],
    }
    for m in models:
        format_chart["datasets"].append({
            "label": m["display_name"],
            "data": [
                round(m["by_format"].get(f, {}).get("avg_extraction_accuracy", 0) * 100, 1)
                for f in format_list
            ],
            "color": m["color"],
        })

    # Build language chart data from per-document results
    # Language codes in filenames: multilingual_{code}_{style}.ext
    # Non-multilingual docs are English
    LANG_NAMES = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "ja": "Japanese",
        "ar": "Arabic",
        "zh": "Chinese",
        "pt": "Portuguese",
        "hu": "Hungarian",
    }

    # Collect per-language extraction accuracies for each model
    lang_accs = {}  # {model_name: {lang_code: [accuracies]}}
    for report in reports:
        name = report.get("model_name")
        lang_accs[name] = {}
        for doc in report.get("per_document", []):
            filename = doc.get("filename", "")
            ext_acc = doc.get("extraction_accuracy", 0)
            if "multilingual_" in filename:
                # Extract language code: e.g., "064_multilingual_es_standard.jpeg" -> "es"
                parts = filename.split("_")
                try:
                    lang_idx = parts.index("multilingual") + 1
                    lang_code = parts[lang_idx]
                except (ValueError, IndexError):
                    lang_code = "unknown"
            else:
                lang_code = "en"
            lang_accs[name].setdefault(lang_code, []).append(ext_acc)

    # Get sorted list of all languages across all models
    all_langs = set()
    for accs in lang_accs.values():
        all_langs.update(accs.keys())
    lang_list = sorted(all_langs, key=lambda l: LANG_NAMES.get(l, l))

    language_chart = {
        "labels": [LANG_NAMES.get(l, l.upper()) for l in lang_list],
        "datasets": [],
    }
    for m in models:
        accs_by_lang = lang_accs.get(m["name"], {})
        language_chart["datasets"].append({
            "label": m["display_name"],
            "data": [
                round((sum(accs_by_lang.get(l, [0])) / len(accs_by_lang.get(l, [1]))) * 100, 1)
                if accs_by_lang.get(l) else 0
                for l in lang_list
            ],
            "color": m["color"],
        })

    return {
        "models": models,
        "effect_chart": effect_chart,
        "format_chart": format_chart,
        "language_chart": language_chart,
    }


def generate_report(reports_dir: Path, output_path: Path, model_filter: list = None):
    """Generate the cross-model comparison HTML report."""
    reports = discover_reports(reports_dir)

    if model_filter:
        reports = [r for r in reports if r.get("model_name") in model_filter]

    if not reports:
        print("ERROR: No model comparison reports found.")
        print(f"Looked in: {reports_dir}")
        sys.exit(1)

    print(f"Found {len(reports)} model reports:")
    for r in reports:
        name = r.get("model_name", "unknown")
        acc = r.get("summary", {}).get("extraction_accuracy", 0)
        print(f"  - {name}: {acc*100:.1f}% extraction accuracy")

    # Build template data
    data = build_comparison_data(reports)

    # Render template
    template_dir = project_root / "src" / "templates"
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("cross_model_report.html")
    html_content = template.render(**data)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nCross-model report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-model OCR comparison report"
    )
    parser.add_argument(
        "--models",
        help="Comma-separated list of model names to include (default: all discovered)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML file path (default: reports/cross_model_comparison.html)"
    )
    args = parser.parse_args()

    reports_dir = project_root / "reports"
    output_path = Path(args.output) if args.output else reports_dir / "cross_model_comparison.html"
    model_filter = [m.strip() for m in args.models.split(",")] if args.models else None

    generate_report(reports_dir, output_path, model_filter)


if __name__ == "__main__":
    main()
