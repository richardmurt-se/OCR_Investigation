"""Comparison logic for OCR results vs ground truth."""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from ..config import Config, get_model_pricing
from .metrics import (
    calculate_document_accuracy,
    calculate_extraction_accuracy,
    calculate_line_items_extraction,
    extract_performance_metrics,
)


@dataclass
class DocumentComparison:
    """Comparison results for a single document with two-tier evaluation.
    
    Tier 1 (Extraction): Was the content found anywhere in the response?
    Tier 2 (Schema): Was the content in the expected field structure?
    """
    # Required fields (no defaults) - must come first
    filename: str
    effect_type: str
    file_format: str
    extraction_accuracy: float  # % of values found in raw response
    schema_accuracy: float      # % of values in correct fields
    overall_accuracy: float     # Combined (weighted average)
    
    # Optional fields with defaults
    extraction_details: Dict[str, Any] = field(default_factory=dict)
    schema_details: Dict[str, Any] = field(default_factory=dict)
    
    # Line items metrics
    line_items_extraction_rate: float = 0.0
    line_items_precision: float = 0.0
    line_items_recall: float = 0.0
    line_items_f1: float = 0.0
    
    # Performance metrics
    processing_time_seconds: float = 0.0
    file_size_bytes: int = 0
    page_count: int = 1
    word_count: int = 0
    avg_confidence: float = 0.0
    estimated_cost_usd: float = 0.0
    
    # Legacy fields for backward compatibility
    field_accuracy_pct: float = 0.0
    matched_fields: int = 0
    total_fields: int = 0
    field_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def load_ground_truth(config: Config) -> Dict[str, Dict]:
    """Load ground truth data indexed by filename.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary mapping filename to ground truth entry
    """
    with open(config.ground_truth_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return {entry["filename"]: entry for entry in data}


def load_ocr_results(config: Config, model_name: str) -> Dict[str, Dict]:
    """Load OCR results indexed by filename.
    
    Args:
        config: Configuration object
        model_name: Name of the OCR model
        
    Returns:
        Dictionary mapping filename to OCR result
    """
    results_dir = config.get_model_results_path(model_name)
    all_results_file = results_dir / "all_results.json"
    
    if all_results_file.exists():
        with open(all_results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {entry["filename"]: entry for entry in data}
    
    # Fall back to individual files
    results = {}
    for result_file in results_dir.glob("*.json"):
        if result_file.name == "all_results.json":
            continue
        with open(result_file, "r", encoding="utf-8") as f:
            entry = json.load(f)
            results[entry["filename"]] = entry
    
    return results


def load_raw_responses(config: Config, model_name: str) -> Dict[str, Dict]:
    """Load raw API responses indexed by result key.
    
    Args:
        config: Configuration object
        model_name: Name of the OCR model
        
    Returns:
        Dictionary mapping result key (e.g., "001_clean_standard_jpeg") to raw response
    """
    results_dir = config.get_model_results_path(model_name)
    raw_dir = results_dir / "raw"
    
    if not raw_dir.exists():
        return {}
    
    raw_responses = {}
    for raw_file in raw_dir.glob("*_raw.json"):
        # Extract result key from raw file name
        # e.g., "001_clean_standard_jpeg_raw.json" -> "001_clean_standard_jpeg"
        stem = raw_file.stem
        if stem.endswith("_raw"):
            result_key = stem[:-4]  # Remove "_raw" suffix
        else:
            result_key = stem
        
        with open(raw_file, "r", encoding="utf-8") as f:
            raw_responses[result_key] = json.load(f)
    
    return raw_responses


def get_result_key(filename: str) -> str:
    """Convert a filename to a result key for matching raw responses.
    
    Args:
        filename: Document filename (e.g., "001_clean_standard.jpeg")
        
    Returns:
        Result key (e.g., "001_clean_standard_jpeg")
    """
    path = Path(filename)
    ext = path.suffix.lstrip(".")
    return f"{path.stem}_{ext}"


def extract_effect_type(ground_truth_entry: Dict) -> str:
    """Extract effect type from ground truth metadata.
    
    Args:
        ground_truth_entry: Ground truth entry with metadata
        
    Returns:
        Effect type string (e.g., "coffee_stains", "low_dpi")
    """
    metadata = ground_truth_entry.get("metadata", {})
    return metadata.get("effect", metadata.get("quality", "unknown"))


def extract_file_format(filename: str) -> str:
    """Extract file format from filename.
    
    Args:
        filename: Document filename
        
    Returns:
        File format string (e.g., "pdf", "png", "jpeg")
    """
    suffix = Path(filename).suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        return "jpeg"
    return suffix.lstrip(".")


def compare_document(
    ocr_result: Dict,
    ground_truth: Dict,
    raw_response: Optional[Dict] = None,
    model_name: str = "document_intelligence"
) -> DocumentComparison:
    """Compare a single document's OCR result against ground truth.
    
    Performs two-tier evaluation:
    - Tier 1 (Extraction): Was the content found anywhere in raw response?
    - Tier 2 (Schema): Was the content in the expected field structure?
    
    Args:
        ocr_result: OCR extraction result (parsed)
        ground_truth: Ground truth data
        raw_response: Raw API response (optional, for extraction accuracy)
        model_name: Name of OCR model (for cost calculation)
        
    Returns:
        DocumentComparison with detailed scores for both tiers
    """
    filename = ocr_result.get("filename", ground_truth.get("filename", "unknown"))
    effect_type = extract_effect_type(ground_truth)
    file_format = extract_file_format(filename)
    
    # Tier 2: Schema accuracy (existing logic)
    schema_result = calculate_document_accuracy(ocr_result, ground_truth)
    schema_accuracy = schema_result["field_accuracy_pct"]
    schema_details = schema_result["field_accuracy"]
    
    # Tier 1: Extraction accuracy (new - requires raw response)
    if raw_response:
        extraction_details = calculate_extraction_accuracy(ground_truth, raw_response)
        
        # Calculate extraction accuracy percentage
        total_extraction_fields = len(extraction_details)
        found_fields = sum(1 for f in extraction_details.values() if f.get("found", False))
        extraction_accuracy = found_fields / total_extraction_fields if total_extraction_fields > 0 else 0.0
        
        # Line items extraction
        line_items_extraction = calculate_line_items_extraction(
            ground_truth.get("items", []),
            raw_response
        )
        line_items_extraction_rate = line_items_extraction["extraction_rate"]
        
        # Extract performance metrics from raw response
        perf_metrics = extract_performance_metrics(raw_response)
    else:
        # No raw response available - fall back to schema accuracy
        extraction_accuracy = schema_accuracy
        extraction_details = {}
        line_items_extraction_rate = schema_result["line_items"]["f1"]
        perf_metrics = {
            "page_count": 1,
            "word_count": 0,
            "avg_field_confidence": 0.0,
        }
    
    # Combined overall accuracy: weighted average of extraction and schema
    # Extraction is more important for fair model comparison (60%)
    # Schema matters for integration (40%)
    overall_accuracy = 0.6 * extraction_accuracy + 0.4 * schema_accuracy
    
    # Extract performance data from OCR result (added during processing)
    processing_time_seconds = ocr_result.get("processing_time_seconds", 0.0)
    file_size_bytes = ocr_result.get("file_size_bytes", 0)
    
    # Calculate estimated cost
    pricing = get_model_pricing(model_name)
    page_count = perf_metrics.get("page_count", 1)
    estimated_cost_usd = page_count * pricing.get("cost_per_page", 0.0)
    
    return DocumentComparison(
        filename=filename,
        effect_type=effect_type,
        file_format=file_format,
        # Tier 1: Extraction
        extraction_accuracy=extraction_accuracy,
        extraction_details=extraction_details,
        # Tier 2: Schema
        schema_accuracy=schema_accuracy,
        schema_details=schema_details,
        # Combined
        overall_accuracy=overall_accuracy,
        # Line items
        line_items_extraction_rate=line_items_extraction_rate,
        line_items_precision=schema_result["line_items"]["precision"],
        line_items_recall=schema_result["line_items"]["recall"],
        line_items_f1=schema_result["line_items"]["f1"],
        # Performance metrics
        processing_time_seconds=processing_time_seconds,
        file_size_bytes=file_size_bytes,
        page_count=page_count,
        word_count=perf_metrics.get("word_count", 0),
        avg_confidence=perf_metrics.get("avg_field_confidence", 0.0),
        estimated_cost_usd=estimated_cost_usd,
        # Legacy fields
        field_accuracy_pct=schema_result["field_accuracy_pct"],
        matched_fields=schema_result["matched_fields"],
        total_fields=schema_result["total_fields"],
        field_details=schema_result["field_accuracy"],
    )


def compare_all(
    config: Config,
    model_name: str = "document_intelligence"
) -> List[DocumentComparison]:
    """Compare all OCR results against ground truth.
    
    Loads raw responses when available for extraction accuracy calculation.
    
    Args:
        config: Configuration object
        model_name: Name of the OCR model
        
    Returns:
        List of DocumentComparison objects
    """
    ground_truth = load_ground_truth(config)
    ocr_results = load_ocr_results(config, model_name)
    raw_responses = load_raw_responses(config, model_name)
    
    comparisons = []
    
    for filename, ocr_result in ocr_results.items():
        truth = ground_truth.get(filename)
        
        if truth is None:
            # Skip documents without ground truth
            continue
        
        # Get raw response for this document
        result_key = get_result_key(filename)
        raw_response = raw_responses.get(result_key)
        
        comparison = compare_document(ocr_result, truth, raw_response, model_name)
        comparisons.append(comparison)
    
    return comparisons


def aggregate_by_effect(
    comparisons: List[DocumentComparison]
) -> Dict[str, Dict[str, Any]]:
    """Aggregate comparison results by effect type.
    
    Args:
        comparisons: List of document comparisons
        
    Returns:
        Dictionary mapping effect type to aggregated metrics including
        both extraction and schema accuracy.
    """
    by_effect: Dict[str, List[DocumentComparison]] = defaultdict(list)
    
    for comp in comparisons:
        by_effect[comp.effect_type].append(comp)
    
    results = {}
    
    for effect, comps in by_effect.items():
        overall_accs = [c.overall_accuracy for c in comps]
        extraction_accs = [c.extraction_accuracy for c in comps]
        schema_accs = [c.schema_accuracy for c in comps]
        items_f1s = [c.line_items_f1 for c in comps]
        
        results[effect] = {
            "count": len(comps),
            # Two-tier metrics
            "avg_extraction_accuracy": sum(extraction_accs) / len(extraction_accs),
            "avg_schema_accuracy": sum(schema_accs) / len(schema_accs),
            "avg_overall_accuracy": sum(overall_accs) / len(overall_accs),
            # Ranges
            "min_extraction": min(extraction_accs),
            "max_extraction": max(extraction_accs),
            "min_schema": min(schema_accs),
            "max_schema": max(schema_accs),
            # Line items
            "avg_items_f1": sum(items_f1s) / len(items_f1s),
            # Legacy (for backward compatibility)
            "avg_accuracy": sum(overall_accs) / len(overall_accs),
            "avg_field_accuracy": sum(schema_accs) / len(schema_accs),
        }
    
    return results


def aggregate_by_format(
    comparisons: List[DocumentComparison]
) -> Dict[str, Dict[str, Any]]:
    """Aggregate comparison results by file format.
    
    Args:
        comparisons: List of document comparisons
        
    Returns:
        Dictionary mapping file format to aggregated metrics including
        both extraction and schema accuracy.
    """
    by_format: Dict[str, List[DocumentComparison]] = defaultdict(list)
    
    for comp in comparisons:
        by_format[comp.file_format].append(comp)
    
    results = {}
    
    for fmt, comps in by_format.items():
        overall_accs = [c.overall_accuracy for c in comps]
        extraction_accs = [c.extraction_accuracy for c in comps]
        schema_accs = [c.schema_accuracy for c in comps]
        items_f1s = [c.line_items_f1 for c in comps]
        
        results[fmt] = {
            "count": len(comps),
            # Two-tier metrics
            "avg_extraction_accuracy": sum(extraction_accs) / len(extraction_accs),
            "avg_schema_accuracy": sum(schema_accs) / len(schema_accs),
            "avg_overall_accuracy": sum(overall_accs) / len(overall_accs),
            # Ranges
            "min_extraction": min(extraction_accs),
            "max_extraction": max(extraction_accs),
            "min_schema": min(schema_accs),
            "max_schema": max(schema_accs),
            # Line items
            "avg_items_f1": sum(items_f1s) / len(items_f1s),
            # Legacy
            "avg_accuracy": sum(overall_accs) / len(overall_accs),
            "avg_field_accuracy": sum(schema_accs) / len(schema_accs),
        }
    
    return results


def get_worst_documents(
    comparisons: List[DocumentComparison],
    n: int = 10,
    by_metric: str = "extraction"
) -> List[Dict[str, Any]]:
    """Get the worst performing documents.
    
    Args:
        comparisons: List of document comparisons
        n: Number of documents to return
        by_metric: Sort by "extraction", "schema", or "overall"
        
    Returns:
        List of dictionaries with filename and accuracy metrics
    """
    if by_metric == "extraction":
        sorted_comps = sorted(comparisons, key=lambda c: c.extraction_accuracy)
    elif by_metric == "schema":
        sorted_comps = sorted(comparisons, key=lambda c: c.schema_accuracy)
    else:
        sorted_comps = sorted(comparisons, key=lambda c: c.overall_accuracy)
    
    return [
        {
            "filename": c.filename,
            "extraction_accuracy": c.extraction_accuracy,
            "schema_accuracy": c.schema_accuracy,
            "overall_accuracy": c.overall_accuracy,
            "effect_type": c.effect_type,
            "file_format": c.file_format,
        }
        for c in sorted_comps[:n]
    ]


def get_best_documents(
    comparisons: List[DocumentComparison],
    n: int = 10,
    by_metric: str = "extraction"
) -> List[Dict[str, Any]]:
    """Get the best performing documents.
    
    Args:
        comparisons: List of document comparisons
        n: Number of documents to return
        by_metric: Sort by "extraction", "schema", or "overall"
        
    Returns:
        List of dictionaries with filename and accuracy metrics
    """
    if by_metric == "extraction":
        sorted_comps = sorted(comparisons, key=lambda c: c.extraction_accuracy, reverse=True)
    elif by_metric == "schema":
        sorted_comps = sorted(comparisons, key=lambda c: c.schema_accuracy, reverse=True)
    else:
        sorted_comps = sorted(comparisons, key=lambda c: c.overall_accuracy, reverse=True)
    
    return [
        {
            "filename": c.filename,
            "extraction_accuracy": c.extraction_accuracy,
            "schema_accuracy": c.schema_accuracy,
            "overall_accuracy": c.overall_accuracy,
            "effect_type": c.effect_type,
            "file_format": c.file_format,
        }
        for c in sorted_comps[:n]
    ]
