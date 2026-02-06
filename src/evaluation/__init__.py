# Evaluation package
from .metrics import (
    normalize_string,
    normalize_number,
    string_match,
    number_match,
    string_similarity,
    calculate_field_accuracy,
    calculate_line_items_accuracy,
    calculate_document_accuracy,
    # Tier 1: Extraction accuracy
    find_value_in_raw_response,
    calculate_extraction_accuracy,
    calculate_line_items_extraction,
    # Performance metrics
    extract_performance_metrics,
)
from .compare import (
    DocumentComparison,
    load_ground_truth,
    load_ocr_results,
    load_raw_responses,
    compare_document,
    compare_all,
    aggregate_by_effect,
    aggregate_by_format,
    get_worst_documents,
    get_best_documents,
)
