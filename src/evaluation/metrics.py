"""Accuracy metrics for OCR evaluation."""

import difflib
from typing import Dict, List, Any, Optional, Tuple


def normalize_string(s: str) -> str:
    """Normalize a string for comparison.
    
    Converts to lowercase, strips whitespace, and collapses multiple spaces.
    
    Args:
        s: Input string
        
    Returns:
        Normalized string
    """
    if not s:
        return ""
    return " ".join(str(s).lower().split())


def normalize_number(s: str) -> Optional[float]:
    """Parse a number from a string, handling various formats.
    
    Handles formats like:
    - "1 032.82" (space as thousands separator)
    - "1,032.82" (comma as thousands separator)
    - "$1,032.82" (with currency symbol)
    
    Args:
        s: String containing a number
        
    Returns:
        Float value or None if not parseable
    """
    if not s:
        return None
    
    try:
        # Remove common formatting characters
        cleaned = str(s).replace(" ", "").replace(",", "")
        # Remove currency symbols
        cleaned = cleaned.replace("$", "").replace("€", "").replace("£", "")
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def string_match(ocr: str, truth: str) -> bool:
    """Check if two strings match after normalization.
    
    Args:
        ocr: OCR extracted value
        truth: Ground truth value
        
    Returns:
        True if strings match
    """
    return normalize_string(ocr) == normalize_string(truth)


def number_match(ocr: str, truth: str, tolerance: float = 0.01) -> bool:
    """Check if two numbers match within tolerance.
    
    Args:
        ocr: OCR extracted value
        truth: Ground truth value
        tolerance: Maximum allowed difference
        
    Returns:
        True if numbers match within tolerance
    """
    ocr_num = normalize_number(ocr)
    truth_num = normalize_number(truth)
    
    if ocr_num is None or truth_num is None:
        # If either can't be parsed, fall back to string comparison
        return string_match(ocr, truth)
    
    return abs(ocr_num - truth_num) < tolerance


def string_similarity(ocr: str, truth: str) -> float:
    """Calculate character-level similarity between two strings.
    
    Uses SequenceMatcher to compute similarity ratio.
    
    Args:
        ocr: OCR extracted value
        truth: Ground truth value
        
    Returns:
        Similarity ratio between 0 and 1
    """
    if not ocr and not truth:
        return 1.0
    if not ocr or not truth:
        return 0.0
    
    ocr_norm = normalize_string(ocr)
    truth_norm = normalize_string(truth)
    
    matcher = difflib.SequenceMatcher(None, ocr_norm, truth_norm)
    return matcher.ratio()


def calculate_field_accuracy(
    ocr_result: Dict[str, Any],
    ground_truth: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Calculate accuracy for each invoice field.
    
    Args:
        ocr_result: OCR extraction result
        ground_truth: Ground truth data
        
    Returns:
        Dictionary mapping field names to accuracy info:
        {
            "field_name": {
                "matched": bool,
                "similarity": float,
                "ocr_value": str,
                "truth_value": str
            }
        }
    """
    results = {}
    
    # Invoice header fields
    invoice_fields = [
        "client_name",
        "client_address",
        "seller_name", 
        "seller_address",
        "invoice_number",
        "invoice_date",
        "due_date",
    ]
    
    ocr_invoice = ocr_result.get("invoice", {})
    truth_invoice = ground_truth.get("invoice", {})
    
    for field in invoice_fields:
        ocr_val = ocr_invoice.get(field, "")
        truth_val = truth_invoice.get(field, "")
        
        results[f"invoice.{field}"] = {
            "matched": string_match(ocr_val, truth_val),
            "similarity": string_similarity(ocr_val, truth_val),
            "ocr_value": str(ocr_val),
            "truth_value": str(truth_val),
        }
    
    # Subtotal fields (numeric)
    subtotal_fields = ["subtotal", "tax", "discount", "total"]
    
    ocr_subtotal = ocr_result.get("subtotal", {})
    truth_subtotal = ground_truth.get("subtotal", {})
    
    for field in subtotal_fields:
        ocr_val = ocr_subtotal.get(field, "")
        truth_val = truth_subtotal.get(field, "")
        
        results[f"subtotal.{field}"] = {
            "matched": number_match(ocr_val, truth_val),
            "similarity": 1.0 if number_match(ocr_val, truth_val) else 0.0,
            "ocr_value": str(ocr_val),
            "truth_value": str(truth_val),
        }
    
    return results


def calculate_line_items_accuracy(
    ocr_items: List[Dict[str, Any]],
    truth_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Calculate accuracy for line items.
    
    Uses order-independent matching based on description similarity.
    
    Args:
        ocr_items: OCR extracted line items
        truth_items: Ground truth line items
        
    Returns:
        Dictionary with metrics:
        {
            "precision": float,  # Correct OCR items / Total OCR items
            "recall": float,     # Matched truth items / Total truth items
            "f1": float,         # Harmonic mean of precision and recall
            "count_match": bool, # Whether item counts match
            "ocr_count": int,
            "truth_count": int,
            "matched_items": List[Tuple[int, int, float]]  # (ocr_idx, truth_idx, score)
        }
    """
    if not truth_items and not ocr_items:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "count_match": True,
            "ocr_count": 0,
            "truth_count": 0,
            "matched_items": [],
        }
    
    if not truth_items:
        return {
            "precision": 0.0,
            "recall": 1.0,  # Nothing to recall
            "f1": 0.0,
            "count_match": False,
            "ocr_count": len(ocr_items),
            "truth_count": 0,
            "matched_items": [],
        }
    
    if not ocr_items:
        return {
            "precision": 1.0,  # Nothing wrong extracted
            "recall": 0.0,
            "f1": 0.0,
            "count_match": False,
            "ocr_count": 0,
            "truth_count": len(truth_items),
            "matched_items": [],
        }
    
    # Build similarity matrix
    similarity_matrix = []
    for i, ocr_item in enumerate(ocr_items):
        row = []
        for j, truth_item in enumerate(truth_items):
            score = _calculate_item_similarity(ocr_item, truth_item)
            row.append(score)
        similarity_matrix.append(row)
    
    # Greedy matching (could be improved with Hungarian algorithm)
    matched_items = []
    used_ocr = set()
    used_truth = set()
    
    # Sort by similarity score descending
    candidates = []
    for i in range(len(ocr_items)):
        for j in range(len(truth_items)):
            candidates.append((i, j, similarity_matrix[i][j]))
    
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Match greedily
    for ocr_idx, truth_idx, score in candidates:
        if ocr_idx not in used_ocr and truth_idx not in used_truth:
            if score > 0.5:  # Threshold for considering a match
                matched_items.append((ocr_idx, truth_idx, score))
                used_ocr.add(ocr_idx)
                used_truth.add(truth_idx)
    
    # Calculate metrics
    correct_matches = len(matched_items)
    precision = correct_matches / len(ocr_items) if ocr_items else 1.0
    recall = correct_matches / len(truth_items) if truth_items else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_match": len(ocr_items) == len(truth_items),
        "ocr_count": len(ocr_items),
        "truth_count": len(truth_items),
        "matched_items": matched_items,
    }


def _calculate_item_similarity(
    ocr_item: Dict[str, Any],
    truth_item: Dict[str, Any]
) -> float:
    """Calculate similarity between two line items.
    
    Args:
        ocr_item: OCR extracted item
        truth_item: Ground truth item
        
    Returns:
        Similarity score between 0 and 1
    """
    scores = []
    
    # Description similarity (weighted higher)
    desc_sim = string_similarity(
        ocr_item.get("description", ""),
        truth_item.get("description", "")
    )
    scores.append(desc_sim * 2)  # Weight description higher
    
    # Quantity match
    qty_match = number_match(
        ocr_item.get("quantity", ""),
        truth_item.get("quantity", "")
    )
    scores.append(1.0 if qty_match else 0.0)
    
    # Unit price match
    unit_match = number_match(
        ocr_item.get("unit_price", ""),
        truth_item.get("unit_price", "")
    )
    scores.append(1.0 if unit_match else 0.0)
    
    # Total price match
    total_match = number_match(
        ocr_item.get("total_price", ""),
        truth_item.get("total_price", "")
    )
    scores.append(1.0 if total_match else 0.0)
    
    # Return weighted average
    return sum(scores) / (len(scores) + 1)  # +1 because description is weighted 2x


# =============================================================================
# Tier 1: Extraction Accuracy - Content-based search in raw response
# =============================================================================

def find_value_in_raw_response(
    expected_value: str,
    raw_response: Dict[str, Any],
    is_numeric: bool = False
) -> Dict[str, Any]:
    """Search raw API response for an expected value.
    
    Searches through all fields in the raw response to find if the expected
    value was extracted anywhere, regardless of which field it ended up in.
    
    Args:
        expected_value: The ground truth value to search for
        raw_response: The raw API response dictionary
        is_numeric: Whether to use numeric comparison
        
    Returns:
        Dictionary with search results:
        {
            "found": bool,
            "found_in_field": str or None,
            "found_value": str or None,
            "confidence": float or None,
            "match_type": "exact" | "normalized" | "partial" | None
        }
    """
    if not expected_value or not expected_value.strip():
        # Empty expected values are considered "found" (nothing to find)
        return {
            "found": True,
            "found_in_field": None,
            "found_value": None,
            "confidence": None,
            "match_type": "empty",
        }
    
    if not raw_response:
        return {
            "found": False,
            "found_in_field": None,
            "found_value": None,
            "confidence": None,
            "match_type": None,
        }
    
    expected_norm = normalize_string(expected_value)
    expected_num = normalize_number(expected_value) if is_numeric else None
    
    # Search through document fields
    documents = raw_response.get("documents", [])
    
    for doc in documents:
        fields = doc.get("fields", {})
        
        for field_name, field_data in fields.items():
            if field_data is None:
                continue
                
            # Get field value and content
            field_value = field_data.get("value")
            field_content = field_data.get("content", "")
            field_confidence = field_data.get("confidence")
            
            # Handle nested value structures (like currency with amount)
            if isinstance(field_value, dict):
                if "amount" in field_value:
                    field_value = str(field_value.get("amount", ""))
                else:
                    field_value = str(field_value)
            
            # Check content field
            if field_content:
                content_norm = normalize_string(str(field_content))
                
                # Exact normalized match
                if content_norm == expected_norm:
                    return {
                        "found": True,
                        "found_in_field": field_name,
                        "found_value": str(field_content),
                        "confidence": field_confidence,
                        "match_type": "exact",
                    }
                
                # Numeric match
                if is_numeric and expected_num is not None:
                    content_num = normalize_number(str(field_content))
                    if content_num is not None and abs(content_num - expected_num) < 0.01:
                        return {
                            "found": True,
                            "found_in_field": field_name,
                            "found_value": str(field_content),
                            "confidence": field_confidence,
                            "match_type": "numeric",
                        }
                
                # Partial match (expected value contained in field)
                if expected_norm in content_norm or content_norm in expected_norm:
                    similarity = string_similarity(str(field_content), expected_value)
                    if similarity > 0.8:
                        return {
                            "found": True,
                            "found_in_field": field_name,
                            "found_value": str(field_content),
                            "confidence": field_confidence,
                            "match_type": "partial",
                        }
            
            # Check value field
            if field_value:
                value_str = str(field_value)
                value_norm = normalize_string(value_str)
                
                if value_norm == expected_norm:
                    return {
                        "found": True,
                        "found_in_field": field_name,
                        "found_value": value_str,
                        "confidence": field_confidence,
                        "match_type": "exact",
                    }
                
                if is_numeric and expected_num is not None:
                    value_num = normalize_number(value_str)
                    if value_num is not None and abs(value_num - expected_num) < 0.01:
                        return {
                            "found": True,
                            "found_in_field": field_name,
                            "found_value": value_str,
                            "confidence": field_confidence,
                            "match_type": "numeric",
                        }
    
    # Also search in the full content (OCR text)
    full_content = raw_response.get("content", "")
    if full_content:
        content_norm = normalize_string(full_content)
        if expected_norm in content_norm:
            return {
                "found": True,
                "found_in_field": "_raw_content",
                "found_value": None,  # Too long to return
                "confidence": None,
                "match_type": "in_content",
            }
    
    return {
        "found": False,
        "found_in_field": None,
        "found_value": None,
        "confidence": None,
        "match_type": None,
    }


def calculate_extraction_accuracy(
    ground_truth: Dict[str, Any],
    raw_response: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Calculate extraction accuracy by searching raw response for ground truth values.
    
    This measures whether the OCR model successfully extracted each expected value,
    regardless of which field it was placed in. This is a fairer metric for comparing
    different OCR models that may use different field naming conventions.
    
    Args:
        ground_truth: Ground truth data with expected values
        raw_response: Raw API response from OCR model
        
    Returns:
        Dictionary mapping field names to extraction results:
        {
            "invoice.client_address": {
                "expected_value": "7121 Tyler Burgs...",
                "found": True,
                "found_in_field": "BillingAddress",
                "confidence": 0.884,
                "match_type": "exact"
            }
        }
    """
    results = {}
    
    # Invoice header fields (string-based)
    invoice_fields = [
        "client_name",
        "client_address", 
        "seller_name",
        "seller_address",
        "invoice_number",
        "invoice_date",
        "due_date",
    ]
    
    truth_invoice = ground_truth.get("invoice", {})
    
    for field in invoice_fields:
        expected_value = truth_invoice.get(field, "")
        search_result = find_value_in_raw_response(
            expected_value, 
            raw_response,
            is_numeric=False
        )
        
        results[f"invoice.{field}"] = {
            "expected_value": expected_value,
            **search_result,
        }
    
    # Subtotal fields (numeric)
    subtotal_fields = ["subtotal", "tax", "discount", "total"]
    
    truth_subtotal = ground_truth.get("subtotal", {})
    
    for field in subtotal_fields:
        expected_value = truth_subtotal.get(field, "")
        search_result = find_value_in_raw_response(
            expected_value,
            raw_response,
            is_numeric=True
        )
        
        results[f"subtotal.{field}"] = {
            "expected_value": expected_value,
            **search_result,
        }
    
    return results


def calculate_line_items_extraction(
    truth_items: List[Dict[str, Any]],
    raw_response: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate extraction accuracy for line items.
    
    Searches the raw response to see if each expected line item was extracted.
    
    Args:
        truth_items: Ground truth line items
        raw_response: Raw API response
        
    Returns:
        Dictionary with line item extraction metrics
    """
    if not truth_items:
        return {
            "items_found": 0,
            "items_expected": 0,
            "extraction_rate": 1.0,
            "details": [],
        }
    
    # Get all extracted content to search through
    full_content = normalize_string(raw_response.get("content", ""))
    
    # Also get items from document fields if available
    extracted_descriptions = set()
    documents = raw_response.get("documents", [])
    for doc in documents:
        fields = doc.get("fields", {})
        items_field = fields.get("Items", {})
        if items_field and isinstance(items_field.get("value"), list):
            for item in items_field.get("value", []):
                if isinstance(item, dict):
                    desc = item.get("value", {})
                    if isinstance(desc, dict):
                        desc_val = desc.get("Description", {})
                        if isinstance(desc_val, dict):
                            extracted_descriptions.add(normalize_string(str(desc_val.get("content", ""))))
    
    found_count = 0
    details = []
    
    for item in truth_items:
        expected_desc = item.get("description", "")
        expected_total = item.get("total_price", "")
        
        desc_norm = normalize_string(expected_desc)
        
        # Check if description was found
        found = False
        
        # Check in full content
        if desc_norm and desc_norm in full_content:
            found = True
        
        # Check in extracted descriptions
        for extracted in extracted_descriptions:
            if string_similarity(desc_norm, extracted) > 0.8:
                found = True
                break
        
        if found:
            found_count += 1
        
        details.append({
            "description": expected_desc,
            "total_price": expected_total,
            "found": found,
        })
    
    return {
        "items_found": found_count,
        "items_expected": len(truth_items),
        "extraction_rate": found_count / len(truth_items) if truth_items else 1.0,
        "details": details,
    }


# =============================================================================
# Tier 2: Schema Accuracy - Field-based comparison (existing logic)
# =============================================================================

def calculate_document_accuracy(
    ocr_result: Dict[str, Any],
    ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate overall accuracy for a single document (schema-based).
    
    Args:
        ocr_result: OCR extraction result
        ground_truth: Ground truth data
        
    Returns:
        Dictionary with detailed accuracy metrics
    """
    field_accuracy = calculate_field_accuracy(ocr_result, ground_truth)
    
    items_accuracy = calculate_line_items_accuracy(
        ocr_result.get("items", []),
        ground_truth.get("items", [])
    )
    
    # Calculate overall accuracy
    total_fields = len(field_accuracy)
    matched_fields = sum(1 for f in field_accuracy.values() if f["matched"])
    
    # Include line items in overall score
    items_weight = 0.3  # Line items contribute 30% to overall score
    field_accuracy_pct = matched_fields / total_fields if total_fields > 0 else 0
    overall_accuracy = (1 - items_weight) * field_accuracy_pct + items_weight * items_accuracy["f1"]
    
    return {
        "overall_accuracy": overall_accuracy,
        "field_accuracy": field_accuracy,
        "field_accuracy_pct": field_accuracy_pct,
        "matched_fields": matched_fields,
        "total_fields": total_fields,
        "line_items": items_accuracy,
    }


# =============================================================================
# Performance Metrics - Speed, cost, and confidence extraction
# =============================================================================

def extract_performance_metrics(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """Extract performance-related metrics from raw API response.
    
    Extracts page count, word count, and average confidence scores from
    the raw OCR API response for performance and cost analysis.
    
    Args:
        raw_response: Raw API response dictionary
        
    Returns:
        Dictionary with performance metrics:
        {
            "page_count": int,
            "word_count": int,
            "avg_field_confidence": float,
            "min_field_confidence": float,
            "max_field_confidence": float,
        }
    """
    if not raw_response:
        return {
            "page_count": 0,
            "word_count": 0,
            "avg_field_confidence": 0.0,
            "min_field_confidence": 0.0,
            "max_field_confidence": 0.0,
        }
    
    # Extract page count
    pages = raw_response.get("pages", [])
    page_count = len(pages) if pages else 1  # Default to 1 if no pages info
    
    # Extract word count
    word_count = 0
    for page in pages:
        words = page.get("words", [])
        word_count += len(words)
    
    # Extract field confidence scores from documents (Azure format)
    confidences = []
    documents = raw_response.get("documents", [])
    
    for doc in documents:
        fields = doc.get("fields", {})
        for field_name, field_data in fields.items():
            if field_data and isinstance(field_data, dict):
                confidence = field_data.get("confidence")
                if confidence is not None:
                    confidences.append(confidence)
    
    # Fallback: extract from detections array (PaddleOCR / other models)
    if not confidences:
        detections = raw_response.get("detections", [])
        for det in detections:
            if isinstance(det, dict):
                confidence = det.get("confidence")
                if confidence is not None:
                    confidences.append(confidence)
    
    # Calculate confidence statistics
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
    else:
        avg_confidence = 0.0
        min_confidence = 0.0
        max_confidence = 0.0
    
    return {
        "page_count": page_count,
        "word_count": word_count,
        "avg_field_confidence": avg_confidence,
        "min_field_confidence": min_confidence,
        "max_field_confidence": max_confidence,
    }
