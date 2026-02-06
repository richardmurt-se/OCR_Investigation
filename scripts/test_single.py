"""Test single document through OCR to verify API connection and field mapping."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, Config
from src.models.document_intelligence import DocumentIntelligenceModel


def get_test_file(file_arg: str | None, config: Config) -> Path:
    """Get the file path to test.
    
    Args:
        file_arg: User-provided file path or None
        config: Configuration object
        
    Returns:
        Path to the test file
    """
    if file_arg:
        path = Path(file_arg)
        if not path.is_absolute():
            path = config.dataset_path / path
        if not path.exists():
            # Try as relative to project root
            path = project_root / file_arg
        return path
    
    # Find first document in dataset
    extensions = [".pdf", ".png", ".jpeg", ".jpg"]
    for ext in extensions:
        files = list(config.dataset_path.glob(f"*{ext}"))
        if files:
            return sorted(files)[0]
    
    raise FileNotFoundError("No documents found in dataset directory")


def load_ground_truth_for_file(filename: str, config: Config) -> dict | None:
    """Load ground truth entry for a specific file.
    
    Args:
        filename: Name of the file to look up
        config: Configuration object
        
    Returns:
        Ground truth dict or None if not found
    """
    with open(config.ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    
    for entry in ground_truth:
        if entry.get("filename") == filename:
            return entry
    
    return None


def display_comparison(ocr_result: dict, ground_truth: dict | None) -> float:
    """Display side-by-side comparison of OCR result and ground truth.
    
    Args:
        ocr_result: OCR extraction result
        ground_truth: Ground truth data or None
        
    Returns:
        Overall accuracy percentage
    """
    print(f"\nProcessing: {ocr_result['filename']}")
    print("=" * 60)
    
    if ground_truth is None:
        print("\nWARNING: No ground truth found for this file")
        print("\nOCR Results:")
        print(json.dumps(ocr_result, indent=2))
        return 0.0
    
    total_fields = 0
    matched_fields = 0
    
    # Compare invoice details
    print("\nINVOICE DETAILS:")
    print(f"  {'Field':<20} | {'OCR Result':<25} | {'Ground Truth':<25} | Match")
    print(f"  {'-'*20} | {'-'*25} | {'-'*25} | {'-'*5}")
    
    invoice_fields = [
        ("client_name", "invoice"),
        ("client_address", "invoice"),
        ("seller_name", "invoice"),
        ("seller_address", "invoice"),
        ("invoice_number", "invoice"),
        ("invoice_date", "invoice"),
        ("due_date", "invoice"),
    ]
    
    for field_name, section in invoice_fields:
        ocr_value = ocr_result.get(section, {}).get(field_name, "")
        truth_value = ground_truth.get(section, {}).get(field_name, "")
        
        # Normalize for comparison
        ocr_norm = str(ocr_value).strip().lower()
        truth_norm = str(truth_value).strip().lower()
        
        match = ocr_norm == truth_norm
        match_str = "OK" if match else "MISS"
        
        total_fields += 1
        if match:
            matched_fields += 1
        
        # Truncate long values for display
        ocr_display = str(ocr_value)[:25]
        truth_display = str(truth_value)[:25]
        
        print(f"  {field_name:<20} | {ocr_display:<25} | {truth_display:<25} | {match_str}")
    
    # Compare line items
    ocr_items = ocr_result.get("items", [])
    truth_items = ground_truth.get("items", [])
    
    print(f"\nLINE ITEMS: {len(ocr_items)} extracted, {len(truth_items)} expected")
    
    for i, (ocr_item, truth_item) in enumerate(zip(ocr_items, truth_items), 1):
        desc_match = ocr_item.get("description", "").strip().lower() == truth_item.get("description", "").strip().lower()
        total_match = normalize_number(ocr_item.get("total_price", "")) == normalize_number(truth_item.get("total_price", ""))
        
        total_fields += 2
        if desc_match:
            matched_fields += 1
        if total_match:
            matched_fields += 1
        
        desc_preview = ocr_item.get("description", "")[:40]
        match_str = "OK" if (desc_match and total_match) else "PARTIAL" if (desc_match or total_match) else "MISS"
        print(f"  [{i}] {desc_preview:<40} | {match_str}")
    
    # Handle mismatched counts
    if len(ocr_items) != len(truth_items):
        diff = abs(len(ocr_items) - len(truth_items))
        total_fields += diff * 2  # Account for missing items
    
    # Compare totals
    print("\nTOTALS:")
    subtotal_fields = ["subtotal", "tax", "discount", "total"]
    
    for field in subtotal_fields:
        ocr_value = ocr_result.get("subtotal", {}).get(field, "")
        truth_value = ground_truth.get("subtotal", {}).get(field, "")
        
        match = normalize_number(ocr_value) == normalize_number(truth_value)
        match_str = "OK" if match else "MISS"
        
        total_fields += 1
        if match:
            matched_fields += 1
        
        print(f"  {field}: {ocr_value} vs {truth_value} | {match_str}")
    
    # Calculate and display overall accuracy
    accuracy = (matched_fields / total_fields * 100) if total_fields > 0 else 0
    print(f"\nOverall: {accuracy:.1f}% field accuracy ({matched_fields}/{total_fields} fields)")
    
    return accuracy


def normalize_number(value: str) -> float | None:
    """Normalize a number string for comparison.
    
    Args:
        value: String that may contain a number
        
    Returns:
        Float value or None if not parseable
    """
    if not value:
        return None
    
    try:
        # Remove spaces, commas, and currency symbols
        cleaned = str(value).replace(" ", "").replace(",", "").replace("$", "").replace("€", "")
        return round(float(cleaned), 2)
    except (ValueError, TypeError):
        return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test OCR on a single document"
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Document to process (default: first file in dataset)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print raw API response"
    )
    args = parser.parse_args()
    
    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config()
        
        # Initialize model
        print("Initializing Document Intelligence model...")
        model = DocumentIntelligenceModel(config)
        
        # Get test file
        file_path = get_test_file(args.file, config)
        print(f"Test file: {file_path}")
        
        if not file_path.exists():
            print(f"ERROR: File not found: {file_path}")
            sys.exit(1)
        
        # Process document
        print("Calling Azure Document Intelligence API...")
        result = model.process_document(file_path)
        
        # Load ground truth
        ground_truth = load_ground_truth_for_file(file_path.name, config)
        
        # Display comparison
        accuracy = display_comparison(result, ground_truth)
        
        # Show raw response if verbose
        if args.verbose:
            print("\n" + "=" * 60)
            print("RAW API RESPONSE:")
            print("=" * 60)
            raw = model.get_raw_response(file_path)
            print(json.dumps(raw, indent=2, default=str))
        
        # Exit with appropriate code
        sys.exit(0 if accuracy >= 50 else 1)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
