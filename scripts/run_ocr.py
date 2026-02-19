"""Batch process all documents through OCR."""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, Config, get_model_pricing
from src.models.document_intelligence import DocumentIntelligenceModel
from src.evaluation.compare import compare_all, aggregate_by_effect, aggregate_by_format, get_worst_documents, get_best_documents
from src.reporting.html_report import generate_html_report


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Silence verbose Azure SDK HTTP logging
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

# Rich console for pretty output
console = Console()


def create_progress_panel(
    model_name: str,
    current_doc: str,
    processed: int,
    total: int,
    errors: int,
    elapsed: float,
    estimated_cost: float,
    progress: Progress
) -> Panel:
    """Create a rich panel showing processing progress."""
    # Calculate speed
    speed = (processed / elapsed * 60) if elapsed > 0 and processed > 0 else 0
    
    # Build stats table
    stats = Table.grid(padding=(0, 2))
    stats.add_column(style="cyan", justify="right")
    stats.add_column(style="white")
    
    stats.add_row("Current:", current_doc if current_doc else "Starting...")
    stats.add_row("Elapsed:", f"{elapsed:.1f}s")
    stats.add_row("Processed:", f"{processed} / {total} documents")
    stats.add_row("Errors:", f"[red]{errors}[/red]" if errors > 0 else "0")
    stats.add_row("Est. Cost:", f"${estimated_cost:.4f}")
    stats.add_row("Speed:", f"{speed:.1f} docs/min")
    
    # Combine progress bar and stats
    content = Table.grid(padding=1)
    content.add_column()
    content.add_row(progress)
    content.add_row(stats)
    
    return Panel(
        content,
        title=f"[bold blue]OCR Processing - {model_name}[/bold blue]",
        border_style="blue"
    )


def get_documents(config: Config) -> list[Path]:
    """Get all documents from the dataset directory.
    
    Args:
        config: Configuration object
        
    Returns:
        List of document paths
    """
    extensions = [".pdf", ".png", ".jpeg", ".jpg"]
    documents = []
    
    for ext in extensions:
        documents.extend(config.dataset_path.glob(f"*{ext}"))
    
    return sorted(documents)


def get_result_filename(doc: Path) -> str:
    """Get the result filename for a document (includes extension to avoid collisions).
    
    Args:
        doc: Document path
        
    Returns:
        Result filename (e.g., '001_clean_standard_jpeg' for '001_clean_standard.jpeg')
    """
    # Replace dots with underscores to create unique filenames
    # e.g., "001_clean_standard.jpeg" -> "001_clean_standard_jpeg"
    ext = doc.suffix.lstrip(".")  # Get extension without dot
    return f"{doc.stem}_{ext}"


def filter_unprocessed(
    documents: list[Path],
    config: Config,
    model_name: str
) -> list[Path]:
    """Filter out already-processed documents.
    
    Args:
        documents: List of document paths
        config: Configuration object
        model_name: Name of the OCR model
        
    Returns:
        List of unprocessed document paths
    """
    results_dir = config.get_model_results_path(model_name)
    unprocessed = []
    
    for doc in documents:
        result_file = results_dir / f"{get_result_filename(doc)}.json"
        if not result_file.exists():
            unprocessed.append(doc)
    
    return unprocessed


def save_individual_result(
    result: dict,
    result_name: str,
    config: Config,
    model_name: str
) -> None:
    """Save individual document result.
    
    Args:
        result: OCR result dictionary
        result_name: Result filename (from get_result_filename)
        config: Configuration object
        model_name: Name of the OCR model
    """
    results_dir = config.get_model_results_path(model_name)
    result_file = results_dir / f"{result_name}.json"
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def save_raw_response(
    raw_response: dict,
    result_name: str,
    config: Config,
    model_name: str
) -> None:
    """Save raw API response.
    
    Args:
        raw_response: Raw API response dictionary
        result_name: Result filename (from get_result_filename)
        config: Configuration object
        model_name: Name of the OCR model
    """
    results_dir = config.get_model_results_path(model_name)
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    raw_file = raw_dir / f"{result_name}_raw.json"
    
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(raw_response, f, indent=2, ensure_ascii=False)


def save_all_results(
    results: list[dict],
    config: Config,
    model_name: str
) -> None:
    """Save combined results file.
    
    Args:
        results: List of all OCR results
        config: Configuration object
        model_name: Name of the OCR model
    """
    results_dir = config.get_model_results_path(model_name)
    combined_file = results_dir / "all_results.json"
    
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved combined results to {combined_file}")


def log_error(doc: Path, error: Exception, config: Config, model_name: str) -> None:
    """Log processing error.
    
    Args:
        doc: Document path that failed
        error: Exception that occurred
        config: Configuration object
        model_name: Name of the OCR model
    """
    results_dir = config.get_model_results_path(model_name)
    error_log = results_dir / "errors.log"
    
    timestamp = datetime.now().isoformat()
    
    with open(error_log, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | {doc.name} | {type(error).__name__}: {error}\n")
    
    logger.error(f"Error processing {doc.name}: {error}")


def round_floats(obj, decimals: int = 3):
    """Recursively round all float values in a data structure."""
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(item, decimals) for item in obj]
    return obj


def generate_combined_test_results(config: Config, model_name: str) -> None:
    """Generate combined_results.json from raw responses in test_results/.
    
    Reads each raw JSON file and produces a combined array with document_id,
    source_filename, extension, page_count, and full text content.
    
    Args:
        config: Configuration object (with test paths)
        model_name: Name of the OCR model
    """
    results_dir = config.get_model_results_path(model_name)
    raw_dir = results_dir / "raw"
    
    if not raw_dir.exists():
        logger.warning(f"No raw directory found at {raw_dir}")
        return
    
    combined = []
    
    for raw_file in sorted(raw_dir.glob("*_raw.json")):
        try:
            with open(raw_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            
            # Extract document_id from filename: e.g., "169574963_pdf_raw.json" -> "169574963"
            # The stem without _raw is "169574963_pdf", split off the extension part
            result_key = raw_file.stem.replace("_raw", "")  # "169574963_pdf"
            parts = result_key.rsplit("_", 1)  # ["169574963", "pdf"]
            document_id = parts[0] if len(parts) > 1 else result_key
            extension = parts[1] if len(parts) > 1 else "unknown"
            
            source_filename = f"{result_key}.json"
            page_count = len(raw.get("pages", []))
            text = raw.get("content", "")
            
            combined.append({
                "document_id": document_id,
                "source_filename": source_filename,
                "extension": extension,
                "page_count": page_count,
                "text": text,
            })
        except Exception as e:
            logger.error(f"Error reading {raw_file.name}: {e}")
    
    # Save to test_results/combined_results.json
    output_path = config.results_path / "combined_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    
    console.print(f"[bold green]Combined results saved to: {output_path}[/bold green]")
    console.print(f"  {len(combined)} documents, total pages: {sum(d['page_count'] for d in combined)}")


def generate_comparison_report(config: Config, model_name: str) -> None:
    """Generate comparison report after OCR processing.
    
    Args:
        config: Configuration object
        model_name: Name of the OCR model
    """
    try:
        comparisons = compare_all(config, model_name)
        
        if not comparisons:
            logger.warning("No documents matched ground truth for comparison report")
            return
        
        # Calculate accuracy summary metrics
        extraction_accs = [c.extraction_accuracy for c in comparisons]
        schema_accs = [c.schema_accuracy for c in comparisons]
        overall_accs = [c.overall_accuracy for c in comparisons]
        
        summary_metrics = {
            "extraction_accuracy": sum(extraction_accs) / len(extraction_accs),
            "schema_accuracy": sum(schema_accs) / len(schema_accs),
            "overall_accuracy": sum(overall_accs) / len(overall_accs),
        }
        
        # Calculate performance summary metrics
        processing_times = [c.processing_time_seconds for c in comparisons]
        confidences = [c.avg_confidence for c in comparisons if c.avg_confidence > 0]
        total_pages = sum(c.page_count for c in comparisons)
        total_cost = sum(c.estimated_cost_usd for c in comparisons)
        total_time = sum(processing_times)
        
        # Filter out zero times for accurate averages (may not have been captured)
        non_zero_times = [t for t in processing_times if t > 0]
        avg_time = sum(non_zero_times) / len(non_zero_times) if non_zero_times else 0
        
        pricing = get_model_pricing(model_name)
        
        performance_metrics = {
            "total_processing_time_seconds": total_time,
            "avg_processing_time_seconds": avg_time,
            "min_processing_time_seconds": min(non_zero_times) if non_zero_times else 0,
            "max_processing_time_seconds": max(non_zero_times) if non_zero_times else 0,
            "throughput_docs_per_min": (60 / avg_time) if avg_time > 0 else 0,
            "total_pages": total_pages,
            "total_cost_usd": total_cost,
            "cost_per_page_usd": pricing.get("cost_per_page", 0),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
        }
        
        # Generate report
        detailed_report = {
            "generated_at": datetime.now().isoformat(),
            "model_name": model_name,
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
        
        # Round all float values
        detailed_report = round_floats(detailed_report, decimals=3)
        
        # Save JSON report
        report_file = config.reports_path / f"{model_name}_comparison_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(detailed_report, f, indent=2, ensure_ascii=False)
        
        # Generate HTML report
        html_file = config.reports_path / f"{model_name}_comparison_report.html"
        generate_html_report(detailed_report, html_file)
        
        # Display accuracy results table
        accuracy_table = Table(title="Comparison Results", border_style="cyan")
        accuracy_table.add_column("Metric", style="cyan")
        accuracy_table.add_column("Value", style="white")
        
        accuracy_table.add_row("Documents Compared", f"{len(comparisons)}")
        
        # Color-code accuracy values
        ext_acc = summary_metrics['extraction_accuracy'] * 100
        ext_color = "green" if ext_acc >= 90 else ("yellow" if ext_acc >= 70 else "red")
        accuracy_table.add_row("Extraction Accuracy", f"[{ext_color}]{ext_acc:.1f}%[/{ext_color}]")
        
        sch_acc = summary_metrics['schema_accuracy'] * 100
        sch_color = "green" if sch_acc >= 90 else ("yellow" if sch_acc >= 70 else "red")
        accuracy_table.add_row("Schema Accuracy", f"[{sch_color}]{sch_acc:.1f}%[/{sch_color}]")
        
        ovr_acc = summary_metrics['overall_accuracy'] * 100
        ovr_color = "green" if ovr_acc >= 90 else ("yellow" if ovr_acc >= 70 else "red")
        accuracy_table.add_row("Overall Accuracy", f"[{ovr_color}]{ovr_acc:.1f}%[/{ovr_color}]")
        
        if avg_time > 0:
            accuracy_table.add_row("Avg Processing Time", f"{avg_time:.3f}s/doc")
        if total_cost > 0:
            accuracy_table.add_row("Total Cost", f"${total_cost:.4f} ({total_pages} pages)")
        if confidences:
            conf_pct = performance_metrics['avg_confidence'] * 100
            accuracy_table.add_row("Avg Confidence", f"{conf_pct:.1f}%")
        
        console.print(accuracy_table)
        
        # Report file paths
        console.print()
        console.print("[bold]Reports:[/bold]")
        console.print(f"  JSON: {report_file}")
        console.print(f"  HTML: {html_file}")
        
    except Exception as e:
        logger.error(f"Error generating comparison report: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process all documents through OCR"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all documents, even if results already exist"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only N documents"
    )
    parser.add_argument(
        "--model",
        default="document_intelligence",
        choices=["document_intelligence"],
        help="OCR model to use (default: document_intelligence)"
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip saving raw API responses (raw is saved by default)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process test_dataset/ and save to test_results/ (with combined JSON output)"
    )
    args = parser.parse_args()
    
    try:
        # Load configuration
        console.print("[dim]Loading configuration...[/dim]")
        config = load_config()
        
        # Override paths for test mode
        if args.test:
            config.dataset_path = project_root / "test_dataset"
            config.results_path = project_root / "test_results"
            config.results_path.mkdir(parents=True, exist_ok=True)
            console.print("[yellow]Test mode: reading from test_dataset/, saving to test_results/[/yellow]")
        
        # Initialize model
        console.print(f"[dim]Initializing {args.model} model...[/dim]")
        if args.model == "document_intelligence":
            model = DocumentIntelligenceModel(config)
        else:
            raise ValueError(f"Unknown model: {args.model}")
        
        model_name = model.get_model_name()
        
        # Get documents
        documents = get_documents(config)
        console.print(f"Found [bold]{len(documents)}[/bold] documents in dataset")
        
        # Skip already-processed documents by default (unless --force)
        if not args.force:
            documents = filter_unprocessed(documents, config, model_name)
            console.print(f"[bold]{len(documents)}[/bold] documents remaining to process")
        else:
            console.print("[yellow]Force mode: reprocessing all documents[/yellow]")
        
        # Apply limit
        if args.limit:
            documents = documents[:args.limit]
            console.print(f"Limited to [bold]{len(documents)}[/bold] documents")
        
        if not documents:
            console.print("[green]No documents to process - all up to date![/green]")
            return
        
        # Process documents with rich live display
        results = []
        errors = 0
        total_processing_time = 0.0
        pricing = get_model_pricing(model_name)
        
        # Create progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        task = progress.add_task("Processing", total=len(documents))
        
        loop_start_time = time.time()
        current_doc_name = ""
        
        with Live(
            create_progress_panel(
                model_name, "", 0, len(documents), 0, 0, 0, progress
            ),
            console=console,
            refresh_per_second=4
        ) as live:
            for i, doc in enumerate(documents):
                current_doc_name = doc.name
                
                # Update live display
                elapsed = time.time() - loop_start_time
                est_cost = len(results) * pricing.get("cost_per_page", 0)
                live.update(create_progress_panel(
                    model_name, current_doc_name, len(results), len(documents),
                    errors, elapsed, est_cost, progress
                ))
                
                try:
                    # Capture timing around OCR processing
                    start_time = time.time()
                    result = model.process_document(doc)
                    elapsed_seconds = round(time.time() - start_time, 3)
                    
                    # Add performance metadata to result
                    result["processing_time_seconds"] = elapsed_seconds
                    result["file_size_bytes"] = doc.stat().st_size
                    
                    total_processing_time += elapsed_seconds
                    
                    result_name = get_result_filename(doc)
                    save_individual_result(result, result_name, config, model_name)
                    results.append(result)
                    
                    # Save raw response by default (unless --no-raw specified)
                    if not args.no_raw:
                        raw_response = model.get_last_raw_response_dict()
                        save_raw_response(raw_response, result_name, config, model_name)
                    
                except Exception as e:
                    log_error(doc, e, config, model_name)
                    errors += 1
                
                # Update progress bar
                progress.update(task, advance=1)
            
            # Final update
            elapsed = time.time() - loop_start_time
            est_cost = len(results) * pricing.get("cost_per_page", 0)
            live.update(create_progress_panel(
                model_name, "Complete!", len(results), len(documents),
                errors, elapsed, est_cost, progress
            ))
        
        # Load existing results (unless --force which means we're replacing everything)
        if not args.force:
            results_dir = config.get_model_results_path(model_name)
            for result_file in results_dir.glob("*.json"):
                if result_file.name != "all_results.json":
                    with open(result_file, "r", encoding="utf-8") as f:
                        existing_result = json.load(f)
                        # Only add if not already in results
                        if not any(r["filename"] == existing_result["filename"] for r in results):
                            results.append(existing_result)
        
        # Save combined results
        save_all_results(results, config, model_name)
        
        # Calculate processing statistics
        successful_count = len(results) - errors
        avg_time = total_processing_time / successful_count if successful_count > 0 else 0
        throughput = (60 / avg_time) if avg_time > 0 else 0
        
        # Note: pricing was already fetched above for the live display
        estimated_cost = successful_count * pricing.get("cost_per_page", 0)
        
        # Summary table
        console.print()
        summary_table = Table(title="Processing Results", border_style="green")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Documents Processed", f"{successful_count}")
        summary_table.add_row("Errors", f"[red]{errors}[/red]" if errors > 0 else "0")
        summary_table.add_row("Total Time", f"{total_processing_time:.3f}s")
        summary_table.add_row("Avg Time/Doc", f"{avg_time:.3f}s")
        summary_table.add_row("Throughput", f"{throughput:.1f} docs/min")
        summary_table.add_row("Estimated Cost", f"${estimated_cost:.4f}")
        
        console.print(summary_table)
        
        # Output paths
        console.print()
        console.print("[bold]Output Files:[/bold]")
        console.print(f"  Results: {config.get_model_results_path(model_name)}")
        if not args.no_raw:
            console.print(f"  Raw:     {config.get_model_results_path(model_name) / 'raw'}")
        
        # Auto-generate comparison report
        if results:
            console.print()
            console.print("[bold cyan]Generating comparison report...[/bold cyan]")
            generate_comparison_report(config, model_name)
        
        # Generate combined test results JSON in test mode
        if args.test and not args.no_raw:
            console.print()
            console.print("[bold cyan]Generating combined test results...[/bold cyan]")
            generate_combined_test_results(config, model_name)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
