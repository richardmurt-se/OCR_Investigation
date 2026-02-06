"""Configuration module for OCR Investigation project."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import os


# Pricing configuration for different OCR models (cost per page in USD)
MODEL_PRICING: Dict[str, Dict[str, Any]] = {
    "document_intelligence": {
        "cost_per_page": 0.0015,  # $1.50 per 1000 pages (prebuilt invoice model)
        "currency": "USD",
        "pricing_url": "https://azure.microsoft.com/pricing/details/ai-document-intelligence/",
    },
    "paddle_ocr": {
        "cost_per_page": 0.0,  # Free / open-source
        "currency": "USD",
    },
    "glm-ocr": {
        "cost_per_page": 0.0,  # Free / open-source
        "currency": "USD",
    },
    "deepseek-ocr-2": {
        "cost_per_page": 0.0,  # Free / open-source
        "currency": "USD",
    },
}


def get_model_pricing(model_name: str) -> Dict[str, Any]:
    """Get pricing information for a model.
    
    Args:
        model_name: Name of the OCR model
        
    Returns:
        Dictionary with cost_per_page and currency
    """
    return MODEL_PRICING.get(model_name, {"cost_per_page": 0.0, "currency": "USD"})


@dataclass
class Config:
    """Configuration settings for the OCR investigation project."""
    azure_endpoint: str
    azure_key: str
    dataset_path: Path
    results_path: Path
    reports_path: Path
    ground_truth_path: Path
    
    def get_model_results_path(self, model_name: str) -> Path:
        """Get the results directory for a specific model."""
        path = self.results_path / model_name
        path.mkdir(parents=True, exist_ok=True)
        return path


def get_project_root() -> Path:
    """Find project root by looking for ground_truth.json."""
    current = Path(__file__).resolve().parent
    
    # Walk up the directory tree looking for ground_truth.json
    for _ in range(10):  # Limit search depth
        if (current / "ground_truth.json").exists():
            return current
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    raise RuntimeError(
        "Could not find project root. "
        "Make sure ground_truth.json exists in the project directory."
    )


def load_config() -> Config:
    """Load configuration from environment variables.
    
    Returns:
        Config: Configuration object with all settings.
        
    Raises:
        RuntimeError: If required environment variables are missing.
    """
    # Load .env file
    project_root = get_project_root()
    env_path = project_root / ".env"
    
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # Try default locations
    
    # Get required environment variables
    azure_key = os.getenv("AZURE_FORM_RECOGNIZER_KEY")
    azure_endpoint = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT")
    
    # Validate required variables
    missing = []
    if not azure_key:
        missing.append("AZURE_FORM_RECOGNIZER_KEY")
    if not azure_endpoint:
        missing.append("AZURE_FORM_RECOGNIZER_ENDPOINT")
    
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Please create a .env file in {project_root} with these variables."
        )
    
    # Build paths
    dataset_path = project_root / "dataset"
    results_path = project_root / "results"
    reports_path = project_root / "reports"
    ground_truth_path = project_root / "ground_truth.json"
    
    # Ensure directories exist
    results_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    # Validate dataset exists
    if not dataset_path.exists():
        raise RuntimeError(
            f"Dataset directory not found: {dataset_path}. "
            "Please ensure the dataset folder exists."
        )
    
    return Config(
        azure_endpoint=azure_endpoint,
        azure_key=azure_key,
        dataset_path=dataset_path,
        results_path=results_path,
        reports_path=reports_path,
        ground_truth_path=ground_truth_path,
    )
