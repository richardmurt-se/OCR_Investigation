"""Abstract base class for OCR models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict


class InvoiceDetails(TypedDict):
    """Invoice header information."""
    client_name: str
    client_address: str
    seller_name: str
    seller_address: str
    invoice_number: str
    invoice_date: str
    due_date: str


class LineItem(TypedDict):
    """Individual line item from an invoice."""
    description: str
    quantity: str
    unit_price: str
    total_price: str
    category: Optional[str]


class SubtotalInfo(TypedDict):
    """Subtotal and total information."""
    subtotal: str
    tax: str
    discount: str
    total: str


class PaymentInstructions(TypedDict):
    """Payment instructions from the invoice."""
    due_date: str
    bank_name: str
    account_number: str
    payment_method: str


class InvoiceResult(TypedDict):
    """Complete invoice extraction result matching ground_truth structure."""
    filename: str
    invoice: InvoiceDetails
    items: List[LineItem]
    subtotal: SubtotalInfo
    payment_instructions: PaymentInstructions


class OCRModel(ABC):
    """Abstract base class for OCR models.
    
    All OCR model implementations should inherit from this class
    and implement the required methods.
    """
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return a unique identifier for this model.
        
        Returns:
            str: Model identifier (e.g., 'document_intelligence', 'tesseract')
        """
        pass
    
    @abstractmethod
    def process_document(self, file_path: Path) -> InvoiceResult:
        """Process a document and extract invoice data.
        
        Args:
            file_path: Path to the document (PDF, PNG, JPEG)
            
        Returns:
            InvoiceResult: Extracted data matching ground_truth structure
            
        Raises:
            Exception: If document processing fails
        """
        pass
    
    @abstractmethod
    def get_raw_response(self, file_path: Path) -> Any:
        """Get the raw API response for debugging purposes.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Raw response from the OCR service (format varies by model)
        """
        pass
    
    def process_documents(self, file_paths: List[Path]) -> List[InvoiceResult]:
        """Process multiple documents.
        
        Default implementation processes sequentially.
        Subclasses may override for batch processing.
        
        Args:
            file_paths: List of document paths
            
        Returns:
            List of InvoiceResult objects
        """
        results = []
        for path in file_paths:
            result = self.process_document(path)
            results.append(result)
        return results
