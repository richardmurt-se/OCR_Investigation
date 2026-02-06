"""Azure Document Intelligence OCR model implementation."""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from .base import OCRModel, InvoiceResult, InvoiceDetails, LineItem, SubtotalInfo, PaymentInstructions
from ..config import Config


logger = logging.getLogger(__name__)


class DocumentIntelligenceModel(OCRModel):
    """Azure Document Intelligence implementation using prebuilt-invoice model."""
    
    def __init__(self, config: Config):
        """Initialize the Document Intelligence client.
        
        Args:
            config: Configuration object with Azure credentials
        """
        self.config = config
        self.client = DocumentAnalysisClient(
            endpoint=config.azure_endpoint,
            credential=AzureKeyCredential(config.azure_key)
        )
        self._last_raw_response: Optional[Any] = None
        
        # Retry configuration
        self.max_retries = 3
        self.base_delay = 1.0  # seconds
    
    def get_model_name(self) -> str:
        """Return the model identifier."""
        return "document_intelligence"
    
    def process_document(self, file_path: Path) -> InvoiceResult:
        """Process a document through Azure Document Intelligence.
        
        Args:
            file_path: Path to the document (PDF, PNG, JPEG)
            
        Returns:
            InvoiceResult with extracted invoice data
        """
        result = self._analyze_with_retry(file_path)
        self._last_raw_response = result
        return self._parse_invoice_result(result, file_path.name)
    
    def get_raw_response(self, file_path: Path) -> Any:
        """Get the raw API response.
        
        If the document was just processed, returns cached response.
        Otherwise, processes the document again.
        """
        if self._last_raw_response is not None:
            return self._convert_to_dict(self._last_raw_response)
        
        result = self._analyze_with_retry(file_path)
        return self._convert_to_dict(result)
    
    def _analyze_with_retry(self, file_path: Path) -> Any:
        """Analyze document with exponential backoff retry.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Analysis result from Azure
            
        Raises:
            HttpResponseError: If all retries fail
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                with open(file_path, "rb") as f:
                    poller = self.client.begin_analyze_document(
                        "prebuilt-invoice",
                        document=f
                    )
                    return poller.result()
                    
            except HttpResponseError as e:
                last_error = e
                
                # Check if it's a rate limit error (429)
                if hasattr(e, 'status_code') and e.status_code == 429:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(
                        f"Rate limited on {file_path.name}, "
                        f"retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                else:
                    # For other errors, raise immediately
                    raise
        
        # All retries exhausted
        raise last_error
    
    def _parse_invoice_result(self, result: Any, filename: str) -> InvoiceResult:
        """Parse Azure response into ground_truth format.
        
        Args:
            result: Azure DocumentAnalysisResult
            filename: Original filename
            
        Returns:
            InvoiceResult matching ground_truth structure
        """
        # Get the first invoice (assuming one per document)
        invoice_doc = result.documents[0] if result.documents else None
        fields = invoice_doc.fields if invoice_doc else {}
        
        # Extract invoice details
        invoice_details: InvoiceDetails = {
            "client_name": self._extract_field_value(fields.get("CustomerName")),
            "client_address": self._extract_field_value(fields.get("CustomerAddress")),
            "seller_name": self._extract_field_value(fields.get("VendorName")),
            "seller_address": self._extract_field_value(fields.get("VendorAddress")),
            "invoice_number": self._extract_field_value(fields.get("InvoiceId")),
            "invoice_date": self._format_date(fields.get("InvoiceDate")),
            "due_date": self._format_date(fields.get("DueDate")),
        }
        
        # Extract line items
        items: list[LineItem] = []
        items_field = fields.get("Items")
        if items_field and items_field.value:
            for item in items_field.value:
                item_fields = item.value if item.value else {}
                line_item: LineItem = {
                    "description": self._extract_field_value(item_fields.get("Description")),
                    "quantity": self._format_number(item_fields.get("Quantity")),
                    "unit_price": self._format_currency(item_fields.get("UnitPrice")),
                    "total_price": self._format_currency(item_fields.get("Amount")),
                    "category": None,  # Not provided by Azure
                }
                items.append(line_item)
        
        # Extract subtotal info
        subtotal_info: SubtotalInfo = {
            "subtotal": self._format_currency(fields.get("SubTotal")),
            "tax": self._format_currency(fields.get("TotalTax")),
            "discount": self._format_currency(fields.get("TotalDiscount")),
            "total": self._format_currency(fields.get("InvoiceTotal")),
        }
        
        # Extract payment instructions
        payment_info: PaymentInstructions = {
            "due_date": self._format_date(fields.get("DueDate")),
            "bank_name": self._extract_field_value(fields.get("PaymentDetails")),
            "account_number": "",  # Not directly available
            "payment_method": "",  # Not directly available
        }
        
        # Try to extract bank details from payment terms if available
        payment_term = fields.get("PaymentTerm")
        if payment_term:
            payment_info["bank_name"] = self._extract_field_value(payment_term)
        
        return InvoiceResult(
            filename=filename,
            invoice=invoice_details,
            items=items,
            subtotal=subtotal_info,
            payment_instructions=payment_info,
        )
    
    def _extract_field_value(self, field: Any) -> str:
        """Safely extract string value from an Azure field.
        
        Args:
            field: Azure DocumentField or None
            
        Returns:
            String value or empty string if not available
        """
        if field is None:
            return ""
        
        if hasattr(field, 'content') and field.content:
            return str(field.content)
        
        if hasattr(field, 'value') and field.value is not None:
            return str(field.value)
        
        return ""
    
    def _format_date(self, field: Any) -> str:
        """Format a date field to match ground_truth format (DD/MM/YYYY).
        
        Args:
            field: Azure date field
            
        Returns:
            Formatted date string or empty string
        """
        if field is None:
            return ""
        
        value = field.value
        if value is None:
            # Try content as fallback
            if hasattr(field, 'content') and field.content:
                return str(field.content)
            return ""
        
        # If it's a date object, format it
        if hasattr(value, 'strftime'):
            return value.strftime("%d/%m/%Y")
        
        return str(value)
    
    def _format_number(self, field: Any) -> str:
        """Format a number field.
        
        Args:
            field: Azure number field
            
        Returns:
            Formatted number string (e.g., "5.00")
        """
        if field is None:
            return ""
        
        value = field.value
        if value is None:
            if hasattr(field, 'content') and field.content:
                return str(field.content)
            return ""
        
        # Format as decimal with 2 places
        try:
            return f"{float(value):.2f}"
        except (ValueError, TypeError):
            return str(value)
    
    def _format_currency(self, field: Any) -> str:
        """Format a currency field to match ground_truth style.
        
        Ground truth uses format like "1 032.82" (space as thousands separator).
        
        Args:
            field: Azure currency field
            
        Returns:
            Formatted currency string
        """
        if field is None:
            return ""
        
        # Try to get the numeric value
        value = None
        if hasattr(field, 'value'):
            if hasattr(field.value, 'amount'):
                value = field.value.amount
            else:
                value = field.value
        
        if value is None:
            # Fallback to content
            if hasattr(field, 'content') and field.content:
                return str(field.content)
            return ""
        
        try:
            num = float(value)
            # Format with space as thousands separator
            formatted = f"{num:,.2f}"
            # Replace comma with space for thousands separator
            formatted = formatted.replace(",", " ")
            return formatted
        except (ValueError, TypeError):
            return str(value)
    
    def get_last_raw_response_dict(self) -> Dict[str, Any]:
        """Get the last raw response as a dictionary.
        
        Returns:
            Dictionary representation of the last API response, or empty dict if none.
        """
        if self._last_raw_response is None:
            return {}
        return self._convert_to_dict(self._last_raw_response)
    
    def _convert_to_dict(self, result: Any) -> Dict[str, Any]:
        """Convert Azure response to a comprehensive JSON-serializable dictionary.
        
        Includes all available data: confidence scores, bounding boxes, pages, etc.
        
        Args:
            result: Azure DocumentAnalysisResult
            
        Returns:
            Dictionary representation of the result
        """
        if result is None:
            return {}
        
        output = {
            "model_id": result.model_id if hasattr(result, 'model_id') else None,
            "api_version": result.api_version if hasattr(result, 'api_version') else None,
            "content": result.content if hasattr(result, 'content') else None,
            "pages": [],
            "tables": [],
            "documents": [],
        }
        
        # Extract page information
        if hasattr(result, 'pages') and result.pages:
            for page in result.pages:
                page_dict = {
                    "page_number": page.page_number if hasattr(page, 'page_number') else None,
                    "width": page.width if hasattr(page, 'width') else None,
                    "height": page.height if hasattr(page, 'height') else None,
                    "unit": page.unit if hasattr(page, 'unit') else None,
                    "angle": page.angle if hasattr(page, 'angle') else None,
                    "words": [],
                    "lines": [],
                }
                
                # Extract words with bounding boxes
                if hasattr(page, 'words') and page.words:
                    for word in page.words:
                        word_dict = {
                            "content": word.content if hasattr(word, 'content') else None,
                            "confidence": word.confidence if hasattr(word, 'confidence') else None,
                            "polygon": self._extract_polygon(word),
                        }
                        page_dict["words"].append(word_dict)
                
                # Extract lines
                if hasattr(page, 'lines') and page.lines:
                    for line in page.lines:
                        line_dict = {
                            "content": line.content if hasattr(line, 'content') else None,
                            "polygon": self._extract_polygon(line),
                        }
                        page_dict["lines"].append(line_dict)
                
                output["pages"].append(page_dict)
        
        # Extract tables
        if hasattr(result, 'tables') and result.tables:
            for table in result.tables:
                table_dict = {
                    "row_count": table.row_count if hasattr(table, 'row_count') else None,
                    "column_count": table.column_count if hasattr(table, 'column_count') else None,
                    "cells": [],
                }
                
                if hasattr(table, 'cells') and table.cells:
                    for cell in table.cells:
                        cell_dict = {
                            "row_index": cell.row_index if hasattr(cell, 'row_index') else None,
                            "column_index": cell.column_index if hasattr(cell, 'column_index') else None,
                            "content": cell.content if hasattr(cell, 'content') else None,
                            "kind": cell.kind if hasattr(cell, 'kind') else None,
                        }
                        table_dict["cells"].append(cell_dict)
                
                output["tables"].append(table_dict)
        
        # Extract documents (invoices)
        if hasattr(result, 'documents') and result.documents:
            for doc in result.documents:
                doc_dict = {
                    "doc_type": doc.doc_type if hasattr(doc, 'doc_type') else None,
                    "confidence": doc.confidence if hasattr(doc, 'confidence') else None,
                    "fields": {},
                }
                
                if hasattr(doc, 'fields') and doc.fields:
                    for name, field in doc.fields.items():
                        doc_dict["fields"][name] = self._extract_field_full(field)
                
                output["documents"].append(doc_dict)
        
        return output
    
    def _extract_polygon(self, obj: Any) -> list | None:
        """Extract polygon/bounding box coordinates from an object.
        
        Args:
            obj: Object that may have polygon attribute
            
        Returns:
            List of coordinate points or None
        """
        if not hasattr(obj, 'polygon') or obj.polygon is None:
            return None
        
        try:
            return [{"x": p.x, "y": p.y} for p in obj.polygon]
        except (AttributeError, TypeError):
            return None
    
    def _extract_field_full(self, field: Any) -> Dict[str, Any]:
        """Extract full field information including confidence and bounding regions.
        
        Args:
            field: Azure DocumentField
            
        Returns:
            Dictionary with complete field information
        """
        if field is None:
            return {"value": None, "content": None, "confidence": None}
        
        field_dict = {
            "value": None,
            "value_type": field.value_type if hasattr(field, 'value_type') else None,
            "content": field.content if hasattr(field, 'content') else None,
            "confidence": field.confidence if hasattr(field, 'confidence') else None,
            "bounding_regions": [],
        }
        
        # Handle value based on type
        if hasattr(field, 'value') and field.value is not None:
            value = field.value
            # Handle different value types
            if hasattr(value, 'amount'):  # CurrencyValue
                field_dict["value"] = {
                    "amount": value.amount,
                    "symbol": value.symbol if hasattr(value, 'symbol') else None,
                    "code": value.code if hasattr(value, 'code') else None,
                }
            elif hasattr(value, 'strftime'):  # Date
                field_dict["value"] = value.isoformat()
            elif isinstance(value, list):  # Array of items
                field_dict["value"] = [
                    self._extract_field_full(item) if hasattr(item, 'value') else str(item)
                    for item in value
                ]
            elif hasattr(value, 'value'):  # Nested document field
                field_dict["value"] = {
                    k: self._extract_field_full(v) 
                    for k, v in value.items()
                } if hasattr(value, 'items') else str(value)
            else:
                field_dict["value"] = str(value)
        
        # Extract bounding regions
        if hasattr(field, 'bounding_regions') and field.bounding_regions:
            for region in field.bounding_regions:
                region_dict = {
                    "page_number": region.page_number if hasattr(region, 'page_number') else None,
                    "polygon": self._extract_polygon(region),
                }
                field_dict["bounding_regions"].append(region_dict)
        
        return field_dict
