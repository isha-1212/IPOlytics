"""
Document loader for PDF files.
Extracts text and metadata from IPO documents.
"""
import os
from pathlib import Path
from typing import List, Dict, Any
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class DocumentLoader:
    """Load and extract text from PDF documents."""
    
    def __init__(self, documents_dir: str = "data/documents"):
        """
        Initialize document loader.
        
        Args:
            documents_dir: Path to directory containing PDF files
        """
        self.documents_dir = Path(documents_dir)
        if not self.documents_dir.exists():
            self.documents_dir.mkdir(parents=True, exist_ok=True)
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Load all PDF documents from the documents directory.
        
        Returns:
            List of documents with text, source, and page metadata
        """
        documents = []
        
        # Find all PDF files
        pdf_files = list(self.documents_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️  No PDF files found in {self.documents_dir}")
            return documents
        
        print(f"📄 Found {len(pdf_files)} PDF file(s)")
        
        for pdf_path in pdf_files:
            try:
                print(f"  Loading: {pdf_path.name}")
                docs_from_file = self._extract_from_pdf(pdf_path)
                documents.extend(docs_from_file)
                print(f"    ✓ Extracted {len(docs_from_file)} pages")
            except Exception as e:
                print(f"    ✗ Error: {str(e)}")
        
        return documents
    
    def _extract_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text and metadata from a single PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of documents with text and metadata
        """
        documents = []
        
        try:
            if not fitz:
                raise ImportError("PyMuPDF (fitz) is required for PDF extraction")
            
            doc = fitz.open(pdf_path)
            filename = pdf_path.name
            
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                
                if text.strip():  # Only add non-empty pages
                    documents.append({
                        "text": text,
                        "source": filename,
                        "page": page_num,
                    })
            
            doc.close()
        
        except Exception as e:
            import traceback
            print(f"    Full traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to extract from {pdf_path.name}: {str(e)}")
        
        return documents
    
    def load_single_document(self, pdf_filename: str) -> List[Dict[str, Any]]:
        """
        Load a single PDF document by filename.
        
        Args:
            pdf_filename: Name of PDF file
            
        Returns:
            List of documents with text and metadata
        """
        pdf_path = self.documents_dir / pdf_filename
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        return self._extract_from_pdf(pdf_path)
