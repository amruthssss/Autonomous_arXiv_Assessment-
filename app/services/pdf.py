"""PDF downloading, extraction, and deterministic text chunking."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import BinaryIO

from app.state import DocumentChunk, Paper


class PdfService:
    def __init__(self, pdf_dir: Path, *, timeout: float = 60.0, max_bytes: int = 50_000_000) -> None:
        self.pdf_dir = pdf_dir
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_bytes = max_bytes

    def path_for(self, paper: Paper) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", paper.arxiv_id)
        return self.pdf_dir / f"{safe_id}.pdf"

    def download(self, paper: Paper) -> Path:
        path = self.path_for(paper)
        if path.exists() and self.validate_pdf(path):
            return path
        if not paper.pdf_url:
            raise ValueError(f"Paper {paper.arxiv_id} has no PDF URL")
        request = urllib.request.Request(
            paper.pdf_url, headers={"User-Agent": "arxiv-research-assistant/1.0"}
        )
        temporary = path.with_suffix(".pdf.part")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = getattr(response, "status", 200)
                if status and int(status) >= 400:
                    raise RuntimeError(f"PDF download returned HTTP {status}")
                headers = getattr(response, "headers", None) or {}
                content_type = str(headers.get("Content-Type", ""))
                if content_type and "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
                    raise RuntimeError(f"Expected a PDF response, got {content_type}")
                with temporary.open("wb") as output:
                    self._copy_limited(response, output)
            if not self.validate_pdf(temporary):
                raise RuntimeError("Downloaded content is not a valid PDF")
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return path

    def _copy_limited(self, response: BinaryIO, output: BinaryIO) -> None:
        total = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > self.max_bytes:
                raise RuntimeError(f"PDF exceeds the {self.max_bytes} byte limit")
            output.write(block)

    @staticmethod
    def validate_pdf(path: Path) -> bool:
        """Cheap validation before reuse; full parsing is performed by extract."""

        try:
            with path.open("rb") as stream:
                return stream.read(5) == b"%PDF-" and path.stat().st_size >= 5
        except OSError:
            return False

    @staticmethod
    def extract(path: Path) -> list[tuple[int, str]]:
        if not PdfService.validate_pdf(path):
            raise ValueError(f"Invalid or missing PDF: {path}")
        try:
            import pymupdf
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to parse PDFs") from exc
        pages: list[tuple[int, str]] = []
        try:
            document_context = pymupdf.open(path)
        except Exception as exc:
            raise ValueError(f"Could not open PDF {path}") from exc
        with document_context as document:
            if document.page_count == 0:
                raise ValueError(f"PDF has no pages: {path}")
            for number, page in enumerate(document, start=1):
                text = page.get_text("text")
                if text.strip():
                    pages.append((number, text))
        if not pages:
            raise ValueError(
                "Unable to reliably extract text from this PDF. "
                "The document may contain scanned pages or complex formatting."
            )
        return pages

    def chunks(self, paper: Paper, *, chunk_size: int = 1200, overlap: int = 180) -> list[DocumentChunk]:
        if chunk_size <= overlap:
            raise ValueError("chunk_size must be greater than overlap")
        output: list[DocumentChunk] = []
        for page, text in self.extract(self.path_for(paper)):
            clean = re.sub(r"\s+", " ", text).strip()
            section_match = re.match(
                r"^(?:\d+(?:\.\d+)*[\s.)-]+)?([A-Z][A-Za-z0-9][A-Za-z0-9 &:/-]{2,80})\s",
                clean,
            )
            section = section_match.group(1).strip() if section_match else ""
            start = 0
            while start < len(clean):
                end = min(len(clean), start + chunk_size)
                piece = clean[start:end].strip()
                if piece:
                    index = len(output)
                    output.append(
                        DocumentChunk(
                            chunk_id=f"{paper.arxiv_id}:{page}:{index}",
                            paper_id=paper.arxiv_id,
                            title=paper.title,
                            text=piece,
                            page=page,
                            start=start,
                            end=end,
                            section=section,
                        )
                    )
                if end == len(clean):
                    break
                start = end - overlap
        return output
