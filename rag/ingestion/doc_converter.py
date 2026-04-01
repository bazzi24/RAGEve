from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path



class Converter(str, Enum):
    LIBREOFFICE = "libreoffice"
    PYTHON_DOCX = "python-docx"  # fallback only: reads .doc as raw bytes / basic text
    CATDOC = "catdoc"
    ANTIWORD = "antiword"
    NONE = "none"


@dataclass
class ConversionResult:
    success: bool
    converter: Converter
    output_path: Path | None
    error: str | None
    message: str


def _find_executable(name: str) -> Path | None:
    found = shutil.which(name)
    if not found:
        return None
    return Path(found)


def _try_libreoffice(src: Path, tmp_dir: Path) -> ConversionResult:
    lo = _find_executable("soffice")
    if not lo:
        return ConversionResult(
            success=False,
            converter=Converter.LIBREOFFICE,
            output_path=None,
            error="soffice (LibreOffice) not found in PATH",
            message="Install LibreOffice to enable .doc conversion: https://www.libreoffice.org/download/download/",
        )

    out = tmp_dir / "converted"
    out.mkdir(exist_ok=True)

    try:
        result = subprocess.run(
            [
                str(lo),
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                str(out),
                str(src),
            ],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            return ConversionResult(
                success=False,
                converter=Converter.LIBREOFFICE,
                output_path=None,
                error=result.stderr.decode(errors="replace"),
                message="LibreOffice conversion failed. Check file is not corrupted.",
            )

        # LibreOffice writes output in same dir as source with new extension.
        # It may write alongside the source file.
        # Try to find the converted docx.
        converted = out / f"{src.stem}.docx"
        if not converted.exists():
            # LibreOffice sometimes writes to --outdir directly
            candidates = list(out.glob(f"{src.stem}*.docx"))
            if candidates:
                converted = candidates[0]

        if not converted.exists():
            return ConversionResult(
                success=False,
                converter=Converter.LIBREOFFICE,
                output_path=None,
                error="LibreOffice did not produce expected output file",
                message="LibreOffice conversion produced no output.",
            )

        return ConversionResult(
            success=True,
            converter=Converter.LIBREOFFICE,
            output_path=converted,
            error=None,
            message=f"Successfully converted {src.name} to DOCX using LibreOffice.",
        )

    except subprocess.TimeoutExpired:
        return ConversionResult(
            success=False,
            converter=Converter.LIBREOFFICE,
            output_path=None,
            error="Conversion timed out after 60 seconds",
            message="File is too large or LibreOffice is slow. Try converting manually.",
        )
    except FileNotFoundError:
        return ConversionResult(
            success=False,
            converter=Converter.LIBREOFFICE,
            output_path=None,
            error="LibreOffice executable not found",
            message="LibreOffice binary not found. Install LibreOffice.",
        )


def _try_catdoc(src: Path) -> ConversionResult:
    catdoc = _find_executable("catdoc")
    if not catdoc:
        return ConversionResult(
            success=False,
            converter=Converter.CATDOC,
            output_path=None,
            error="catdoc not found in PATH",
            message="Install catdoc for .doc text extraction fallback.",
        )

    try:
        result = subprocess.run(
            [str(catdoc), "-d", "utf-8", str(src)],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            return ConversionResult(
                success=False,
                converter=Converter.CATDOC,
                output_path=None,
                error=result.stderr.decode(errors="replace"),
                message="catdoc extraction failed.",
            )

        text = result.stdout.decode("utf-8", errors="replace").strip()
        if not text:
            return ConversionResult(
                success=False,
                converter=Converter.CATDOC,
                output_path=None,
                error="No text extracted",
                message="catdoc produced empty output.",
            )

        return ConversionResult(
            success=True,
            converter=Converter.CATDOC,
            output_path=None,  # text-only, no docx file
            error=None,
            message=f"Extracted text from {src.name} using catdoc.",
        )

    except subprocess.TimeoutExpired:
        return ConversionResult(
            success=False,
            converter=Converter.CATDOC,
            output_path=None,
            error="Timeout",
            message="catdoc timed out.",
        )


def _try_antiword(src: Path) -> ConversionResult:
    antiword = _find_executable("antiword")
    if not antiword:
        return ConversionResult(
            success=False,
            converter=Converter.ANTIWORD,
            output_path=None,
            error="antiword not found in PATH",
            message="Install antiword for .doc text extraction fallback.",
        )

    try:
        result = subprocess.run(
            [str(antiword), "-f", str(src)],
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            return ConversionResult(
                success=False,
                converter=Converter.ANTIWORD,
                output_path=None,
                error=result.stderr.decode(errors="replace"),
                message="antiword extraction failed.",
            )

        text = result.stdout.decode("utf-8", errors="replace").strip()
        if not text:
            return ConversionResult(
                success=False,
                converter=Converter.ANTIWORD,
                output_path=None,
                error="No text extracted",
                message="antiword produced empty output.",
            )

        return ConversionResult(
            success=True,
            converter=Converter.ANTIWORD,
            output_path=None,
            error=None,
            message=f"Extracted text from {src.name} using antiword.",
        )

    except subprocess.TimeoutExpired:
        return ConversionResult(
            success=False,
            converter=Converter.ANTIWORD,
            output_path=None,
            error="Timeout",
            message="antiword timed out.",
        )


def convert_doc_to_docx(src: Path) -> ConversionResult:
    """
    Convert a .doc file to .docx using available tools.
    Fallback order: LibreOffice -> catdoc (text only) -> antiword (text only) -> error.

    Returns ConversionResult with:
      - success: whether conversion/extraction succeeded
      - converter: which converter was used
      - output_path: Path to converted .docx (if produced), else None
      - error: error details if failed
      - message: human-readable status
    """
    if not src.exists():
        return ConversionResult(
            success=False,
            converter=Converter.NONE,
            output_path=None,
            error=f"File not found: {src}",
            message="Source file does not exist.",
        )

    if src.suffix.lower() != ".doc":
        return ConversionResult(
            success=False,
            converter=Converter.NONE,
            output_path=None,
            error="Not a .doc file",
            message=f"Expected .doc extension, got {src.suffix}",
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Try LibreOffice first (produces proper .docx)
        result = _try_libreoffice(src, tmp_dir)
        if result.success:
            return result

        # Fallback to catdoc (text only)
        result = _try_catdoc(src)
        if result.success:
            return result

        # Fallback to antiword (text only)
        result = _try_antiword(src)
        if result.success:
            return result

        # All failed
        return ConversionResult(
            success=False,
            converter=Converter.NONE,
            output_path=None,
            error="All converters failed",
            message=(
                "No suitable converter found for .doc files. "
                "Please install LibreOffice (recommended): https://www.libreoffice.org/download/download/ "
                "Or install catdoc / antiword as fallbacks."
            ),
        )
