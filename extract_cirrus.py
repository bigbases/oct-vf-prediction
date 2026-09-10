#!/usr/bin/env python3
"""
Cirrus OCT / HFA Report Region Extractor
=========================================

Batch-extract anatomical regions from Cirrus OCT (GCA, RNFL) and HFA (SFA) reports.

Supports:
  - PDF inputs (Ganglion Cell / ONH and RNFL analyses) — rendered at 200 DPI
  - JPG/PNG inputs (Single Field Analysis) — used at native resolution
  - Auto-detection of report type by filename hints
  - Automatic scaling when source resolution differs from calibration

Output structure:
  output_dir/
    by_region/
      GCA_od_thickness_map/
        <patient_id>__<source_basename>.png
        ...
    by_case/
      <patient_id>__<source_basename>/
        od_thickness_map.png
        os_thickness_map.png
        ...

Usage:
  python extract_cirrus.py \\
      --input /path/to/reports \\
      --output /path/to/output \\
      --regions cirrus_regions.json

  # Dry run to preview without processing:
  python extract_cirrus.py --input ... --output ... --regions ... --dry-run

  # Verify coordinates visually (draws colored boxes on source images, no cropping):
  python extract_cirrus.py --input ... --output ... --regions ... --verify
  # Verify only the first N samples per type (default 1):
  python extract_cirrus.py --input ... --output ... --regions ... --verify --verify-n 3

  # Specific file:
  python extract_cirrus.py --input one_file.pdf --output out/ --regions cirrus_regions.json

Requirements:
  pip install pillow pdf2image
  # pdf2image needs poppler: apt-get install poppler-utils
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PDF_DPI = 200  # MUST match calibration; changing this will break coordinates

# Windows에서만 필요한 poppler bin 경로. 환경변수 HVF_POPPLER_PATH로 지정한다.
# 리눅스(poppler-utils 설치)에서는 None이며 PATH를 쓴다.
POPPLER_PATH = paths.POPPLER_PATH

# Report-type detection patterns (checked in order against the filename).
# Customize these if your filename convention differs.
TYPE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("GCA",  re.compile(r"(macular[_\s-]*cube|ganglion[_\s-]*cell)",  re.I)),
    ("RNFL", re.compile(r"(optic[_\s-]*disc[_\s-]*cube|onh[_\s-]*and[_\s-]*rnfl|rnfl)", re.I)),
    ("SFA",  re.compile(r"(single[_\s-]*field|hfa|threshold|24[_\s-]*2|30[_\s-]*2|\bopt\b)", re.I)),
]

# Supported input extensions
PDF_EXTS   = {".pdf"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("cirrus_extract")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                        datefmt="%H:%M:%S"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def detect_type(filename: str) -> Optional[str]:
    """Infer report type (GCA / RNFL / SFA) from the filename."""
    name = filename.lower()
    for rtype, pattern in TYPE_PATTERNS:
        if pattern.search(name):
            return rtype
    return None


def extract_patient_id(filename: str) -> str:
    """
    Pull a patient ID from the filename if possible. Falls back to the stem.
    Customize the regex for your filename convention.
    """
    # Common patterns: name__ID_... or first 8 digits, etc.
    m = re.search(r"(\d{7,10})", filename)
    if m:
        return m.group(1)
    return Path(filename).stem


def pdf_to_image(pdf_path: Path, dpi: int = PDF_DPI) -> Image.Image:
    """Render the first page of a PDF to a PIL Image at the given DPI."""
    try:
        from pdf2image import convert_from_path  # lazy import
    except ImportError as e:
        raise RuntimeError(
            "pdf2image not installed. Run: pip install pdf2image  "
            "(and ensure poppler-utils is available on your system)"
        ) from e

    pages = convert_from_path(str(pdf_path), dpi=dpi, first_page=1, last_page=1,
                              poppler_path=POPPLER_PATH or None)
    if not pages:
        raise RuntimeError(f"No pages rendered from {pdf_path}")
    return pages[0].convert("RGB")


def load_source_image(src_path: Path) -> Image.Image:
    """Load the source as an RGB PIL image, rendering PDF if needed."""
    suffix = src_path.suffix.lower()
    if suffix in PDF_EXTS:
        return pdf_to_image(src_path, dpi=PDF_DPI)
    if suffix in IMAGE_EXTS:
        return Image.open(src_path).convert("RGB")
    raise ValueError(f"Unsupported file type: {src_path}")


def compute_scaled_bbox(
    bbox: Tuple[int, int, int, int],
    calib_dims: Tuple[int, int],
    actual_dims: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """
    Scale a calibration-space bbox to the actual image dimensions.
    If sizes match, returns the bbox unchanged.
    """
    cw, ch = calib_dims
    aw, ah = actual_dims
    if (cw, ch) == (aw, ah):
        return bbox
    sx, sy = aw / cw, ah / ch
    x1, y1, x2, y2 = bbox
    return (
        max(0, int(round(x1 * sx))),
        max(0, int(round(y1 * sy))),
        min(aw, int(round(x2 * sx))),
        min(ah, int(round(y2 * sy))),
    )


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_one(
    src_path: Path,
    regions_config: dict,
    output_dir: Path,
    dry_run: bool = False,
    forced_type: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Process a single source file.
    Returns (success, message).
    """
    rtype = forced_type or detect_type(src_path.name)
    if rtype is None:
        return False, f"Could not detect report type: {src_path.name}"
    if rtype not in regions_config:
        return False, f"Report type '{rtype}' not in regions config: {src_path.name}"

    cfg = regions_config[rtype]
    regions: Dict[str, dict] = cfg["regions"]
    calib_dims = tuple(cfg["calibration_dims"])

    patient_id = extract_patient_id(src_path.name)
    case_key = f"{patient_id}__{src_path.stem}"

    if dry_run:
        return True, f"[DRY] {rtype}  {case_key}  ({len(regions)} regions)"

    # Load & (if PDF) render
    try:
        img = load_source_image(src_path)
    except Exception as e:
        return False, f"Load failed: {src_path.name}: {e}"

    actual_dims = img.size  # (width, height)
    if actual_dims != calib_dims:
        logger.debug(
            "Source dims %s != calibration %s for %s — scaling bboxes.",
            actual_dims, calib_dims, src_path.name,
        )

    by_region_root = output_dir / "by_region"
    by_case_root   = output_dir / "by_case" / case_key
    by_case_root.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    for region_name, spec in regions.items():
        bbox = tuple(spec["bbox"])
        scaled = compute_scaled_bbox(bbox, calib_dims, actual_dims)
        try:
            crop = img.crop(scaled)
        except Exception as e:
            logger.warning("Crop failed for %s/%s: %s", case_key, region_name, e)
            continue

        # Save into both layouts
        reg_dir = by_region_root / f"{rtype}_{region_name}"
        reg_dir.mkdir(parents=True, exist_ok=True)
        crop.save(reg_dir / f"{case_key}.png")
        crop.save(by_case_root / f"{region_name}.png")
        ok_count += 1

    return True, f"{rtype}  {case_key}  ({ok_count}/{len(regions)} regions saved)"


# ---------------------------------------------------------------------------
# Verification (--verify): draw coordinate overlays instead of cropping
# ---------------------------------------------------------------------------

_OVERLAY_PALETTE = [
    (255, 0, 0), (0, 128, 255), (0, 180, 0), (255, 140, 0),
    (200, 0, 200), (0, 180, 180), (255, 0, 255), (128, 64, 0),
    (0, 0, 255), (230, 180, 0), (120, 200, 120), (230, 100, 100),
    (100, 100, 200), (160, 80, 40), (80, 200, 255),
]


def _load_overlay_fonts() -> Tuple:
    """Load fonts for overlay; fall back to default if unavailable."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    label_font, title_font = None, None
    for p in font_paths:
        try:
            label_font = ImageFont.truetype(p, 22)
            title_font = ImageFont.truetype(p, 36)
            break
        except Exception:
            continue
    if label_font is None:
        label_font = ImageFont.load_default()
        title_font = label_font
    return label_font, title_font


def verify_one(
    src_path: Path,
    regions_config: dict,
    output_dir: Path,
    forced_type: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Draw coordinate overlay on a source file and save it to output_dir/verify/.
    Returns (success, message).
    """
    rtype = forced_type or detect_type(src_path.name)
    if rtype is None:
        return False, f"Could not detect report type: {src_path.name}"
    if rtype not in regions_config:
        return False, f"Report type '{rtype}' not in regions config: {src_path.name}"

    try:
        img = load_source_image(src_path)
    except Exception as e:
        return False, f"Load failed: {src_path.name}: {e}"

    cfg = regions_config[rtype]
    regions: Dict[str, dict] = cfg["regions"]
    calib_dims = tuple(cfg["calibration_dims"])
    actual_dims = img.size

    label_font, title_font = _load_overlay_fonts()
    draw = ImageDraw.Draw(img, "RGBA")

    for i, (name, spec) in enumerate(regions.items()):
        bbox = tuple(spec["bbox"])
        sx1, sy1, sx2, sy2 = compute_scaled_bbox(bbox, calib_dims, actual_dims)
        color = _OVERLAY_PALETTE[i % len(_OVERLAY_PALETTE)]
        draw.rectangle((sx1, sy1, sx2, sy2), outline=color + (255,), width=5)
        # Label with white background for readability
        tx, ty = sx1 + 6, sy1 + 6
        bb = draw.textbbox((tx, ty), name, font=label_font)
        draw.rectangle(bb, fill=(255, 255, 255, 230))
        draw.text((tx, ty), name, fill=color + (255,), font=label_font)

    # Title banner at top
    title = f"{rtype} VERIFY — {src_path.name}  (src {actual_dims[0]}x{actual_dims[1]}, calib {calib_dims[0]}x{calib_dims[1]})"
    draw.rectangle((0, 0, img.width, 60), fill=(0, 0, 0, 220))
    draw.text((20, 14), title, fill=(255, 255, 255), font=title_font)

    verify_dir = output_dir / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    out_path = verify_dir / f"{rtype}_{src_path.stem}_overlay.png"
    img.save(out_path)
    return True, f"{rtype}  {src_path.name}  -> {out_path.name}"


def select_verify_samples(inputs: List[Path], n_per_type: int,
                          forced_type: Optional[str]) -> List[Path]:
    """Pick up to n samples per detected type (or all if --type is forced)."""
    if forced_type is not None:
        return inputs[:n_per_type]
    buckets: Dict[str, List[Path]] = {}
    unknown: List[Path] = []
    for p in inputs:
        t = detect_type(p.name)
        if t is None:
            unknown.append(p)
        else:
            buckets.setdefault(t, []).append(p)
    selected: List[Path] = []
    for t, files in buckets.items():
        selected.extend(files[:n_per_type])
    return selected


def find_inputs(input_path: Path) -> List[Path]:
    """Gather all processable files from a file or directory."""
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input not found: {input_path}")
    files: List[Path] = []
    for ext in list(PDF_EXTS) + list(IMAGE_EXTS):
        files.extend(input_path.rglob(f"*{ext}"))
        files.extend(input_path.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-extract Cirrus OCT / HFA report regions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",   required=True, type=Path,
                   help="Input file or directory (searched recursively).")
    p.add_argument("--output",  required=True, type=Path,
                   help="Output directory. Created if missing.")
    p.add_argument("--regions", required=True, type=Path,
                   help="Path to cirrus_regions.json (the coordinate config).")
    p.add_argument("--type", choices=["GCA", "RNFL", "SFA"], default=None,
                   help="Force a specific report type (skip filename detection).")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would be processed without writing files.")
    p.add_argument("--verify", action="store_true",
                   help="Verification mode: draw coordinate overlays on sample "
                        "source images instead of cropping. Use this to visually "
                        "confirm coordinates before running a full extraction.")
    p.add_argument("--verify-n", type=int, default=1,
                   help="When --verify is set, number of samples per report "
                        "type to overlay (default: 1).")
    p.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if not args.regions.is_file():
        logger.error("Regions config not found: %s", args.regions)
        return 2

    with args.regions.open("r", encoding="utf-8") as f:
        regions_config = json.load(f)
    # Strip meta block if present
    regions_config = {k: v for k, v in regions_config.items() if not k.startswith("_")}

    inputs = find_inputs(args.input)
    if not inputs:
        logger.error("No input files found under %s", args.input)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # --verify branch: draw overlays on sample source files and exit.
    # -----------------------------------------------------------------------
    if args.verify:
        samples = select_verify_samples(inputs, args.verify_n, args.type)
        if not samples:
            logger.error("No samples selected for verification.")
            return 1
        logger.info("=== VERIFY MODE ===  %d sample(s) selected", len(samples))
        ok = fail = 0
        for i, src in enumerate(samples, 1):
            success, msg = verify_one(src, regions_config, args.output,
                                      forced_type=args.type)
            if success:
                ok += 1
                logger.info("[%d/%d] %s", i, len(samples), msg)
            else:
                fail += 1
                logger.warning("[%d/%d] FAIL %s", i, len(samples), msg)
        logger.info("-" * 60)
        logger.info("Verification complete: %d overlay(s) in %s/verify/",
                    ok, args.output)
        logger.info("Open them in an image viewer to visually confirm "
                    "coordinates before running full extraction.")
        return 0 if fail == 0 else 3

    logger.info("Found %d input file(s). Dry-run=%s", len(inputs), args.dry_run)
    ok, fail, skip = 0, 0, 0
    skipped_files: List[str] = []
    for i, src in enumerate(inputs, 1):
        success, msg = process_one(
            src, regions_config, args.output, dry_run=args.dry_run,
            forced_type=args.type,
        )
        if success:
            ok += 1
            logger.info("[%d/%d] %s", i, len(inputs), msg)
        else:
            if "Could not detect" in msg or "not in regions" in msg:
                skip += 1
                skipped_files.append(f"  - {src.name}: {msg}")
            else:
                fail += 1
            logger.warning("[%d/%d] SKIP %s", i, len(inputs), msg)

    logger.info("-" * 60)
    logger.info("Summary: %d ok, %d failed, %d skipped (unrecognized type)",
                ok, fail, skip)
    if skipped_files and skip <= 20:
        logger.info("Skipped files:\n" + "\n".join(skipped_files))
    return 0 if fail == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
