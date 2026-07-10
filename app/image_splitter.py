"""Pre-split tall manhwa images for YOLO panel detection.

YOLO's panel.pt model is trained on typical manhwa strip sizes (~800x2000-4000px).
When images are very tall (>1500px without detected content gaps), YOLO fails to
detect panels reliably. This module finds natural cut points (low content-density
horizontal strips — the gaps between panels) and splits tall images into smaller
pieces that YOLO can handle.

Adapted from the cut detection logic in momo-cutter (processing/segmenter.py,
cuts/detector.py).
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# After this many pixels without a cut, force a search for the best cut point
MAX_GAP_WITHOUT_CUT = 1500

# Minimum height for a split piece (don't create tiny fragments)
MIN_PIECE_HEIGHT = 300

# Density analysis strip height (px)
STRIP_HEIGHT = 10

# Density thresholds (% of non-background pixels)
EXCELLENT_DENSITY = 5.0    # Pure background strip
GOOD_DENSITY = 20.0        # Mostly background
MAX_DENSITY = 50.0          # Won't cut above this


def find_cut_points(image: np.ndarray, max_gap: int = MAX_GAP_WITHOUT_CUT) -> List[int]:
    """Find natural horizontal cut points in a tall image.

    Scans the image for low content-density horizontal strips (the gaps between
    panels — usually pure white or pure black). Returns Y positions suitable for
    splitting.

    Args:
        image: BGR numpy array (the full page image).
        max_gap: Maximum pixels between cuts before forcing a search.

    Returns:
        Sorted list of Y positions to cut at. May be empty if the image
        doesn't need splitting or no good cuts were found.
    """
    h, w = image.shape[:2]
    if h <= max_gap:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg_color = _detect_background_color(gray)

    # Build density map
    density_map = _content_density_map(gray, bg_color)

    # Walk through the image finding cut points
    cuts = []
    last_cut = 0

    while last_cut + max_gap < h:
        # Search window: from last_cut + MIN_PIECE_HEIGHT to last_cut + max_gap + some slack
        search_start = last_cut + MIN_PIECE_HEIGHT
        search_end = min(h - MIN_PIECE_HEIGHT, last_cut + max_gap + max_gap // 2)

        if search_start >= search_end:
            break

        best = _best_cut_in_range(density_map, search_start, search_end)
        if best is not None:
            cuts.append(best)
            last_cut = best
        else:
            # No good cut found — skip ahead and try again
            last_cut += max_gap
            logger.debug(f"No cut found in range {search_start}-{search_end}, skipping ahead")

    return cuts


def split_image(image: np.ndarray, cut_points: List[int]) -> List[Tuple[np.ndarray, int]]:
    """Split an image at the given Y positions.

    Args:
        image: BGR numpy array.
        cut_points: Sorted list of Y positions to cut at.

    Returns:
        List of (piece_image, y_offset) tuples. y_offset is the position
        of this piece in the original image.
    """
    if not cut_points:
        return [(image, 0)]

    pieces = []
    boundaries = [0] + sorted(cut_points) + [image.shape[0]]

    for i in range(len(boundaries) - 1):
        y_start = boundaries[i]
        y_end = boundaries[i + 1]
        if y_end - y_start < 10:
            continue
        piece = image[y_start:y_end].copy()
        pieces.append((piece, y_start))

    return pieces


def pre_split_for_yolo(image: np.ndarray, max_gap: int = MAX_GAP_WITHOUT_CUT) -> List[Tuple[np.ndarray, int]]:
    """Convenience: find cuts and split in one call.

    Returns list of (piece_image, y_offset) — if no splitting needed,
    returns [(image, 0)].
    """
    cuts = find_cut_points(image, max_gap)
    if not cuts:
        return [(image, 0)]

    logger.info(f"Splitting {image.shape[1]}x{image.shape[0]} image at {len(cuts)} cut points: {cuts}")
    return split_image(image, cuts)


# ---------------------------------------------------------------------------
# Stitch-before-detect pipeline
# ---------------------------------------------------------------------------

def stitch_images(image_paths: List[str]) -> Optional[Tuple[np.ndarray, List[Tuple[int, int, int]]]]:
    """Stitch multiple images into one vertical strip.

    All images are scaled to the most common width (proportional height).

    Returns:
        (stitched_image, positions) where positions is a list of
        (start_y, end_y, original_index) for each source image in the
        stitched coordinate space.  Returns None if no images could be loaded.
    """
    from collections import Counter

    images = []
    valid_indices = []
    for i, path in enumerate(image_paths):
        raw = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
            valid_indices.append(i)

    if not images:
        return None

    # Find target width (most common)
    widths = [img.shape[1] for img in images]
    target_width = Counter(widths).most_common(1)[0][0]

    # Scale all images to target width and track positions
    scaled = []
    positions: List[Tuple[int, int, int]] = []  # (start_y, end_y, original_index)
    current_y = 0
    for img, orig_idx in zip(images, valid_indices):
        h, w = img.shape[:2]
        if w != target_width and w > 0:
            new_h = int(h * (target_width / w))
            img = cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_h = h
        scaled.append(img)
        positions.append((current_y, current_y + new_h, orig_idx))
        current_y += new_h

    stitched = np.vstack(scaled)
    return stitched, positions


def stitch_split_for_yolo(
    image_paths: List[str],
    max_gap: int = MAX_GAP_WITHOUT_CUT,
) -> List[Tuple[np.ndarray, int, int, float]]:
    """Stitch all images, find clean cut points, split, and map back.

    This solves the problem where panels are split across image boundaries.
    By stitching first, YOLO sees whole panels even when the original files
    cut them in half.

    Returns:
        List of (piece_image, original_image_index, y_offset_in_original, scale_factor)
        for each segment.  scale_factor converts stitched coords → original coords.
    """
    result = stitch_images(image_paths)
    if result is None:
        return []

    stitched, positions = result
    logger.info(f"Stitched {len(positions)} images into {stitched.shape[1]}x{stitched.shape[0]} strip")

    # Find clean cut points on the full stitched strip
    cuts = find_cut_points(stitched, max_gap)
    pieces = split_image(stitched, cuts)
    logger.info(f"Split stitched strip into {len(pieces)} pieces at {len(cuts)} cut points")

    # Map each piece back to original image coordinates
    mapped: List[Tuple[np.ndarray, int, int, float]] = []
    for piece_img, stitch_y_offset in pieces:
        piece_h = piece_img.shape[0]
        piece_center = stitch_y_offset + piece_h // 2

        # Find which original image this piece's center falls in
        best_idx = 0
        best_start = 0
        for start_y, end_y, orig_idx in positions:
            if start_y <= piece_center < end_y:
                best_idx = orig_idx
                best_start = start_y
                break
        else:
            # Past the end — use the last image
            best_idx = positions[-1][2]
            best_start = positions[-1][0]

        # y_offset relative to the original image's start in the stitched space
        y_in_original = stitch_y_offset - best_start

        # Compute scale factor: original image width / stitched width
        # (we may have scaled images to match widths)
        raw = np.fromfile(image_paths[best_idx], dtype=np.uint8)
        orig = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if orig is not None:
            scale = orig.shape[1] / stitched.shape[1]
        else:
            scale = 1.0

        mapped.append((piece_img, best_idx, y_in_original, scale))

    return mapped


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_background_color(gray: np.ndarray) -> int:
    """Detect dominant background color from image edges."""
    h, w = gray.shape

    # Sample edges (top row, bottom row, left col, right col)
    samples = np.concatenate([
        gray[0, :].ravel(),
        gray[-1, :].ravel(),
        gray[:, 0].ravel(),
        gray[:, -1].ravel(),
    ])

    hist, _ = np.histogram(samples, bins=256, range=(0, 256))
    bg = int(np.argmax(hist))

    # If middle-range, check if white or black dominates
    if 50 < bg < 200:
        white_count = np.sum(hist[240:])
        black_count = np.sum(hist[:21])
        mid_count = hist[bg]
        if white_count > mid_count * 0.7:
            bg = 250
        elif black_count > mid_count * 0.7:
            bg = 10

    return bg


def _content_density_map(gray: np.ndarray, bg_color: int, tolerance: int = 20) -> List[Tuple[int, float]]:
    """Build a map of (y_center, density%) for horizontal strips.

    Density = percentage of pixels that are NOT the background color.
    Low density = gap between panels (good cut candidate).
    """
    h, w = gray.shape
    result = []

    for y in range(0, h, STRIP_HEIGHT):
        y_end = min(y + STRIP_HEIGHT, h)
        strip = gray[y:y_end, :]

        bg_mask = np.abs(strip.astype(np.int16) - bg_color) <= tolerance
        content_pixels = np.sum(~bg_mask)
        density = (content_pixels / strip.size) * 100.0

        center_y = y + (y_end - y) // 2
        result.append((center_y, density))

    return result


def _best_cut_in_range(
    density_map: List[Tuple[int, float]],
    y_min: int,
    y_max: int,
) -> Optional[int]:
    """Find the best cut point within a Y range.

    Picks the lowest-density strip in the range. If nothing is below
    MAX_DENSITY, returns None.
    """
    candidates = [
        (y, d) for y, d in density_map
        if y_min <= y <= y_max and d <= MAX_DENSITY
    ]

    if not candidates:
        return None

    # Pick the one with lowest density
    best_y, best_d = min(candidates, key=lambda c: c[1])

    quality = "excellent" if best_d <= EXCELLENT_DENSITY else "good" if best_d <= GOOD_DENSITY else "acceptable"
    logger.debug(f"Best cut at y={best_y} density={best_d:.1f}% ({quality})")

    return best_y
