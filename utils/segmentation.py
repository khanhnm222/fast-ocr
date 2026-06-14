import cv2
import numpy as np

# Minimum pixel dimensions to count as a real character (filters noise)
MIN_CHAR_W = 5
MIN_CHAR_H = 8

# Horizontal gap (px) between adjacent characters that signals a word space
WORD_GAP_RATIO = 1.5


def segment_lines(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Use horizontal projection profile to find text line bounding boxes.
    Returns list of (x, y, w, h) in top-to-bottom order.
    """
    proj = np.sum(binary, axis=1)
    in_line = False
    line_boxes = []
    y_start = 0

    for y, val in enumerate(proj):
        if val > 0 and not in_line:
            in_line = True
            y_start = y
        elif val == 0 and in_line:
            in_line = False
            if y - y_start >= MIN_CHAR_H:
                line_boxes.append((0, y_start, binary.shape[1], y - y_start))

    if in_line:
        y = len(proj)
        if y - y_start >= MIN_CHAR_H:
            line_boxes.append((0, y_start, binary.shape[1], y - y_start))

    return line_boxes


def segment_characters(
    binary: np.ndarray,
) -> list[dict]:
    """
    Segment all characters in the binary image.

    Returns a list of dicts sorted left-to-right, top-to-bottom:
        {
            'image': np.ndarray,   # cropped character (binary)
            'bbox': (x, y, w, h),  # position in original image
            'line': int,           # 0-based line index
            'space_before': bool,  # True if a word space precedes this char
        }
    """
    line_boxes = segment_lines(binary)
    results = []

    for line_idx, (lx, ly, lw, lh) in enumerate(line_boxes):
        line_img = binary[ly:ly + lh, lx:lx + lw]
        chars_in_line = _extract_chars_from_line(line_img)

        prev_x_end = None
        avg_char_w = (
            np.median([c['bbox'][2] for c in chars_in_line])
            if chars_in_line else 10
        )

        for char_info in chars_in_line:
            cx, cy, cw, ch = char_info['bbox']
            space_before = False
            if prev_x_end is not None:
                gap = cx - prev_x_end
                space_before = bool(gap > avg_char_w * WORD_GAP_RATIO)
            prev_x_end = cx + cw

            results.append({
                'image': char_info['image'],
                'bbox': (lx + cx, ly + cy, cw, ch),
                'line': line_idx,
                'space_before': space_before,
            })

    return results


def _extract_chars_from_line(line_img: np.ndarray) -> list[dict]:
    """Find individual character bounding boxes within a single line image."""
    contours, _ = cv2.findContours(
        line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    chars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < MIN_CHAR_W or h < MIN_CHAR_H:
            continue
        char_img = line_img[y:y + h, x:x + w]
        chars.append({'image': char_img, 'bbox': (x, y, w, h)})

    chars.sort(key=lambda c: c['bbox'][0])
    return chars