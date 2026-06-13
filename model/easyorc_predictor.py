"""
EasyOCR-based predictor for real-world handwritten and printed text.
Supports full alphanumeric recognition without any custom training.
Model weights (~100 MB) are downloaded automatically on first use.
"""
import cv2
import numpy as np


class EasyOCRPredictor:
    def __init__(self, languages: list[str] = None, gpu: bool = False):
        import easyocr
        self.reader = easyocr.Reader(languages or ['en'], gpu=gpu, verbose=False)

    def predict(self, image: np.ndarray) -> dict:
        """
        Run OCR on a BGR image.

        Returns a dict with:
            text     – full text joined by newlines
            lines    – list of text strings per detected line
            words    – list of {text, confidence, bbox} per detection
        """
        # EasyOCR expects RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image

        raw = self.reader.readtext(rgb, detail=1, paragraph=False)
        if not raw:
            return {'text': '', 'lines': [], 'words': []}

        words = []
        for (quad, text, conf) in raw:
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            words.append({
                'text': text,
                'confidence': round(float(conf), 4),
                'bbox': {'x': x, 'y': y, 'w': w, 'h': h},
                '_y_center': y + h // 2,
                '_h': h,
            })

        # Group into visual lines by y-center proximity
        words.sort(key=lambda w: (w['_y_center'], w['bbox']['x']))
        line_groups: list[list[dict]] = []
        current: list[dict] = [words[0]]

        for word in words[1:]:
            threshold = max(current[-1]['_h'], word['_h']) * 0.55
            if abs(word['_y_center'] - current[-1]['_y_center']) <= threshold:
                current.append(word)
            else:
                line_groups.append(current)
                current = [word]
        line_groups.append(current)

        # Clean internal helper fields
        for w in words:
            del w['_y_center']
            del w['_h']

        text_lines = [' '.join(w['text'] for w in grp) for grp in line_groups]
        return {
            'text': '\n'.join(text_lines),
            'lines': text_lines,
            'words': words,
        }