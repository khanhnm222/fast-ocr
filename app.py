"""
OCR REST API — Flask application

Endpoints:
    GET  /health              — liveness check + engine info
    POST /ocr                 — multipart/form-data, field 'image'
                                ?engine=easyocr (default) | cnn
    POST /ocr/base64          — JSON { "image": "<base64>", "engine": "easyocr" }

Run:
    python app.py
    gunicorn -w 2 app:app   (production)
"""
import base64
import os

from flask import Flask, jsonify, request

from utils.preprocessing import load_image_from_bytes, preprocess
from utils.segmentation import segment_characters

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded predictors
# ---------------------------------------------------------------------------
_easyocr_predictor = None
_cnn_predictor = None


def _get_easyocr():
    global _easyocr_predictor
    if _easyocr_predictor is None:
        from model.easyocr_predictor import EasyOCRPredictor
        _easyocr_predictor = EasyOCRPredictor(gpu=False)
    return _easyocr_predictor


def _get_cnn():
    global _cnn_predictor
    if _cnn_predictor is None:
        from model.predictor import OCRPredictor
        _cnn_predictor = OCRPredictor()
    return _cnn_predictor


def _cnn_ready() -> bool:
    return any(
        os.path.exists(p)
        for p in ('saved_models/emnist_cnn.keras', 'saved_models/mnist_cnn.keras')
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get('/health')
def health():
    return jsonify({
        'status': 'ok',
        'engines': {
            'easyocr': True,
            'cnn': _cnn_ready(),
        },
        'model_ready': True,
    })


@app.post('/ocr')
def ocr_upload():
    if 'image' not in request.files:
        return jsonify({'error': "Missing 'image' field in form-data."}), 400
    engine = request.args.get('engine', 'easyocr')
    image_bytes = request.files['image'].read()
    return _run_ocr(image_bytes, engine)


@app.post('/ocr/base64')
def ocr_base64():
    body = request.get_json(silent=True)
    if not body or 'image' not in body:
        return jsonify({'error': "Missing 'image' key in JSON body."}), 400
    try:
        image_bytes = base64.b64decode(body['image'])
    except Exception:
        return jsonify({'error': 'Invalid base64 string.'}), 400
    engine = body.get('engine', 'easyocr')
    return _run_ocr(image_bytes, engine)


# ---------------------------------------------------------------------------
# Core OCR dispatch
# ---------------------------------------------------------------------------

def _run_ocr(image_bytes: bytes, engine: str = 'easyocr'):
    try:
        image = load_image_from_bytes(image_bytes)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    if engine == 'cnn':
        return _run_cnn(image)
    return _run_easyocr(image)


def _run_easyocr(image):
    try:
        result = _get_easyocr().predict(image)
    except Exception as exc:
        return jsonify({'error': f'EasyOCR error: {exc}'}), 500

    # Expose words as `characters` too for frontend compatibility
    result['characters'] = [
        {
            'character': w['text'],
            'confidence': w['confidence'],
            'bbox': w['bbox'],
        }
        for w in result['words']
    ]
    result['engine'] = 'easyocr'
    return jsonify(result)


def _run_cnn(image):
    if not _cnn_ready():
        return jsonify({'error': "CNN model not trained. Run `python train.py` first."}), 503

    try:
        predictor = _get_cnn()
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 503

    binary = preprocess(image)
    segments = segment_characters(binary)

    if not segments:
        return jsonify({'text': '', 'lines': [], 'words': [], 'characters': [], 'engine': 'cnn'})

    char_images = [s['image'] for s in segments]
    predictions = predictor.predict_characters(char_images)

    characters = []
    lines: dict[int, list[str]] = {}

    for seg, (label, confidence) in zip(segments, predictions):
        x, y, w, h = seg['bbox']
        characters.append({
            'character': label,
            'confidence': round(confidence, 4),
            'bbox': {'x': x, 'y': y, 'w': w, 'h': h},
            'line': seg['line'],
            'space_before': seg['space_before'],
        })
        lines.setdefault(seg['line'], [])
        if seg['space_before']:
            lines[seg['line']].append(' ')
        lines[seg['line']].append(label)

    text_lines = [''.join(chars) for _, chars in sorted(lines.items())]
    return jsonify({
        'text': '\n'.join(text_lines),
        'lines': text_lines,
        'words': [{'text': c['character'], 'confidence': c['confidence'], 'bbox': c['bbox']}
                  for c in characters],
        'characters': characters,
        'engine': 'cnn',
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)