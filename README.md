# Fast OCR

Language versions:

- English: this file
- Vietnamese: [README.vi.md](README.vi.md)

Fast OCR is an OCR project with a Flask REST API and two OCR engines:

- EasyOCR (default): best for real-world text without training.
- CNN (optional): custom character-level model trained on MNIST or EMNIST.

The repository also includes a React + TypeScript + Vite + Theme UI frontend scaffold in `web/`.

## Features

- REST API for image upload and base64 input.
- Engine selection per request (`easyocr` or `cnn`).
- Character-level output with confidence and bounding boxes.
- CNN training pipeline for MNIST (digits) and EMNIST (alphanumeric).
- Automatic model discovery (`emnist` first, then `mnist`).

## Project Structure

```text
fast-ocr/
├── app.py
├── train.py
├── requirements.txt
├── model/
│   ├── cnn_model.py
│   ├── predictor.py
│   └── easyorc_predictor.py
├── utils/
│   ├── preprocessing.py
│   └── segmentation.py
├── saved_models/
└── web/
```

## OCR Engines

### 1) EasyOCR (Default)

- Used when no engine is provided.
- No custom training required.
- Good for practical OCR on mixed text in images.

### 2) CNN (Optional)

- Character segmentation + classification pipeline.
- Requires trained model files in `saved_models/`.
- Supports:
  - MNIST: `0-9`
  - EMNIST byclass: `0-9`, `A-Z`, `a-z`

## CNN Pipeline Overview

```text
Input image
  -> preprocess (grayscale, denoise, adaptive threshold, morph open)
  -> segment lines/characters
  -> classify each char (CNN)
  -> build lines/text + confidence + bounding boxes
```

## API

Default server port: `5000`

### `GET /health`

Returns service and engine availability.

Example response:

```json
{
  "status": "ok",
  "engines": {
    "easyocr": true,
    "cnn": false
  },
  "model_ready": true
}
```

### `POST /ocr`

Multipart upload with field name `image`.

Optional query parameter:

- `engine=easyocr` (default)
- `engine=cnn`

Example:

```bash
curl -X POST "http://localhost:5000/ocr?engine=easyocr" \
  -F "image=@sample.jpg"
```

### `POST /ocr/base64`

JSON body:

```json
{
  "image": "<base64-string>",
  "engine": "easyocr"
}
```

Example:

```bash
curl -X POST http://localhost:5000/ocr/base64 \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64-string>","engine":"cnn"}'
```

## Response Shape

Common fields:

- `text`: full recognized text
- `lines`: text split by lines
- `words`: detected units with confidence and `bbox`
- `characters`: compatibility field used by frontend
- `engine`: engine used (`easyocr` or `cnn`)

CNN responses include extra character metadata:

- `line`: 0-based line index
- `space_before`: whether a word-space is inferred before this character

## Setup

### Requirements

- Python 3.10+
- Node.js 18+ (for `web/`)

### 1) Install Python dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 2) (Optional) Train CNN model

MNIST (digits only):

```bash
python train.py
```

EMNIST (alphanumeric, larger download):

```bash
python train.py --dataset emnist
```

Custom training params:

```bash
python train.py --dataset emnist --epochs 20 --batch-size 128
```

Generated files are saved in `saved_models/` (model, labels, and training plot).

### 3) Run API

Development:

```bash
python app.py
```

Production example:

```bash
gunicorn -w 2 app:app
```

## Frontend (Current State)

The `web/` app is currently a starter Theme UI page and is not yet wired to OCR endpoints.

Run it with:

```bash
cd web
npm install
npm run dev
```

## Notes

- CNN engine returns `503` if no trained model exists.
- EasyOCR model weights are downloaded automatically on first use.
- Model load priority for CNN inference is:
  1. `saved_models/emnist_cnn.keras`
  2. `saved_models/mnist_cnn.keras`