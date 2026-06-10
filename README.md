# OCR Tool

Công cụ nhận dạng ký tự (OCR) từ ảnh chụp, sử dụng mạng CNN được huấn luyện trên tập dữ liệu MNIST / EMNIST. Bao gồm REST API (Flask) và giao diện web (React TypeScript + Vite + Theme-UI).

---

## Kiến trúc tổng thể

```
Ảnh đầu vào
     │
     ▼
Preprocessing          (grayscale → denoise → adaptive threshold → morph open)
     │
     ▼
Segmentation           (projection profile → line boxes → contour → char boxes)
     │
     ▼
CNN Prediction         (resize 28×28 → normalize → softmax → label + confidence)
     │
     ▼
JSON Response          { text, lines[], characters[{char, confidence, bbox, line}] }
```

---

## Cấu trúc dự án

```
ocr/
├── app.py                      # Flask REST API
├── train.py                    # Script huấn luyện model
├── requirements.txt
│
├── model/
│   ├── cnn_model.py            # Kiến trúc CNN (build_model)
│   └── predictor.py            # OCRPredictor — load model, batch predict
│
├── utils/
│   ├── preprocessing.py        # Tiền xử lý ảnh (grayscale, threshold, deskew)
│   └── segmentation.py         # Phân đoạn dòng và ký tự
│
├── saved_models/               # Được tạo sau khi train
│   ├── mnist_cnn.keras
│   ├── mnist_labels.json
│   ├── emnist_cnn.keras         # (nếu train EMNIST)
│   └── emnist_labels.json
│
└── web/                        # Frontend React TypeScript + Vite + Theme-UI
    ├── index.html
    ├── vite.config.ts
    ├── tsconfig.json
    ├── tsconfig.app.json
    ├── tsconfig.node.json
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── theme.ts               # Theme-UI theme (màu, font, spacing, variants)
        ├── types.ts               # TypeScript types dùng chung (OcrResult, DetectionItem…)
        └── components/
            ├── ImageUploader.tsx
            ├── ImagePreview.tsx   # Bounding box overlay (SVG)
            └── OCRResult.tsx      # Tab: full text / per-line / per-character
```

---

## Model

### Kiến trúc CNN

File: [model/cnn_model.py](model/cnn_model.py)

| Layer            | Chi tiết                          |
|------------------|-----------------------------------|
| Input            | 28 × 28 × 1 (grayscale)           |
| Conv2D + BN      | 32 filters, 3×3, padding=same, ReLU |
| Conv2D           | 32 filters, 3×3, padding=same, ReLU |
| MaxPool + Dropout| 2×2 pool, dropout 0.25            |
| Conv2D + BN      | 64 filters, 3×3, padding=same, ReLU |
| Conv2D           | 64 filters, 3×3, padding=same, ReLU |
| MaxPool + Dropout| 2×2 pool, dropout 0.25            |
| Flatten          |                                   |
| Dense + BN       | 256 units, ReLU, dropout 0.5      |
| Dense (output)   | `num_classes` units, softmax      |

### Tập dữ liệu

| Dataset       | Classes | Train     | Test    | Nhãn                      |
|---------------|---------|-----------|---------|---------------------------|
| MNIST         | 10      | 60 000    | 10 000  | `0`–`9`                   |
| EMNIST byclass| 62      | 697 932   | 116 323 | `0`–`9`, `A`–`Z`, `a`–`z` |

### Data augmentation (khi train)

- `RandomRotation` ±10%
- `RandomZoom` ±10%
- `RandomTranslation` ±10% (cả hai chiều)

### Callbacks

- `EarlyStopping` — patience=5, khôi phục trọng số tốt nhất
- `ReduceLROnPlateau` — patience=3, factor=0.5

---

## Tiền xử lý ảnh

File: [utils/preprocessing.py](utils/preprocessing.py)

1. **Grayscale** — chuyển BGR → 1 kênh
2. **Denoise** — `fastNlMeansDenoising(h=10)` để xử lý nhiễu ảnh chụp
3. **Adaptive threshold** — `ADAPTIVE_THRESH_GAUSSIAN_C`, blockSize=15, C=8 (xử lý ánh sáng không đều)
4. **Morphological opening** — kernel 2×2 để loại điểm nhiễu nhỏ lẻ
5. **Deskew** (hàm riêng) — phát hiện góc nghiêng qua `minAreaRect`, xoay bù bằng affine transform

---

## Phân đoạn ký tự

File: [utils/segmentation.py](utils/segmentation.py)

**Phân đoạn dòng** — `segment_lines()`  
Dùng horizontal projection profile (tổng pixel theo hàng) để tìm vùng có văn bản.

**Phân đoạn ký tự** — `segment_characters()`  
Với mỗi dòng, dùng `findContours` để trích xuất bounding box từng ký tự. Ngưỡng lọc nhiễu: `MIN_CHAR_W=5`, `MIN_CHAR_H=8`. Khoảng cách giữa từ được phát hiện khi gap > `WORD_GAP_RATIO × median_char_width` (mặc định 1.5).

---

## REST API

File: [app.py](app.py) — Flask server, mặc định port **5000**

### Endpoints

#### `GET /health`

```json
{ "status": "ok", "model_ready": true }
```

`model_ready: false` nghĩa là chưa có file model trong `saved_models/`.

---

#### `POST /ocr`

Upload ảnh qua `multipart/form-data`, field tên `image`.

```bash
curl -X POST http://localhost:5000/ocr \
  -F "image=@photo.jpg"
```

---

#### `POST /ocr/base64`

Gửi ảnh dạng base64 qua JSON body.

```bash
curl -X POST http://localhost:5000/ocr/base64 \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-string>"}'
```

---

### Response format

```json
{
  "text": "Hello 123",
  "lines": ["Hello 123"],
  "characters": [
    {
      "character": "H",
      "confidence": 0.9821,
      "bbox": { "x": 12, "y": 5, "w": 18, "h": 24 },
      "line": 0,
      "space_before": false
    }
  ]
}
```

| Trường          | Ý nghĩa                                      |
|-----------------|----------------------------------------------|
| `text`          | Toàn bộ văn bản được nhận dạng               |
| `lines`         | Mảng chuỗi theo từng dòng                    |
| `characters`    | Từng ký tự với tọa độ và độ tin cậy          |
| `confidence`    | Giá trị 0–1, xác suất dự đoán của softmax    |
| `bbox`          | Vị trí ký tự trên ảnh gốc (px)              |
| `space_before`  | `true` nếu có khoảng trắng trước ký tự này  |

---

## Giao diện Web

Stack: **React 19 + TypeScript + Vite + Theme-UI**, chạy tại `http://localhost:5173`

- Drag & drop hoặc click để tải ảnh lên
- Hiển thị ảnh gốc với bounding box SVG overlay (màu emerald cho EasyOCR, tím cho CNN, kèm nhãn và % confidence)
- Badge trạng thái API: **sẵn sàng** / **chưa train model** / **offline**
- Kết quả hiển thị dạng tab: toàn văn bản / theo dòng / từng ký tự
- Proxy `/ocr` và `/health` đến Flask backend (không cần cấu hình CORS)
- Dark mode mặc định, hỗ trợ light mode qua Theme-UI color modes
- Styling dùng `sx` prop của Theme-UI — color tokens (`primary`, `surface`, `muted`…) thay cho utility classes

---

## Hướng dẫn chạy

### Yêu cầu

- Python 3.10+
- Node.js 18+

### Backend

```bash
# 1. Cài dependencies
python3 -m pip install -r requirements.txt

# 2. Huấn luyện model
python train.py                        # MNIST — digits 0-9 (~15 epochs, nhanh)
python train.py --dataset emnist       # EMNIST — thêm A-Z, a-z (~500 MB download)
python train.py --epochs 20 --batch-size 128   # tuỳ chỉnh tham số

# 3. Khởi động API server
python app.py                          # development (port 5000)
gunicorn -w 4 app:app                  # production
```

Sau bước 2, các file sau được tạo trong `saved_models/`:
- `mnist_cnn.keras` / `emnist_cnn.keras` — trọng số model
- `mnist_labels.json` / `emnist_labels.json` — bản đồ nhãn
- `mnist_training_history.png` — biểu đồ accuracy và loss

### Frontend

```bash
cd web
npm install
npm run dev        # http://localhost:5173 (dev server với HMR)
npm run build      # type-check + production build → dist/
```

---

## Mở rộng: hỗ trợ chữ cái (EMNIST)

Để nhận dạng cả chữ cái A–Z / a–z, dùng EMNIST thay MNIST:

```bash
pip install emnist
python train.py --dataset emnist
```

Model tự động được phát hiện theo thứ tự ưu tiên: `emnist_cnn.keras` → `mnist_cnn.keras`.  
Không cần thay đổi code API hay frontend.

---

## Dependencies chính

| Thư viện       | Mục đích                        |
|----------------|----------------------------------|
| Flask          | REST API server                  |
| TensorFlow 2   | Xây dựng và chạy model CNN       |
| OpenCV         | Xử lý ảnh, phân đoạn ký tự      |
| NumPy          | Thao tác ma trận                 |
| Matplotlib     | Vẽ biểu đồ lịch sử huấn luyện   |
| emnist         | Tải tập dữ liệu EMNIST           |
| React 19       | Giao diện web                    |
| TypeScript     | Type safety cho toàn bộ frontend |
| Vite           | Build tool & dev server          |
| Theme-UI       | Styling với design tokens & `sx` |