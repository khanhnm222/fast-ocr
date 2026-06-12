# Fast OCR

Phiên bản ngôn ngữ:

- Tiếng Việt: file này
- English: [README.md](README.md)

Fast OCR là dự án OCR gồm Flask REST API và 2 engine nhận dạng:

- EasyOCR (mặc định): phù hợp với ảnh thực tế, không cần train.
- CNN (tùy chọn): mô hình nhận dạng ở mức ký tự, train bằng MNIST hoặc EMNIST.

Repository cũng có frontend scaffold dùng React + TypeScript + Vite + Theme UI trong thư mục `web/`.

## Tính năng

- REST API cho upload ảnh và input base64.
- Chọn engine theo từng request (`easyocr` hoặc `cnn`).
- Kết quả mức ký tự với confidence và bounding box.
- Pipeline train CNN cho MNIST (chữ số) và EMNIST (chữ số + chữ cái).
- Tự động phát hiện model (`emnist` ưu tiên trước, rồi `mnist`).

## Cấu trúc dự án

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

### 1) EasyOCR (Mặc định)

- Được dùng khi không truyền engine.
- Không cần train model riêng.
- Phù hợp cho OCR thực tế với nhiều dạng chữ trong ảnh.

### 2) CNN (Tùy chọn)

- Pipeline phân đoạn ký tự + phân loại.
- Cần có file model đã train trong `saved_models/`.
- Hỗ trợ:
  - MNIST: `0-9`
  - EMNIST byclass: `0-9`, `A-Z`, `a-z`

## Tổng quan pipeline CNN

```text
Ảnh đầu vào
  -> preprocess (grayscale, denoise, adaptive threshold, morph open)
  -> segment dòng/ký tự
  -> classify từng ký tự (CNN)
  -> ghép dòng/text + confidence + bounding box
```

## API

Port mặc định: `5000`

### `GET /health`

Trả về trạng thái service và khả dụng của engine.

Ví dụ response:

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

Upload multipart với field tên `image`.

Query parameter tùy chọn:

- `engine=easyocr` (mặc định)
- `engine=cnn`

Ví dụ:

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

Ví dụ:

```bash
curl -X POST http://localhost:5000/ocr/base64 \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64-string>","engine":"cnn"}'
```

## Cấu trúc response

Các field chung:

- `text`: toàn bộ văn bản nhận dạng
- `lines`: văn bản tách theo dòng
- `words`: đơn vị nhận dạng kèm confidence và `bbox`
- `characters`: field tương thích frontend
- `engine`: engine đã dùng (`easyocr` hoặc `cnn`)

Với CNN sẽ có thêm metadata mức ký tự:

- `line`: chỉ số dòng bắt đầu từ 0
- `space_before`: có khoảng trắng trước ký tự hay không

## Cài đặt

### Yêu cầu

- Python 3.10+
- Node.js 18+ (cho `web/`)

### 1) Cài dependencies Python

```bash
python3 -m pip install -r requirements.txt
```

### 2) (Tùy chọn) Train model CNN

MNIST (chỉ chữ số):

```bash
python train.py
```

EMNIST (chữ số + chữ cái, tải lớn hơn):

```bash
python train.py --dataset emnist
```

Tùy chỉnh tham số train:

```bash
python train.py --dataset emnist --epochs 20 --batch-size 128
```

File sinh ra sẽ lưu trong `saved_models/` (model, labels, training plot).

### 3) Chạy API

Development:

```bash
python app.py
```

Ví dụ production:

```bash
gunicorn -w 2 app:app
```

## Frontend (Trạng thái hiện tại)

Ứng dụng trong `web/` hiện là trang starter của Theme UI và chưa được nối với OCR endpoints.

Chạy frontend:

```bash
cd web
npm install
npm run dev
```

## Ghi chú

- CNN engine trả về `503` nếu chưa có model đã train.
- EasyOCR tự tải weights ở lần chạy đầu tiên.
- Thứ tự ưu tiên load model CNN:
  1. `saved_models/emnist_cnn.keras`
  2. `saved_models/mnist_cnn.keras`