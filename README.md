# 🚦 Hệ thống phát hiện phương tiện vượt đèn đỏ

Đây là dự án sử dụng mô hình YOLOv10 để phát hiện phương tiện, kết hợp với mô hình tùy chỉnh để nhận diện tín hiệu đèn giao thông (đỏ, vàng, xanh). Hệ thống sẽ:

- Phát hiện phương tiện trong video
- Nhận diện trạng thái đèn giao thông
- Chọn vùng quan tâm (ROI) bằng cách click chuột
- Lưu ảnh các phương tiện vi phạm đèn đỏ

---

## 📁 Cấu trúc dự án

```
project/
│
├── tracker.py                # File chứa class Tracker để theo dõi đối tượng
├── yolov10s.pt               # Mô hình YOLOv10 phát hiện phương tiện
├── train14/weights/best.pt   # Mô hình tùy chỉnh phát hiện đèn giao thông
├── coco.txt                  # Danh sách class tương ứng với mô hình yolov10s.pt
├── tr.mp4                    # Video đầu vào
├── saved_images/             # Thư mục tự tạo để lưu ảnh vi phạm
├── main.py                   # File chính xử lý
└── README.md                 # File hướng dẫn (chính là file này)
```

---

## ⚙️ Yêu cầu cài đặt

Cài đặt các thư viện cần thiết bằng lệnh sau:

```bash
pip install -r requirements.txt
```

### Nội dung `requirements.txt`

```
opencv-python
ultralytics==8.0.20
roboflow
numpy
cvzone
```

> ⚠️ **Lưu ý:** Phải dùng `opencv-python` thay vì `opencv-python-headless` vì chương trình có sử dụng giao diện hiển thị (`cv2.imshow`, chuột, vẽ...).

---

## 🚀 Cách sử dụng

1. **Chuẩn bị**:
   - Đảm bảo có file mô hình `yolov10s.pt` dùng để phát hiện phương tiện.
   - Đảm bảo có mô hình phát hiện đèn tại `train14/weights/best.pt`.
   - File `coco.txt` chứa danh sách các class tương ứng với mô hình phương tiện.
   - File video đầu vào là `tr.mp4` hoặc chỉnh sửa tên trong `main.py`.

2. **Chạy chương trình**:

```bash
python main.py
```

3. **Chọn vùng (ROI)**:
   - Chương trình sẽ hiển thị khung hình đầu tiên.
   - Bước 1: Click 4 điểm để chọn vùng chứa đèn giao thông.
   - Bước 2: Click 4 điểm để chọn vùng kiểm tra vi phạm (vùng vượt đèn).
   - Sau khi chọn xong, video sẽ tự chạy.

4. **Kết quả**:
   - Nếu phương tiện **vượt đèn đỏ**, ảnh sẽ được lưu vào thư mục `saved_images/YYYY-MM-DD/`.
   - Nhấn `q` để thoát chương trình.

---

## 📌 Ghi chú

- Mô hình phát hiện đèn phải có các class `"red"`, `"yellow"`, `"green"` tương ứng với trạng thái đèn.
- Chỉ kiểm tra vi phạm đối với class `"car"` (ô tô).
- Mỗi phương tiện đều được gán một `ID` để theo dõi xuyên suốt video.

---

## 🧠 Tín dụng

- YOLOv10 từ Ultralytics
- Roboflow dùng để huấn luyện mô hình đèn (nếu có)
- OpenCV dùng để xử lý hình ảnh và GUI
- `cvzone` giúp hiển thị khung chữ đẹp và dễ nhìn hơn

---

## 🛠️ Gợi ý phát triển thêm

- [ ] Lưu log vi phạm vào file CSV
- [ ] Hỗ trợ thêm các loại phương tiện như xe máy, xe tải...
- [ ] Thêm giao diện chọn vùng ROI trực quan hơn
