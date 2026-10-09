# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Nguyễn Văn Chiến
- MSSV: 2A202602926
- Email: nvchien1904@gmail.com
- Link repo (fork): https://github.com/nvchien19/K4-Track4-Day23-Sensor-Fusion-Student
- Commit hash nộp (`git rev-parse HEAD`): 5ec2591 (code + artifacts commit)

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`:
  `compare`, `[0, 198]`, `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `seed = 0`.
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`:
  `precision = 0.9701`, `recall = 0.7004`, `tp = 519`, `fp = 16`, `fn = 222`.
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`:
  `rmse = 0.15032`, `matches = 502`, `sum_sq_err = 11.34365`, `ghost_track_frames = 0`,
  `missed_gt_frames = 239`, `mean_confirmed_tracks = 2.5226`.
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`:
  `rmse = 0.13587`, `matches = 502`, `sum_sq_err = 9.26681`, `ghost_track_frames = 0`,
  `missed_gt_frames = 239`, `mean_confirmed_tracks = 2.5226`.
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  Hai mode dùng **cùng một bộ track** (lidar khởi tạo/quản lý). Vì vậy `matches`,
  `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks` giống hệt nhau:
  camera không tạo, không xóa và không ghi nhận hit cho track. Khác biệt duy nhất là
  `sum_sq_err`/`rmse`: ở mode `fused`, sau khi lidar update, một lượt camera bổ sung
  update EKF bằng tâm hộp 2D GT FRONT, làm vị trí 3D mượt hơn nên `rmse` giảm
  (0.1503 → 0.1359). `ghost = 0` và `missed_gt_frames = 239` phản ánh recall detector
  còn thấp (0.70): nhiều GT không có detection để khởi tạo track, chứ không phải track bị xóa.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. **Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?**
   Lidar cho `z = [x, y, z]` (3 chiều) ở hệ vehicle: `Sensor.get_hx` biến đổi
   `veh_to_sens`, `H = [R_rot | 0]` nên quan hệ *tuyến tính* với vị trí; `R` là
   `diag(sigma_lidar_x², sigma_lidar_y², sigma_lidar_z²)` đơn vị mét. Camera cho
   `z = [u, v]` (2 chiều) là pixel qua phép chiếu pinhole *phi tuyến*; `H` là
   Jacobian chiếu `∂(u,v)/∂(x,y,z) @ R_rot`, `R` là
   `diag(sigma_cam_i², sigma_cam_j²)` đơn vị pixel². Vì vậy innovation của camera
   nằm trong không gian pixel với thang đo/đơn vị khác hẳn lidar.

2. **Vì sao cần gating Mahalanobis trước khi gán?**
   Khoảng cách Mahalanobis bình phương `γᵀ S⁻¹ γ` (với `γ` = innovation, `S` =
   innovation covariance) chuẩn hóa residual theo độ bất định của cả track lẫn
   measurement. Ngưỡng `chi2.ppf(gating_threshold, dim_meas)` cho phép chỉ nhận cặp
   có residual hợp lý về mặt thống kê, tránh gán nhầm detection xa vào track và
   tránh update làm hỏng state; gating cũng xử lý khác đơn vị giữa lidar (m) và
   camera (pixel) đúng theo từng sensor.

3. **Track-then-fuse hay fuse-then-track? Chỉ ra trên log.**
   Là **track-then-fuse** (measurement-to-track fusion): mỗi frame dự đoán track
   hiện có bằng EKF, rồi lần lượt (a) gán detection lidar vào track và update, (b)
   gán tâm hộp 2D camera vào *cùng* các track đó và update. Log `grade_run.log` cho
   thấy mỗi `(mode, frame)` có đúng một record; ở mode `compare`, các cột detection
   trùng nhau giữa hai mode và mode `fused` chỉ khác `sum_sq_err` → xác nhận fusion
   diễn ra *sau* khi track đã hình thành, không tạo track từ fusion.

4. **Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?**
   Phép chiếu `h(x)` sai lệch hệ thống so với pixel quan sát, nên innovation
   `z − h(x)` của camera có **trung bình khác 0** (bias theo một hướng), khoảng cách
   Mahalanobis tăng đều và gate chi-square thường xuyên loại measurement hợp lệ
   (hoặc nhận nhầm). Trên fusion, điều này đẩy state lệch khỏi lidar, làm `rmse`
   của mode `fused` xấu đi thay vì cải thiện.

5. **Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Vì sao lidar quyết định score/init/delete còn camera chỉ EKF update?**
   Kể cả khi `meas_list` rỗng, `manage_tracks` vẫn phải chạy để ghi nhận *miss*
   trong FOV, cập nhật score và xóa track; hành vi này phụ thuộc sensor đang xét nên
   phải truyền `sensor` tường minh thay vì suy ra từ danh sách rỗng. Chỉ lidar mới
   là sensor "đo tồn tại" (khởi tạo track, +1/window khi hit, −1/window khi miss
   trong FOV, xóa track), còn camera chỉ cung cấp update EKF tinh chỉnh vị trí; nếu
   để camera tham gia scoring/xóa thì các track không thấy trên ảnh sẽ bị xóa sai.

6. **Điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.**
   Khởi tạo: `score = 1/window`, `state = "initialized"`. Mỗi lidar hit cộng
   `1/window` (tối đa 1). `score > confirmed_threshold (0.8)` → `"confirmed"`; track
   đã confirmed **giữ nguyên confirmed** khi bị miss (miss chỉ trừ score). Xóa track
   khi: (OR) `P[0,0] > max_P` hoặc `P[1,1] > max_P`; hoặc track confirmed có
   `score < delete_threshold (0.6)`; hoặc track chưa confirmed có `score <= 0`.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Trợ lý lập trình AI (opencode)
- Dùng cho phần nào (hàm, câu hỏi, debug): Gợi ý/soát lỗi khi implement Part E–H (`kalman.py`, `association.py`, `camera_fusion.py`, `track_management.py`) và tạo notebook Colab.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Chạy `pytest student/tests -q` (128 passed) và chạy `fusion-run-lab --fusion compare --seed 0` trên segment Waymo thật, đối chiếu `metrics.json` với `grade_run.log`.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [ ] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
