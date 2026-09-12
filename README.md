# ScribbleMIS

Benchmark scribble-supervised segmentation 3D trên 3 dataset **ACDC**, **MSCMR**, **WORD** (ScribbleBench), so sánh 4 phương pháp:

- **pCE** — partial cross-entropy baseline (chỉ học trên voxel có scribble)
- **CycleMix** (Zhang & Zhuang, CVPR 2022)
- **DMSPS** (Han et al., MedIA 2024) — train 2 giai đoạn
- **SDT-Net** (Nguyen et al. 2026) — dual-teacher/single-student

Toàn bộ đều dùng chung 1 backbone `UNet3D` và cùng 1 bộ dữ liệu 3D, chỉ khác nhau ở cách dùng scribble để tạo loss/pseudo-label.

## Cấu trúc project

```
code/
  dataloader/
    scribblebench_3d.py     # Dataset 3D dùng chung (ACDC/MSCMR/WORD), augment, remap nhãn WORD
  networks/
    unet_3d.py               # UNet3D (backbone chính, dùng cho pCE/CycleMix/SDT-Net)
    unet_cct_3d.py            # UNet3D dual-decoder (dùng cho DMSPS)
    resunet_3d.py, nnunet_3d.py
  utils/
    sliding_window_3d.py      # Suy luận sliding-window cho volume lớn
    cyclemix.py, dmsps.py, sdtnet.py   # Loss/thuật toán riêng của từng method
  train/
    common_3d.py               # Hạ tầng train dùng chung (split, checkpoint, validate, pCE loss)
    train_pce_3d.py             # Baseline
    train_cyclemix_3d.py
    train_dmsps_3d.py            # --stage 1 hoặc --stage 2
    train_sdtnet_3d.py
    run_baselines.sh              # Chạy full: train + test cả 4 method x 3 dataset
    legacy_splits.py               # Split train/val/test cố định (không tự sample)
  test/
    test_pce_3d.py, test_dmsps_3d.py   # Evaluator trên tập test chính thức
    metrics_3d.py                       # Dice / HD95 / ASSD dùng chung
    append_metrics_csv.py                 # Ghi kết quả evaluator vào CSV tổng hợp
    test_*.py (còn lại)                    # Unit test (chạy CPU, nhanh)

dataset/ScribbleBench/<ACDC|MSCMR|WORD>/   # Dữ liệu (không commit lên git)
  imagesTr/ labelsTr/ labelsTr_dense/ imagesTs/ labelsTs/

checkpoints/   # Output lúc train: best.pth, last.pth, split.json, log, tensorboard
results/       # Output lúc test: metrics.json + baselines_summary.csv
```

## Cài đặt

```bash
pip install -r requirements.txt
```

Dữ liệu cần đặt sẵn ở `dataset/ScribbleBench/<ACDC|MSCMR|WORD>/` (xem cấu trúc thư mục ở trên).

## Chạy thử nhanh (kiểm tra code không lỗi)

```bash
python code/test/test_networks_3d.py       # test network, chạy CPU vài giây
python code/test/test_cyclemix_utils.py
python code/test/test_dmsps_utils.py
python code/test/test_sdtnet_utils.py
python code/test/test_metrics_3d.py
python code/test/test_word_label_remap.py
python code/test/test_common_3d.py
```

## Train từng method riêng lẻ

```bash
# pCE baseline
python code/train/train_pce_3d.py --dataset ACDC --amp

# CycleMix
python code/train/train_cyclemix_3d.py --dataset ACDC --amp

# DMSPS (2 giai đoạn, giai đoạn 2 cần checkpoint tốt nhất của giai đoạn 1)
python code/train/train_dmsps_3d.py --dataset ACDC --stage 1 --amp
python code/train/train_dmsps_3d.py --dataset ACDC --stage 2 --amp \
  --init_checkpoint checkpoints/ScribbleBench_DMSPS/ACDC/stage1/best.pth

# SDT-Net
python code/train/train_sdtnet_3d.py --dataset ACDC --amp
```

Thay `--dataset ACDC` bằng `MSCMR` hoặc `WORD` để chạy dataset khác. Mỗi lệnh tự tạo `checkpoints/ScribbleBench_<Method>/<Dataset>/{best.pth,last.pth,train.log,...}`.

Xem hết các tham số có thể chỉnh (batch size, patch size, learning rate, số iteration...):

```bash
python code/train/train_pce_3d.py --help
```

## Test 1 checkpoint đã train

```bash
# pCE / CycleMix / SDT-Net đều dùng chung evaluator này
python code/test/test_pce_3d.py --checkpoint checkpoints/ScribbleBench_pCE/ACDC/best.pth

# DMSPS dùng evaluator riêng (network dual-decoder)
python code/test/test_dmsps_3d.py --checkpoint checkpoints/ScribbleBench_DMSPS/ACDC/stage2/best.pth
```

Kết quả (Dice/HD95/ASSD từng lớp + trung bình) được ghi ra `results/.../metrics.json`.

## Chạy full benchmark (train + test tất cả 4 method x 3 dataset trong 1 lệnh)

```bash
bash code/train/run_baselines.sh
```

Chạy xong sẽ có 1 file CSV tổng hợp toàn bộ kết quả: `results/baselines_summary.csv`.

Có thể chỉnh qua biến môi trường, ví dụ:

```bash
# chỉ chạy ACDC, batch size 4, tắt AMP
SCRIBBLE_DATASETS=ACDC SCRIBBLE_BATCH_SIZE=4 SCRIBBLE_AMP_FLAG="" bash code/train/run_baselines.sh
```

Xem đầu file `code/train/run_baselines.sh` để biết đầy đủ các biến môi trường có thể chỉnh (số dataset chạy, batch size, AMP, thư mục output...).
