# ScribbleMIS

Benchmark scribble-supervised segmentation trên 3 dataset **ACDC**, **MSCMR**, **WORD** (ScribbleBench), so sánh 7 phương pháp:

- **pCE** — partial cross-entropy baseline (chỉ học trên voxel có scribble)
- **CycleMix** (Zhang & Zhuang, CVPR 2022)
- **DMSPS** (Han et al., MedIA 2024) — train 2 giai đoạn
- **SDT-Net** (Nguyen et al. 2026) — dual-teacher/single-student
- **VoxTrust-3D** (phương pháp đề xuất, `VoxTrust3D_CVPR2026_Proposed_Method.pdf`) — student + EMA teacher (Mean Teacher), calibration quyết định pseudo-label nào được học
- **EFFDNet** (Liu et al., MICCAI 2025) — Mean Teacher + Foreground-Background Separation Loss (FBSL, contrastive) + Foreground Augmentation with Diverse Context (FADC, copy-paste augmentation)
- **ModelMix** (Zhang & Patel, MICCAI 2024) — phương pháp duy nhất train **2 dataset cùng lúc**: trộn ngẫu nhiên 1 layer encoder giữa 2 model riêng biệt của ACDC và MSCMR (2 dataset cùng 4-class cardiac label space), regularize model trộn phải khớp với model gốc của từng task. Không chạy trên WORD vì không có dataset "cùng họ" để ghép cặp.

ACDC và MSCMR (MRI tim, spacing rất anisotropic) train theo kiểu **2D slice** (backbone `UNet2D`), đúng protocol chuẩn trong literature scribble-supervision (WSL4MIS, DMSPS, CycleMix, ScribFormer) — dự đoán từng lát cắt rồi ghép lại thành volume 3D lúc evaluate. WORD (CT bụng, spacing gần isotropic) vẫn train theo kiểu **3D volume** với backbone `VNet3D`. 6/7 phương pháp dùng chung 2 pipeline này (1 dataset/lần chạy); riêng ModelMix luôn train ACDC+MSCMR đồng thời trong 1 lần chạy.

## Cấu trúc project

```
code/
  dataloader/
    scribblebench_3d.py     # Dataset 3D theo case (ACDC/MSCMR/WORD), augment, remap nhãn WORD
    scribblebench_2d.py      # Dataset 2D theo slice (ACDC/MSCMR), dựng trên scribblebench_3d.py
  networks/
    unet_2d.py                # UNet2D / UNetCCT2D (backbone ACDC/MSCMR)
    vnet_3d.py                 # VNet3D / VNetCCT3D (backbone WORD)
    unet_3d.py, unet_cct_3d.py, resunet_3d.py, nnunet_3d.py   # Không còn dùng trực tiếp trong train script nào
  utils/
    sliding_window_3d.py      # Suy luận sliding-window cho volume lớn (WORD)
    cyclemix.py, dmsps.py, sdtnet.py, voxtrust3d.py, effdnet.py   # Loss/thuật toán từng method, dùng chung cho cả 2D và 3D
    modelmix.py                 # Riêng cho ModelMix (chỉ 2D): image/model mixup, trộn layer encoder, xoay ảnh
  train/
    common_3d.py                # Hạ tầng dùng chung: split, checkpoint, pCE loss, validate (WORD)
    common_2d.py                 # validate_2d()/predict_volume_2d(): suy luận từng slice rồi ghép lại (ACDC/MSCMR)
    legacy_splits.py               # Split train/val/test cố định (không tự sample)
    train_pce_2d.py / train_cyclemix_2d.py / train_dmsps_2d.py / train_sdtnet_2d.py /
      train_voxtrust3d_2d.py / train_effdnet_2d.py   # Chỉ chạy ACDC/MSCMR
    train_pce_3d.py / train_cyclemix_3d.py / train_dmsps_3d.py / train_sdtnet_3d.py /
      train_voxtrust3d_3d.py / train_effdnet_3d.py   # Chỉ chạy WORD
    train_modelmix_2d.py          # Không có --dataset, luôn train ACDC+MSCMR cùng lúc, ra 2 checkpoint
    run_baselines.sh              # Chạy full: train + test cả 7 method x 3 dataset (ModelMix riêng ACDC+MSCMR)
  test/
    test_pce_2d.py, test_dmsps_2d.py   # Evaluator cho ACDC/MSCMR (suy luận từng slice) -- cũng dùng cho EFFDNet/ModelMix
    test_pce_3d.py, test_dmsps_3d.py   # Evaluator cho WORD (sliding-window) -- cũng dùng cho EFFDNet
    metrics_3d.py                       # Dice / HD95 / ASSD dùng chung cho cả 4 evaluator trên
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
python code/test/test_networks_2d.py        # UNet2D, chạy CPU vài giây
python code/test/test_networks_3d.py
python code/test/test_vnet_3d.py             # VNet3D/VNetCCT3D (backbone WORD)
python code/test/test_scribblebench_2d.py    # dataset + augmentation 2D
python code/test/test_cyclemix_utils.py
python code/test/test_dmsps_utils.py
python code/test/test_sdtnet_utils.py
python code/test/test_voxtrust3d_utils.py
python code/test/test_effdnet_utils.py
python code/test/test_modelmix_utils.py
python code/test/test_train_pce_2d.py
python code/test/test_train_dmsps_2d.py
python code/test/test_train_voxtrust3d_2d.py
python code/test/test_train_effdnet_3d.py
python code/test/test_train_modelmix_2d.py
python code/test/test_common_2d.py
python code/test/test_common_3d.py
python code/test/test_metrics_3d.py
python code/test/test_word_label_remap.py
```

## Train từng method riêng lẻ

ACDC/MSCMR dùng script `_2d.py`, WORD dùng script `_3d.py`:

```bash
# pCE baseline
python code/train/train_pce_2d.py --dataset ACDC --amp    # ACDC | MSCMR
python code/train/train_pce_3d.py --dataset WORD --amp    # chỉ WORD

# CycleMix
python code/train/train_cyclemix_2d.py --dataset ACDC --amp
python code/train/train_cyclemix_3d.py --dataset WORD --amp

# DMSPS (2 giai đoạn, giai đoạn 2 cần checkpoint tốt nhất của giai đoạn 1)
python code/train/train_dmsps_2d.py --dataset ACDC --stage 1 --amp
python code/train/train_dmsps_2d.py --dataset ACDC --stage 2 --amp \
  --init_checkpoint checkpoints/ScribbleBench_DMSPS/ACDC/stage1/best.pth

# SDT-Net
python code/train/train_sdtnet_2d.py --dataset ACDC --amp

# VoxTrust-3D (phương pháp đề xuất)
python code/train/train_voxtrust3d_2d.py --dataset ACDC --amp

# EFFDNet
python code/train/train_effdnet_2d.py --dataset ACDC --amp
python code/train/train_effdnet_3d.py --dataset WORD --amp

# ModelMix (không có --dataset, luôn train ACDC+MSCMR cùng lúc)
python code/train/train_modelmix_2d.py --amp
# -> checkpoints/ScribbleBench_ModelMix/ACDC/best.pth và .../MSCMR/best.pth
```

Thay `--dataset ACDC` bằng `MSCMR` cho các script `_2d.py`; script `_3d.py` chỉ nhận `--dataset WORD`. Mỗi lệnh tự tạo `checkpoints/ScribbleBench_<Method>/<Dataset>/{best.pth,last.pth,train.log,...}`.

Xem hết các tham số có thể chỉnh (batch size, patch size, learning rate, số iteration...):

```bash
python code/train/train_pce_2d.py --help
```

## Test 1 checkpoint đã train

```bash
# pCE / CycleMix / SDT-Net / VoxTrust-3D / EFFDNet / ModelMix đều dùng chung evaluator này
python code/test/test_pce_2d.py --checkpoint checkpoints/ScribbleBench_pCE/ACDC/best.pth
python code/test/test_pce_3d.py --checkpoint checkpoints/ScribbleBench_pCE/WORD/best.pth

# DMSPS dùng evaluator riêng (network dual-decoder)
python code/test/test_dmsps_2d.py --checkpoint checkpoints/ScribbleBench_DMSPS/ACDC/stage2/best.pth
python code/test/test_dmsps_3d.py --checkpoint checkpoints/ScribbleBench_DMSPS/WORD/stage2/best.pth
```

Kết quả (Dice/HD95/ASSD từng lớp + trung bình) được ghi ra `results/.../metrics.json`.

## Chạy full benchmark (train + test tất cả 7 method x 3 dataset trong 1 lệnh)

```bash
bash code/train/run_baselines.sh
```

Script tự route ACDC/MSCMR qua pipeline 2D, WORD qua pipeline 3D, và chạy thêm 1 block riêng cho ModelMix (ACDC+MSCMR, bỏ qua nếu 1 trong 2 dataset không nằm trong `SCRIBBLE_DATASETS`). Chạy xong sẽ có 1 file CSV tổng hợp toàn bộ kết quả: `results/baselines_summary.csv`.

Có thể chỉnh qua biến môi trường, ví dụ:

```bash
# chỉ chạy ACDC, batch size 4, tắt AMP
SCRIBBLE_DATASETS=ACDC SCRIBBLE_BATCH_SIZE=4 SCRIBBLE_AMP_FLAG="" bash code/train/run_baselines.sh
```

Xem đầu file `code/train/run_baselines.sh` để biết đầy đủ các biến môi trường có thể chỉnh (số dataset chạy, batch size, AMP, thư mục output...).
