# FoundIR 사용 가이드

FoundIR는 이미지 복원(deblurring, denoising, deraining, dehazing, low-light enhancement 등)을 위한 foundation model입니다. 이 가이드는 사전 학습 모델을 다운로드하여 로컬 이미지를 평가하는 방법과, 직접 데이터를 사용하여 Fine-tuning하는 방법을 설명합니다.

---

## 1. 사전 학습 모델 다운로드 및 설치

### 1.1 Generalist 모델 (주 모델)

사전 학습된 Generalist 모델은 다음 링크에서 다운로드할 수 있습니다:

- **모델 링크**: https://github.com/House-Leo/FoundIR/releases/download/Premodel/model-2000.pt
- **저장 위치**: 프로젝트 루트의 `./premodel` 폴더

```bash
# premodel 폴더가 없다면 생성
mkdir -p premodel

# wget으로 다운로드 (Linux/macOS)
wget -O premodel/model-2000.pt https://github.com/House-Leo/FoundIR/releases/download/Premodel/model-2000.pt

# Windows PowerShell의 경우
# Invoke-WebRequest -Uri "https://github.com/House-Leo/FoundIR/releases/download/Premodel/model-2000.pt" -OutFile "premodel/model-2000.pt"
```

### 1.2 Specialist 모델 (선택 사항)

Generalist 모델의 출력을 refinement하기 위한 specialist 모델도 제공됩니다:

- **Lowlight 모델**: 어두운 이미지 enhancement
- **Weather 모델**: 비, 안개 등 날씨 관련 이미지 복원

Specialist 모델은 다음 링크에서 다운로드할 수 있습니다 (Google Drive):

- **GT 데이터셋**: https://drive.google.com/file/d/1KjRZcyA1THRzHZhX2yTGMtOdUW_wuGsI/view?usp=sharing
- **LQ (입력) 데이터셋**: https://drive.google.com/file/d/1wOaquAjnuzCh6Jv3CJz76mgnx4nfZgBY/view?usp=sharing
- **FoundIR 결과**: https://drive.google.com/file/d/1MLSV4OPvictpKYsDdqF7LcjnIebYYNUw/view?usp=sharing
- **타 방법 결과**: https://pan.baidu.com/s/1ORZVrHkgsVMymSSI4Yng-g?pwd=b6qb

### 1.3 모델 종류 설명

| 모델 | 용도 | 설명 |
|------|------|------|
| **Generalist** | 다양한 이미지 복원 | 단일 모델로 blur, noise, rain, fog, low-light 등 처리 |
| **Lowlight Specialist** | 어두운 이미지 | Generalist 출력을進一步 개선 |
| **Weather Specialist** | 비/안개 이미지 | 날씨 관련 degradations 추가 개선 |

---

## 2. 로컬 이미지로 평가하기

### 2.1 데이터셋 폴더 구조

평가 원하는 이미지를 다음과 같이 구성하세요:

```
dataset/
└── LQ/                  # Low-Quality (입력 이미지)
    ├── 0001.png
    ├── 0002.png
    ├── 0003.tif        # TIFF/TIF 이미지 भी 지원
    ├── 0004.tiff
    └── ...

# GT (Ground Truth) 폴더가 있으면 PSNR, SSIM 등 metric 계산 가능
dataset/
├── LQ/                  # 입력 이미지
│   ├── 0001.png
│   ├── 0002.tif        # TIFF/TIF 지원
│   └── ...
└── GT/                 # 정답 이미지 (선택)
    ├── 0001.png
    ├── 0002.tif
    └── ...
```

> **지원 이미지 형식**: PNG, JPG, JPEG, BMP, PPM, TIFF, TIF

### 2.1.1 TIFF/TIF 이미지 사용

FoundIR는 TIFF/TIF 이미지를 기본적으로 지원합니다. `data/image_folder.py`에서 지원되는 형식으로 등록되어 있습니다:

```python
'.tif', '.TIF', '.tiff', '.TIFF'
```

학습/평가 시 이미지 형식을 지정할 필요가 없습니다. 폴더에 TIFF/TIF 이미지를 넣으면 자동으로 로드됩니다.

### 2.2 평가 실행

```bash
# 기본 명령어
python test.py --dataroot ./dataset --meta None
```

이 명령어는 `./dataset/LQ` 폴더의 모든 이미지를 평가합니다.

### 2.3 주요 파라미터

`test.py`에서調整 가능한 주요 파라미터:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--dataroot` | ./MillionIRData/Test | 데이터셋 경로 |
| `--meta` | None | 메타 정보 파일 (사용자 데이터는 None) |
| `--load_size` | 256 | 이미지 로드 시 크기 |
| `--crop_size` | 256 | cropping 크기 |
| `--bsize` | 2 | 배치 크기 |

GPU 메모리가 부족한 경우 `test.py`의 `crop_size`를 줄이세요 (line 102).

### 2.4 출력 결과

평가 결과는 `./results` 폴더에 저장됩니다:

```
results/
└── *.png   # 복원된 이미지
```

---

## 3. 내 데이터셋으로 Fine-tuning

### 3.1 학습 데이터셋 구조

학습 데이터를 다음과 같이 구성하세요:

```
dataset/
└── Train/
    ├── GT/              # High-Quality (정답 이미지)
    │   ├── 0001.png
    │   ├── 0002.tif    # TIFF/TIF 이미지 지원
    │   ├── 0003.tiff
    │   └── ...
    └── LQ/              # Low-Quality (입력 이미지)
        ├── 0001.png
        ├── 0002.tif
        └── ...
```

> **TIFF/TIF 사용**: 학습 데이터에도 TIFF/TIF 이미지를 사용할 수 있습니다. 이미지 형식은 자동으로 인식됩니다.

### 3.2 메타 정보 파일 생성

Fine-tuning을 위해서는 메타 정보 파일이 필요합니다. 파일 형식은 다음과 같습니다:

```
{GT 경로}|{LQ 경로}|{degradation 타입}
```

예시:
```
./dataset/Train/GT/0001.png|./dataset/Train/LQ/0001.png|blur
./dataset/Train/GT/0002.png|./dataset/Train/LQ/0002.png|noise
./dataset/Train/GT/0003.tif|./dataset/Train/LQ/0003.tif|rain
./dataset/Train/GT/0004.tiff|./dataset/Train/LQ/0004.tiff|lowlight
```

### 3.3 Degradation 타입

지원하는 degradation 타입:

- `blur`: 블러 제거
- `noise`: 노이즈 제거
- `rain`: 비 제거
- `snow`: 눈 제거
- `fog`: 안개 제거
- `lowlight`: 저조도 이미지 개선

### 3.4 학습 실행

**1단계: 단일 degradation 학습 (권장)**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=7689 train.py --meta ./your_single_train_meta_info.txt
```

**2단계: 전체 degradation 학습 (선택)**

1단계完成后, `train.py`의 line 48-49 주석을 풀고:

```python
# train_num_steps = 500000 # for single degradation training
train_num_steps = 2000000 # for all training
```

2단계 학습 실행:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=7689 train.py --meta ./your_train_meta_info.txt
```

### 3.5 주요 학습 파라미터

`train.py`에서調整 가능한 파라미터:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--batch_size` | 80 | 배치 크기 |
| `--meta` | ./MillionIRData_train_meta_info.txt | 메타 정보 파일 경로 |
| `--load_size` | 268 | 이미지 로드 크기 |
| `--crop_size` | 256 | cropping 크기 |

코드 내 주요 설정 (train.py):

```python
image_size = 512           # 학습 이미지 크기
train_lr = 1e-4            # 학습률
train_num_steps = 500000   # 학습 스텝 수
ema_decay = 0.995          # EMA decay rate
```

### 3.6 체크포인트 저장

학습 결과는 `./ckpt_single_multi` 폴더(또는 설정한 `results_folder`)에 저장됩니다:

```
ckpt_single_multi/
├── model-{step}.pt    # 모델 체크포인트
└── ...
```

### 3.7 저장된 모델로 평가

체크포인트로 평가하려면 `test.py`의 `trainer.load(2000)` 부분을 수정:

```python
# 2000 대신 원하는 step 번호
trainer.load(원하는_스텝번호)
```

---

## 4. 평가 결과 해석

### 4.1 Metrics 종류

FoundIR 평가에 사용되는 주요 Metrics:

| Metric | 의미 | 높을수록 좋음 |
|--------|------|---------------|
| **PSNR** | Peak Signal-to-Noise Ratio | ✓ |
| **SSIM** | Structural Similarity | ✓ |
| **LPIPS** | Learned Perceptual Image Patch Similarity | ✗ (낮을수록) |
| **FID** | Fréchet Inception Distance | ✗ (낮을수록) |
| **CLIP-IQA** | CLIP-based Image Quality Assessment | ✓ |
| **MANIQA** | Multi-dimension Attention Image Quality Assessment | ✓ |
| **MUSIQ** | Multi-scale Image Quality Transformer | ✓ |
| **NIQE** | Natural Image Quality Evaluator | ✗ (낮을수록) |
| **NIMA** | Neural Image Quality Assessment | ✓ |

### 4.2 Metrics 계산

GT 이미지가 있는 경우, `cal_metrics.py`로_metrics를 계산할 수 있습니다:

```bash
python cal_metrics.py --inp_imgs ./results --gt_imgs ./dataset/GT --log ./metrics_log
```

### 4.3 Specialist 모델 적용 (선택)

특정 이미지에 specialist 모델을 적용하여 추가 개선:

```bash
# Lowlight 개선
cd ./specialist_model
python inference_lowlight.py

# Weather 개선 (비, 안개 등)
python inference_weather.py
```

적용 대상 (paper 기준):
- **Weather model**: 0501-0700, 1051-1100
- **Lowlight model**: 0701-0800, 1101-1250, 1301-1500

---

## 5. Troubleshooting

### GPU 메모리 부족

`test.py`의 `crop_size`를 줄이세요:

```python
# test.py line 102
trainer.test(last=True, crop_phase='im2overlap', crop_size=512, crop_stride=256)
```

### CUDA 에러

GPU 번호 지정:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py
```

### 데이터 로드 에러

이미지 형식이 PNG, JPG, JPEG, BMP, PPM, TIFF, TIF인지 확인하세요.

### TIFF/TIF 이미지 관련

- TIFF/TIF 이미지는 기본 지원됩니다 (별도 설정 불필요)
- 16-bit TIFF는 8-bit로 자동 변환됩니다
- 출력은 항상 PNG 형식으로 저장됩니다

---

## 6. 참고 자료

- **Paper**: https://arxiv.org/abs/2412.01427
- **Project Page**: https://www.foundir.net
- **GitHub**: https://github.com/House-Leo/FoundIR
- **Contact**: haoli@njust.edu.cn, chenxiang@njust.edu.cn
