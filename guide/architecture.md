# FoundIR 모델 아키텍처

이 문서는 FoundIR의 모델 아키텍처에 대한 상세한 기술적 설명을 제공합니다.

---

## 1. 전체 개요

FoundIR는 **조건부 Diffusion 모델**을 기반으로 한 이미지 복원 foundation model입니다.

### 1.1 아키텍처 유형

| 구성 | 유형 | 설명 |
|------|------|------|
| **Diffusion 유형** | Plain Diffusion | VAE 없이 직접 처리 |
| **백본 네트워크** | U-Net (CNN) | ResNetBlock + Attention |
| **조건부 생성** | Image-to-Image | LQ → GT 변환 |

### 1.2 파이프라인

```
입력 이미지 (LQ)
    ↓
조건부 Diffusion Process
    ↓
U-Net (CNN + Self-Attention)
    ↓
Residual Prediction (잔차 예측)
    ↓
복원된 이미지 (GT 예측)
```

---

## 2. Diffusion 모델

### 2.1 ResidualDiffusion 클래스

위치: `src/model.py:622-1159`

```python
class ResidualDiffusion(nn.Module):
    def __init__(
        self,
        model,              # U-Net 백본
        image_size,         # 이미지 크기
        timesteps=1000,     # Diffusion steps
        objective='pred_res',  # 예측 대상
        condition=True,     # 조건부 여부
        ...
    )
```

### 2.2 Diffusion Schedule

주요 파라미터:

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `timesteps` | 1000 | Diffusion 시퀀스 길이 |
| `sampling_timesteps` | 4 | 실제 샘플링 시 사용 step 수 |
| `delta_end` | 1.4e-3 | 잔차 스케일 파라미터 |
| `sum_scale` | 0.01 | 노이즈 스케일 |

### 2.3 Loss 함수

```python
loss_type = 'l1'  # L1 또는 L2 loss 사용
```

### 2.4 Sampling 방법

- **DDIM Sampling**: 빠른 추론 가능
- **Full Sampling**:高品质 결과

---

## 3. U-Net 백본

### 3.1 Unet 클래스

위치: `src/model.py:362-505`

전통적인 U-Net 구조를 따르며, 다음과 같은 구성요소를 포함합니다:

```python
class Unet(nn.Module):
    def __init__(
        self,
        dim=64,                    #基础 차원
        dim_mults=(1, 2, 4, 8),    # 각 stage의 차수 multiplier
        channels=3,                # 입력 채널
        resnet_block_groups=8,     # GroupNorm 그룹 크기
        condition=False,           # 조건부 여부
    )
```

### 3.2 인코더 (Downsampling)

4개의 downsampling stage:

| Stage | 차원 (dim=64) | 크기 |
|-------|---------------|------|
| 1 | 64 | H/2 × W/2 |
| 2 | 128 | H/4 × W/4 |
| 3 | 256 | H/8 × W/8 |
| 4 | 512 | H/16 × W/16 |

각 stage 구성:
1. **ResNetBlock** (×2): 시간 임베딩 조건부
2. **LinearAttention**: 효율적인 어텐션
3. **Downsample**: spatial 크기 축소

### 3.3 디코더 (Upsampling)

4개의 upsampling stage (인코더와 대칭):

| Stage | 차원 | 크기 |
|-------|------|------|
| 1 | 512 | H/16 × W/16 |
| 2 | 256 | H/8 × W/8 |
| 3 | 128 | H/4 × W/4 |
| 4 | 64 | H/2 × W/2 |

### 3.4 미들 블록

```python
self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
```

---

## 4. 핵심 구성 요소

### 4.1 ResNetBlock

위치: `src/model.py:275-300`

```python
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim else None
        
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
```

특징:
- **GroupNorm**: 8개 그룹 사용
- **SiLU 활성화 함수**
- **시간 조건부**: sinusoidal position embedding通过 MLP

### 4.2 Attention 모듈

#### Standard Attention

```python
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        self.heads = heads
        self.scale = dim_head ** -0.5
```

#### LinearAttention

효율적인 어텐션 메커니즘:

```python
class LinearAttention(nn.Module):
    # Q, K, V를 분리하지 않고 효율적으로 계산
    # O(N × C) 복잡도 (표준 어텍션은 O(N² × C))
```

### 4.3 Weight Standardized Conv2d

위치: `src/model.py:177-192`

```python
class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
```

### 4.4 Layer Normalization

```python
class LayerNorm(nn.Module):
    def __init__(self, dim):
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
```

---

## 5. 시간 임베딩

### 5.1 SinusoidalPosEmb

```python
class SinusoidalPosEmb(nn.Module):
    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
```

### 5.2 Time MLP

```python
self.time_mlp = nn.Sequential(
    sinu_pos_emb,           # Sinusoidal embedding
    nn.Linear(fourier_dim, time_dim),
    nn.GELU(),
    nn.Linear(time_dim, time_dim)
)
```

---

## 6. 조건부 이미지 처리

### 6.1 조건부 입력

`condition=True`일 때:

```python
if self.condition:
    x_in = torch.cat((x, x_input), dim=1)  # [noisy_img, condition_img]
else:
    x_in = x
```

### 6.2 예측 목표

| objective | 설명 |
|-----------|------|
| `pred_res` | 잔차 (LQ - GT) 예측 |
| `pred_noise` | 노이즈 예측 |
| `pred_res_noise` | 잔차 + 노이즈 예측 |

FoundIR는 `pred_res` (잔차 예측) 방식을 사용합니다:

```
 GT = LQ - Residual
```

---

## 7. Trainer 클래스

위치: `src/model.py:1160-`

### 7.1 주요 기능

| 기능 | 설명 |
|------|------|
| **EMA** | Exponential Moving Average |
| **Mixed Precision** | FP16 훈련 지원 |
| **Distributed Training** | Accelerate 기반 멀티 GPU |
| **Gradient Accumulation** | 큰 배치 사이즈 시뮬레이션 |

### 7.2 체크포인트 저장

```python
def save(self, milestone):
    data = {
        'step': self.step,
        'model': self.accelerator.get_state_dict(self.model),
        'opt0': self.opt0.state_dict(),
        'ema': self.ema.state_dict(),
        'scaler': self.accelerator.scaler.state_dict()
    }
    torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
```

---

## 8. SEM 이미지를 위한 수정

### 8.1 채널 수 변경

기본값: 3채널 (RGB)

SEM 이미지는 그레이스케일이므로 1채널로 변경:

```python
# train.py 또는 test.py에서
model = UnetRes(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    channels=1,  # 1채널로 변경
    ...
)
```

### 8.2 이미지 전처리

SEM 이미지는 16-bit인 경우가 많으므로 8-bit로 변환 필요:

```python
# 16-bit → 8-bit 변환
img = (img / 256).astype(np.uint8)
# 또는
img = img.astype(np.float32) / 65535.0 * 255.0
```

---

## 9. U-Net 선택 이유: 왜 Transformer보다 U-Net인가?

### 9.1Foundation Model의 의미

"Foundation Model"은 반드시 Transformer를 의미하지 않습니다.

> **Foundation Model** = 대규모 다양한 데이터로 학습되어 **다양한 태스크에 적응** 가능한 모델

FoundIR의 경우:
- ✅ 100만 이상의 실제 이미지 데이터로 학습
- ✅ 단일 모델로 blur, noise, rain, fog, lowlight 등 20가지 이상의 복원 태스크 처리
- ❌ Transformer 아키텍처 사용 안 함 (U-Net 사용)

### 9.2 논문에서 Transformer 모델들과의 성능 비교

FoundIR는 SOTA를 달성했습니다:

| 모델 | 아키텍처 | FoundIR보다 좋은가? |
|------|----------|---------------------|
| **Restormer** | Transformer (Uformer) | ❌ |
| **TransWeather** | Transformer | ❌ |
| **X-Restormer** | Transformer | ❌ |
| **PromptIR** | CNN + Prompts | ❌ |
| **DiffIR** | Diffusion + CNN | ❌ |
| **DiffUIR** | Diffusion | ❌ |
| **DA-CLIP** | CLIP-based | ❌ |
| **InstructIR** | Transformer + InstructBLIP | ❌ |
| **SUPIR** | Diffusion + CLIP | ❌ |

### 9.3 U-Net으로 충분한 이유

#### 1) Diffusion이 이미 Global Context를 처리

```
Diffusion Process:
- Iterative denozing으로 점진적으로 복원
- 각 step에서 전체 이미지 정보가 순환
- U-Net은 local 특징 추출에 집중, global은 diffusion이 담당
```

#### 2) Residual Prediction (잔차 예측)

FoundIR는 **GT를 예측하는 것이 아니라 LQ - GT (잔차)**를 예측합니다:
- 입력 이미지가 직접 condition으로 제공됨
- 네트워크는 노이즈/劣化 성분만 예측
- 전체 이미지 생성보다 더 단순한 문제

#### 3) 조건부 Diffusion 수식 (Paper Equation 1)

```
I_t = (ᾱ_t - β̄_t)I_LQ + (1 - ᾱ_t)I_HQ + δ̄_t·ε
```

**핵심**: 입력 이미지 I_LQ가 직접 condition으로 들어가므로, 네트워크가 "무엇을 복원해야 하는지" 이미 알고 있습니다.

#### 4) 데이터 스케일이 더 중요한 요소

FoundIR의 핵심 기여:
> **데이터 스케일** (100만 이미지) + **Incremental Learning**

```
Model capacity < Data capacity
U-Net으로 충분한 representation learning 가능
→ 더 많은 데이터가 더 중요한 요소
```

### 9.4 Transformer가 더 유리한 경우

| 상황 | 권장 |
|------|------|
| 텍스트/프롬프트 기반 조건 | ✅ Transformer |
| 다양한 task自适应 | ✅ Transformer |
| 장면 이해가 중요한 경우 | ✅ Transformer |
| **입력이 직접 condition (Image-to-Image)** | ❌ Transformer 불필요 |

FoundIR는 **Image-to-Image** 태스크로, 입력이 직접 condition으로 제공되므로 U-Net으로 충분합니다.

---

## 10. FoundIR-V2 (2025.12)

### 10.1 FoundIR-V2 개요

| 항목 | FoundIR-V1 | FoundIR-V2 |
|------|------------|------------|
| **아키텍처** | CNN U-Net + Diffusion | CNN U-Net + Diffusion + MoE |
| **특징** | 단일 Generalist | Mixture-of-Experts (MoE) |
| **태스크** | 20+ | 50+ 서브 태스크 |
| **데이터** | Million-scale | 데이터 mixture 최적화 |

### 10.2 핵심创新

- **데이터 균형 스케줄링**: 다양한 degradation 유형의 데이터 비율 최적화
- **MoE 기반 스케줄러**: 각 복원 태스크에 최적화된 diffusion prior 할당
- **Paper**: https://arxiv.org/abs/2512.09282

---

## 11. 참고 문헌

- **U-Net**: Convolutional Networks for Biomedical Image Segmentation
- **Group Normalization**: https://arxiv.org/abs/1803.08494
- **Weight Standardization**: https://arxiv.org/abs/1903.10520
- **Diffusion Models**: DDPM, DDIM, Score-based models
- **FoundIR Paper**: https://arxiv.org/abs/2412.01427
- **FoundIR-V2 Paper**: https://arxiv.org/abs/2512.09282
