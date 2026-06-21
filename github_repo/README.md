# 🫁 폐렴 X-ray 이진분류 | DACON 대회

> **최종 리더보드: 0.9615** (목표: 0.9700)
> **ConvNeXt-Tiny + DenseNet121 OR 앙상블**

---

## 📋 프로젝트 소개

다콘(DACON) 플랫폼의 폐렴 X-ray 이진분류 대회입니다.
흉부 X-ray 이미지를 입력받아 **정상(0) / 폐렴(1)** 을 판별하는 딥러닝 모델을 개발했습니다.

| 항목 | 내용 |
|------|------|
| 과제 | 이진분류 (정상 vs 폐렴) |
| 평가 지표 | Balanced Accuracy Score |
| 학습 데이터 | 흉부 X-ray 5,216장 |
| 목표 점수 | 0.9700 |
| 최종 달성 | **0.9615** ★ |

---

## 📁 프로젝트 구조

```
pneumonia-xray-classification/
│
├── README.md
├── final_model_OR_ensemble.py        ← 최종 제출 코드 (0.9615)
│
├── experiments/                      ← 실험별 코드 (Try 13~19)
│   ├── try13_convtiny_ep30_patience7.py
│   ├── try14_convtiny_tta10.py
│   ├── try15_convtiny_colojitter_gauss.py
│   ├── try16_convtiny_clahe.py
│   ├── try17_xception.py
│   ├── try18_densenet121.py
│   ├── try19_convtiny_centercrop.py
│   └── extra_balanced_dataset_test.py  ← 외부 데이터셋 검증
│
├── results/
│   └── best_submission_0.9615.csv    ← 최고점 제출 파일
│
└── reports/
    └── 폐렴_Xray_분류_프로젝트_보고서_v2.docx
```

---

## 🏆 전체 실험 결과

| Try | 모델 | 핵심 기법 | 리더보드 |
|-----|------|----------|---------|
| 1 | ResNet18 | Baseline, random split | 0.9327 |
| 2 | EfficientNet-B0 | K-Fold 5겹 + TTA 5회 | 0.9487 |
| 3 | EfficientNet-B0 | + 오버샘플링 | 0.9519 |
| 4 | B0 + B1 앙상블 | 50:50 평균 앙상블 | 0.9583 |
| 5 | B0 + FakeWires | 노이즈 증강 | 0.9455 ↓ |
| 6 | B0/B1 + Mixup | 이미지 혼합 증강 | 0.9359 ↓ |
| 7 | B1 + CLAHE | 대비 향상 전처리 | 0.9487 |
| 8 | EfficientNet-B3 | 더 큰 모델 시도 | 0.9567 |
| 9 | B0+B1+ConvTiny 3모델 | 3모델 균등 앙상블 | 0.9599 |
| **10** | **ConvNeXt-Tiny 단독** | **K-Fold + TTA + 오버샘플링** | **0.9599 ⭐** |
| 11 | ConvNeXt Tiny+Small | 앙상블 | 0.9551 ↓ |
| 12 | ConvTiny + Swin-Tiny | 앙상블 | 0.9551 ↓ |
| 13 | ConvTiny epoch↑ patience↑ | epoch 30, patience 7 | 0.9519 ↓ |
| 14 | ConvTiny TTA 10회 | TTA 5→10 | 0.9519 ↓ |
| 15 | ConvTiny + GaussianBlur | ColorJitter↑ + Blur | 0.9487 ↓ |
| 16 | ConvTiny + CLAHE | 대비 향상 전처리 | 0.9503 ↓ |
| 17 | Xception | 논문 근거 97.97% 모델 | 0.9519 ↓ |
| 18 | DenseNet121 | CheXNet 논문 근거 | 0.9567 |
| 19 | ConvTiny + CenterCrop | 비율 유지 리사이즈 | 0.8910 ↓ |
| **20** | **ConvTiny + DenseNet OR** | **OR 앙상블** | **0.9615 🏆** |

---

## 🔧 최종 모델 구조

```
ConvNeXt-Tiny (28M params)         DenseNet121 (8M params)
       ↓                                   ↓
K-Fold 5겹 학습                     K-Fold 5겹 학습
TTA 5회 추론                        TTA 5회 추론
오버샘플링 적용                      오버샘플링 적용
       ↓                                   ↓
  예측값 (0/1)                        예측값 (0/1)
            ↘                       ↙
              OR 앙상블
         (하나라도 1이면 → 1)
                  ↓
           리더보드 0.9615 🏆
```

---

## ⚙️ 핵심 하이퍼파라미터

```python
IMG_SIZE   = 224
BATCH_SIZE = 32
N_FOLDS    = 5
EPOCHS     = 25
PATIENCE   = 5
LR         = 1e-4
N_TTA      = 5
OPTIMIZER  = Adam
SCHEDULER  = StepLR(step_size=5, gamma=0.5)
LOSS       = CrossEntropyLoss(weight=class_weights)
SEED       = 42
```

---

## 🚀 실행 방법

### 1. 환경 설치
```bash
pip install torch torchvision scikit-learn pandas numpy Pillow
```

### 2. Colab 기준 경로 설정
```python
# final_model_OR_ensemble.py 상단에서 수정
SAVE_DIR = "/content/drive/MyDrive/your_path"
```

### 3. 실행
```python
# Colab GPU 환경에서 실행 (약 90~100분 소요)
# ConvNeXt-Tiny → DenseNet121 → OR 앙상블 순서로 자동 실행
python final_model_OR_ensemble.py
```

---

## 💡 핵심 발견

```
1. val_bacc ≠ 리더보드
   train/val과 test의 도메인 갭 존재
   → val_bacc 높아도 리더보드는 낮을 수 있음

2. 작은 모델이 소규모 데이터에 유리
   ConvNeXt-Tiny(28M) > B3(12M) > Small(50M)
   → 5,216장 데이터셋엔 28M이 최적

3. OR 앙상블이 폐렴 분류에 최적
   폐렴(1)을 정상(0)으로 놓치는 것이 더 큰 손해
   → OR 방식으로 Recall 극대화

4. 증강 강화는 오히려 역효과
   CLAHE, GaussianBlur, FakeWires, Mixup 모두 하락
   → test 분포와 멀어지는 것이 원인

5. CenterCrop 주의
   Resize(256)+CenterCrop(224) 시 쇄골 위 부위 잘림
   → 0.9599에서 0.8910으로 급락
```

---

## 🔬 외부 데이터셋 검증

Kaggle의 **Chest X-Ray Pneumonia Balanced Dataset** (8,530장, 1:1 균형) 으로 검증 결과:

| 모델 | Balanced Accuracy | Accuracy |
|------|------------------|---------|
| ConvNeXt-Tiny | 1.0000 | 1.0000 |
| DenseNet121 | 1.0000 | 1.0000 |
| OR 앙상블 | 1.0000 | 1.0000 |

> ⚠️ test set이 30장으로 소규모라 참고 수준으로 해석 필요

---

## 📚 참고 논문

- **CheXNet** (Stanford, 2017): DenseNet121로 폐렴 X-ray 방사선과 의사 수준 달성
- 동일 데이터셋 관련 연구: CLAHE 적용 CNN → 98.7% / Xception → 97.97%

---

## 🗂️ 데이터셋

- **다콘 대회 데이터**: 비공개
- **외부 검증 데이터**: [Chest X-Ray Pneumonia Balanced Dataset](https://www.kaggle.com/datasets/yusufmurtaza01/chest-xray-pneumonia-balanced-dataset)

---

## 🙏 소감

이전 실습에서 구성한 코드를 기반으로 모델 크기 확장과 전처리 개선부터 시작하였습니다.
팀 토의를 통해 FakeWires 노이즈, 쇄골/레터박스 비율 조정 등 혼자서는 생각하지 못했던 아이디어를 접할 수 있었고,
다양한 모델과 기법을 조합하여 실험하는 좋은 기회가 되었습니다.
데이터의 크기와 특성에 따라 모델 선택과 전처리 방식을 달리해야 한다는 점을 깊이 체감하였습니다.
