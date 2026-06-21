# ══════════════════════════════════════════════════════════════════
# 폐렴 X-ray 분류 | ConvNeXt-Tiny + CLAHE | Kaggle 버전
# Try 18 기준 (최고점 0.9599) + CLAHE 전처리 추가
# lr=1e-4 | epoch=25 | patience=5 | K-Fold 5 | TTA 5
# ══════════════════════════════════════════════════════════════════

# ── 1. 라이브러리 설치 (Kaggle 환경) ─────────────────────────────
import subprocess
subprocess.run(["pip", "install", "opencv-python-headless", "-q"])

# ── 2. 임포트 ─────────────────────────────────────────────────────
import os
import random
import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

# ── 3. 시드 고정 ──────────────────────────────────────────────────
SEED = 42

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything(SEED)

# ── 4. 경로 설정 ──────────────────────────────────────────────────
BASE_DIR   = Path("/kaggle/input/xrfile")
TRAIN_CSV  = BASE_DIR / "train.csv"
TEST_CSV   = BASE_DIR / "test.csv"
SAMPLE_CSV = BASE_DIR / "sample_submission.csv"
TRAIN_IMG  = BASE_DIR / "train"
TEST_IMG   = BASE_DIR / "test"

SAVE_DIR   = Path("/kaggle/working")
SAVE_DIR.mkdir(exist_ok=True)

# ── 5. 하이퍼파라미터 (Try 18 원본값 유지) ────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
N_FOLDS    = 5
EPOCHS     = 25      # Try 18 원본
PATIENCE   = 5       # Try 18 원본
LR         = 1e-4
N_TTA      = 5       # Try 18 원본
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("  ConvNeXt-Tiny + CLAHE 전처리 | Kaggle")
print("=" * 60)
print(f"  Device   : {DEVICE}")
print(f"  Epochs   : {EPOCHS}  |  Patience : {PATIENCE}")
print(f"  LR       : {LR}  |  TTA : {N_TTA}회")
print(f"  변경사항 : CLAHE 전처리 추가 (train/val/test 모두 적용)")
print(f"  근거     : 동일 데이터셋 논문에서 CLAHE → 98.7% 달성")

# ── 6. CLAHE 전처리 함수 ──────────────────────────────────────────
# CLAHE: 이미지를 작은 타일로 나눠 각 타일별로 히스토그램 평활화
# → X-ray의 밝기 불균일 문제 해결, 폐 내부 구조 더 선명하게
def apply_clahe(pil_img, clip_limit=2.0, tile_grid=(8, 8)):
    """
    PIL Image → CLAHE 적용 → PIL Image 반환
    - clip_limit : 대비 제한값 (너무 크면 노이즈 증폭, 2.0이 안전)
    - tile_grid  : 타일 크기 (8x8이 X-ray에 최적)
    """
    # PIL → numpy (RGB)
    img_np = np.array(pil_img)

    # RGB → LAB 색공간 변환 (L채널에만 CLAHE 적용)
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)

    # CLAHE 적용 (L채널만)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_clahe = clahe.apply(l)

    # LAB → RGB 복원
    img_clahe = cv2.merge([l_clahe, a, b])
    img_rgb   = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)

    return Image.fromarray(img_rgb)

# ── 7. Transform 정의 ─────────────────────────────────────────────
# CLAHE는 Dataset __getitem__에서 이미지 로드 직후 적용
# → 모든 증강 전에 먼저 대비 향상

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1,
                           saturation=0.1, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

tta_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── 8. Dataset (CLAHE 내장) ───────────────────────────────────────
class XRayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None,
                 has_label=True, use_clahe=True):
        self.df         = df.reset_index(drop=True)
        self.img_dir    = Path(img_dir)
        self.transform  = transform
        self.has_label  = has_label
        self.use_clahe  = use_clahe   # ★ CLAHE 적용 여부

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if "img_path" in row.index:
            fname = row["img_path"]
        elif "filename" in row.index:
            fname = row["filename"]
        else:
            fname = row.iloc[0]

        fpath = self.img_dir / Path(fname).name
        img   = Image.open(fpath).convert("RGB")

        # ★ CLAHE 적용 (로드 직후, 증강 전)
        if self.use_clahe:
            img = apply_clahe(img, clip_limit=2.0, tile_grid=(8, 8))

        if self.transform:
            img = self.transform(img)

        if self.has_label:
            return img, int(row["label"])
        return img

# ── 9. 오버샘플링 ─────────────────────────────────────────────────
def oversample_df(df):
    counts  = df["label"].value_counts()
    max_cnt = counts.max()
    parts   = [df]
    for cls, cnt in counts.items():
        if cnt < max_cnt:
            diff = max_cnt - cnt
            parts.append(
                df[df["label"] == cls].sample(diff, replace=True,
                                               random_state=SEED)
            )
    return (pd.concat(parts)
              .sample(frac=1, random_state=SEED)
              .reset_index(drop=True))

# ── 10. 모델 빌드 ─────────────────────────────────────────────────
def build_model():
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)
    return model.to(DEVICE)

# ── 11. 클래스 가중치 ─────────────────────────────────────────────
def get_class_weights(df):
    counts  = df["label"].value_counts().sort_index()
    total   = len(df)
    weights = total / (len(counts) * counts.values)
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

# ── 12. 데이터 로드 ───────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f"\n  Train : {len(train_df)}  |  Test : {len(test_df)}")
print(f"  컬럼  : {list(train_df.columns)}")
print(f"  클래스 분포:\n{train_df['label'].value_counts()}")

# CLAHE 적용 샘플 확인
print("\n  CLAHE 전처리 테스트 중...")
try:
    sample_row = train_df.iloc[0]
    fname = sample_row["img_path"] if "img_path" in sample_row.index else sample_row.iloc[0]
    test_img = Image.open(TRAIN_IMG / Path(fname).name).convert("RGB")
    clahe_img = apply_clahe(test_img)
    print(f"  ✓ CLAHE 적용 성공! 원본 크기: {test_img.size}")
except Exception as e:
    print(f"  ✗ CLAHE 오류: {e}")

# ── 13. K-Fold 학습 루프 ──────────────────────────────────────────
skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=SEED)
fold_scores = []
test_probs  = np.zeros((len(test_df), 2))

for fold, (tr_idx, val_idx) in enumerate(
        skf.split(train_df, train_df["label"]), 1):

    print(f"\n{'='*60}")
    print(f"  Fold {fold} / {N_FOLDS}")
    print(f"{'='*60}")

    tr_df  = train_df.iloc[tr_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    tr_df = oversample_df(tr_df)
    print(f"  Train (오버샘플링 후) : {len(tr_df)}  |  Val : {len(val_df)}")

    # ★ use_clahe=True — 모든 split에 CLAHE 적용
    tr_loader = DataLoader(
        XRayDataset(tr_df,  TRAIN_IMG, train_tf, use_clahe=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        XRayDataset(val_df, TRAIN_IMG, val_tf, use_clahe=True),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    model     = build_model()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss(weight=get_class_weights(tr_df))

    best_bacc  = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):

        # Train
        model.train()
        train_loss = 0.0
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs  = imgs.to(DEVICE)
                preds = model(imgs).argmax(dim=1).cpu().numpy()
                preds_all.extend(preds)
                labels_all.extend(labels.numpy())

        bacc = balanced_accuracy_score(labels_all, preds_all)
        flag = "  ← best ★" if bacc > best_bacc else ""
        print(f"  Epoch {epoch:2d}/{EPOCHS}  |  "
              f"loss: {train_loss/len(tr_loader):.4f}  |  "
              f"val_bacc: {bacc:.4f}{flag}")

        if bacc > best_bacc:
            best_bacc  = bacc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping @ epoch {epoch} "
                      f"(patience={PATIENCE})")
                break

    fold_scores.append(best_bacc)
    print(f"\n  Fold {fold} best val_bacc : {best_bacc:.4f}")

    # ── TTA 추론 ───────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()

    fold_test_probs = np.zeros((len(test_df), 2))
    # ★ test도 CLAHE 적용
    test_ds = XRayDataset(test_df, TEST_IMG,
                          transform=tta_tf,
                          has_label=False, use_clahe=True)

    print(f"  TTA 추론 중 ({N_TTA}회)...", end=" ")
    for tta_i in range(N_TTA):
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=4, pin_memory=True
        )
        batch_probs = []
        with torch.no_grad():
            for imgs in test_loader:
                imgs = imgs.to(DEVICE)
                prob = torch.softmax(model(imgs), dim=1).cpu().numpy()
                batch_probs.append(prob)
        fold_test_probs += np.vstack(batch_probs)
        print(f"{tta_i+1}", end=" " if tta_i < N_TTA-1 else "\n")

    fold_test_probs /= N_TTA
    test_probs      += fold_test_probs

    torch.save(best_state,
               SAVE_DIR / f"convtiny_clahe_fold{fold}.pth")
    print(f"  모델 저장 완료 → convtiny_clahe_fold{fold}.pth")

# ── 14. 최종 결과 요약 ────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ConvNeXt-Tiny + CLAHE | Try18 기준 + CLAHE 전처리")
print(f"  lr={LR} | epoch={EPOCHS} | patience={PATIENCE} | TTA={N_TTA}")
print(f"{'='*60}")
for i, sc in enumerate(fold_scores, 1):
    print(f"  Fold {i}: {sc:.4f}")
print(f"  {'─'*45}")
print(f"  평균 val_bacc : {np.mean(fold_scores):.4f}")
print(f"  표준편차      : {np.std(fold_scores):.4f}")
print(f"\n  현재 최고점(0.9599) 대비: "
      f"{np.mean(fold_scores) - 0.9599:+.4f}")

# ── 15. 제출 파일 생성 ────────────────────────────────────────────
test_probs /= N_FOLDS
all_preds   = test_probs.argmax(axis=1)

submission          = pd.read_csv(SAMPLE_CSV)
submission["label"] = all_preds

out_name = "dodo_project-convtiny-clahe-kfold-tta.csv"
submission.to_csv(SAVE_DIR / out_name, index=False)

print(f"\n  제출 파일 저장 완료 ✓  →  {out_name}")
print(f"  예측 분포: {dict(pd.Series(all_preds).value_counts())}")
print(submission.head())
