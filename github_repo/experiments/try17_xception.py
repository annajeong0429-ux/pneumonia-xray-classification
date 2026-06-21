# ══════════════════════════════════════════════════════════════════
# 폐렴 X-ray 분류 | Xception | Colab 버전
# 논문 근거: 동일 데이터셋 97.97% 달성 모델
# lr=1e-4 | epoch=25 | patience=5 | K-Fold 5 | TTA 5
# ══════════════════════════════════════════════════════════════════

# ── 0. 패키지 설치 ────────────────────────────────────────────────
# Xception은 torchvision에 없으므로 timm 사용
import subprocess
subprocess.run(["pip", "install", "timm", "-q"])

# ── 1. Google Drive 마운트 ────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── 2. 임포트 ─────────────────────────────────────────────────────
import os
import random
import copy
from pathlib import Path

import timm
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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
# ※ 본인 Google Drive 경로에 맞게 수정하세요
BASE_DIR   = Path("/content/drive/MyDrive/dacon/xrfile")
TRAIN_CSV  = BASE_DIR / "train.csv"
TEST_CSV   = BASE_DIR / "test.csv"
SAMPLE_CSV = BASE_DIR / "sample_submission.csv"
TRAIN_IMG  = BASE_DIR / "train"
TEST_IMG   = BASE_DIR / "test"

SAVE_DIR   = Path("/content/drive/MyDrive/dacon/output")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── 5. 하이퍼파라미터 ─────────────────────────────────────────────
# Xception 권장 입력: 299x299
IMG_SIZE   = 299
BATCH_SIZE = 32
N_FOLDS    = 5
EPOCHS     = 25
PATIENCE   = 5
LR         = 1e-4
N_TTA      = 5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("  Xception | K-Fold 5 | TTA 5 | Colab")
print("=" * 60)
print(f"  Device   : {DEVICE}")
print(f"  IMG_SIZE : {IMG_SIZE} (Xception 권장 299x299)")
print(f"  Epochs   : {EPOCHS}  |  Patience : {PATIENCE}")
print(f"  LR       : {LR}  |  TTA : {N_TTA}회")
print(f"  근거     : 논문 동일 데이터셋 Xception 97.97% 달성")

# ── 6. 데이터 로드 ────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f"\n  Train : {len(train_df)}  |  Test : {len(test_df)}")
print(f"  컬럼  : {list(train_df.columns)}")
print(f"  클래스 분포:\n{train_df['label'].value_counts()}")

# ── 7. Transform 정의 ─────────────────────────────────────────────
# Xception은 ImageNet mean/std 동일하게 사용 가능
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

# ── 8. Dataset ────────────────────────────────────────────────────
class XRayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, has_label=True):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.transform = transform
        self.has_label = has_label

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

# ── 10. 모델 빌드 (timm Xception) ────────────────────────────────
def build_model():
    """
    timm의 Xception 사용
    - pretrained=True: ImageNet 사전학습 가중치
    - num_classes=2: 정상 vs 폐렴 이진분류
    """
    model = timm.create_model(
        "xception",
        pretrained=True,
        num_classes=2
    )
    return model.to(DEVICE)

# 모델 구조 확인
print("\n  Xception 모델 로드 중...")
_m = build_model()
total_params = sum(p.numel() for p in _m.parameters()) / 1e6
print(f"  ✓ Xception 파라미터 수: {total_params:.1f}M")
del _m
torch.cuda.empty_cache()

# ── 11. 클래스 가중치 ─────────────────────────────────────────────
def get_class_weights(df):
    counts  = df["label"].value_counts().sort_index()
    total   = len(df)
    weights = total / (len(counts) * counts.values)
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

# ── 12. K-Fold 학습 루프 ──────────────────────────────────────────
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

    tr_loader = DataLoader(
        XRayDataset(tr_df,  TRAIN_IMG, train_tf),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        XRayDataset(val_df, TRAIN_IMG, val_tf),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
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
    test_ds = XRayDataset(test_df, TEST_IMG,
                          transform=tta_tf, has_label=False)

    print(f"  TTA 추론 중 ({N_TTA}회)...", end=" ")
    for tta_i in range(N_TTA):
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=2, pin_memory=True
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
               SAVE_DIR / f"xception_fold{fold}.pth")
    print(f"  모델 저장 완료 → xception_fold{fold}.pth")

    # 메모리 정리
    del model
    torch.cuda.empty_cache()

# ── 13. 최종 결과 요약 ────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Xception | 오버샘플링 + K-Fold {N_FOLDS} + TTA {N_TTA}")
print(f"  lr={LR} | epoch={EPOCHS} | patience={PATIENCE}")
print(f"{'='*60}")
for i, sc in enumerate(fold_scores, 1):
    print(f"  Fold {i}: {sc:.4f}")
print(f"  {'─'*45}")
print(f"  평균 val_bacc : {np.mean(fold_scores):.4f}")
print(f"  표준편차      : {np.std(fold_scores):.4f}")
print(f"\n  현재 최고점(0.9599) 대비: "
      f"{np.mean(fold_scores) - 0.9599:+.4f}")

# ── 14. 제출 파일 생성 ────────────────────────────────────────────
test_probs /= N_FOLDS
all_preds   = test_probs.argmax(axis=1)

submission          = pd.read_csv(SAMPLE_CSV)
submission["label"] = all_preds

out_name = "dodo_project-xception-kfold-tta.csv"
submission.to_csv(SAVE_DIR / out_name, index=False)

print(f"\n  제출 파일 저장 완료 ✓")
print(f"  → {SAVE_DIR / out_name}")
print(f"  예측 분포: {dict(pd.Series(all_preds).value_counts())}")
print(submission.head())

# ── 15. ConvNeXt-Tiny 앙상블 (선택) ──────────────────────────────
# Xception 결과가 좋으면 기존 ConvNeXt-Tiny(0.9599)와 앙상블 가능
# 아래 코드는 두 모델의 확률값을 평균내는 방법
print("""
══════════════════════════════════════════════════════
  💡 앙상블 팁
  Xception 리더보드 > 0.9599 이면:
  → Xception + ConvNeXt-Tiny 앙상블 시도!

  앙상블 방법:
  xception_probs  = test_probs / N_FOLDS  (위에서 계산됨)
  convtiny_probs  = 기존 Try18 확률값 (.npy로 저장해두면 편함)
  ensemble_probs  = (xception_probs + convtiny_probs) / 2
  final_preds     = ensemble_probs.argmax(axis=1)
══════════════════════════════════════════════════════
""")

# Xception 확률값 저장 (나중에 앙상블용)
np.save(SAVE_DIR / "xception_test_probs.npy", test_probs / N_FOLDS)
print("  확률값 저장 완료 → xception_test_probs.npy (앙상블용)")
