# ══════════════════════════════════════════════════════════════════
# 당뇨망막병증 다콘 | ConvNeXt-Tiny | Kaggle
# Try 23: lr=1e-4 | epoch=30 | patience=7 | K-Fold 5 | TTA 5
# ══════════════════════════════════════════════════════════════════

# ── 1. 임포트 ─────────────────────────────────────────────────────
import os
import random
import copy
from pathlib import Path

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

# ── 2. 시드 고정 ──────────────────────────────────────────────────
SEED = 42

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything(SEED)

# ── 3. 경로 설정 ──────────────────────────────────────────────────
BASE_DIR   = Path("/kaggle/input/xrfile")          # ★ 수정
TRAIN_CSV  = BASE_DIR / "train.csv"
TEST_CSV   = BASE_DIR / "test.csv"
SAMPLE_CSV = BASE_DIR / "sample_submission.csv"
TRAIN_IMG  = BASE_DIR / "train"
TEST_IMG   = BASE_DIR / "test"

SAVE_DIR   = Path("/kaggle/working")
SAVE_DIR.mkdir(exist_ok=True)

# ── 4. 하이퍼파라미터 ─────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
N_FOLDS    = 5
EPOCHS     = 30        # ★ 25 → 30
PATIENCE   = 7         # ★  5 →  7
LR         = 1e-4
N_TTA      = 5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device   : {DEVICE}")
print(f"Epochs   : {EPOCHS}  |  Patience: {PATIENCE}  |  LR: {LR}")
print(f"IMG_SIZE : {IMG_SIZE}  |  BATCH: {BATCH_SIZE}  |  TTA: {N_TTA}")

# ── 5. 데이터 로드 ────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f"\nTrain : {len(train_df)}  |  Test : {len(test_df)}")
print(f"train.csv 컬럼: {list(train_df.columns)}")
print(f"test.csv  컬럼: {list(test_df.columns)}")
print(f"클래스 분포:\n{train_df['label'].value_counts()}")

# ── 6. Transform 정의 ─────────────────────────────────────────────
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

# ── 7. Dataset ────────────────────────────────────────────────────
class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, has_label=True):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # CSV 컬럼명 자동 대응
        # train.csv에 'img_path' 또는 'filename' 컬럼이 있는 경우 모두 처리
        if "img_path" in row.index:
            fname = row["img_path"]
        elif "filename" in row.index:
            fname = row["filename"]
        else:
            # 첫 번째 컬럼을 파일명으로 사용 (ID 컬럼 등)
            fname = row.iloc[0]

        # 파일명만 있는 경우 img_dir과 합치기
        # 예: "00001.png" → /kaggle/input/xrfile/train/00001.png
        fpath = self.img_dir / Path(fname).name
        img   = Image.open(fpath).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.has_label:
            return img, int(row["label"])
        return img

# ── 8. 오버샘플링 ─────────────────────────────────────────────────
def oversample_df(df):
    """소수 클래스를 다수 클래스 수만큼 복제 (train fold만 적용)"""
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

# ── 9. 모델 빌드 ──────────────────────────────────────────────────
def build_model():
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 2)
    return model.to(DEVICE)

# ── 10. 클래스 가중치 ─────────────────────────────────────────────
def get_class_weights(df):
    counts  = df["label"].value_counts().sort_index()
    total   = len(df)
    weights = total / (len(counts) * counts.values)
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

# ── 11. K-Fold 학습 루프 ──────────────────────────────────────────
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

    # 오버샘플링 — train fold만 적용
    tr_df = oversample_df(tr_df)
    print(f"  Train (오버샘플링 후): {len(tr_df)}  |  Val: {len(val_df)}")

    # DataLoader
    tr_loader = DataLoader(
        RetinopathyDataset(tr_df,  TRAIN_IMG, train_tf),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        RetinopathyDataset(val_df, TRAIN_IMG, val_tf),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # 모델 / 옵티마이저 / 스케줄러 / 손실함수
    model     = build_model()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss(weight=get_class_weights(tr_df))

    best_bacc  = 0.0
    best_state = None
    no_improve = 0

    # ── Epoch 루프 ─────────────────────────────────────────────────
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
                imgs   = imgs.to(DEVICE)
                preds  = model(imgs).argmax(dim=1).cpu().numpy()
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
    test_ds = RetinopathyDataset(test_df, TEST_IMG,
                                 transform=tta_tf, has_label=False)

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

    fold_test_probs /= N_TTA
    test_probs      += fold_test_probs

    # Fold별 모델 저장 (선택)
    torch.save(best_state,
               SAVE_DIR / f"convtiny_fold{fold}_ep30_p7.pth")
    print(f"  모델 저장 완료 → convtiny_fold{fold}_ep30_p7.pth")

# ── 12. 최종 결과 요약 ────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  ConvNeXt-Tiny | lr={LR} | epoch={EPOCHS} | patience={PATIENCE}")
print(f"{'='*55}")
for i, sc in enumerate(fold_scores, 1):
    print(f"  Fold {i}: {sc:.4f}")
print(f"  {'─'*40}")
print(f"  평균 val_bacc : {np.mean(fold_scores):.4f}")
print(f"  표준편차      : {np.std(fold_scores):.4f}")
print(f"\n  현재 최고점(0.9599) 대비: "
      f"{np.mean(fold_scores) - 0.9599:+.4f}")

# ── 13. 제출 파일 생성 ────────────────────────────────────────────
test_probs /= N_FOLDS
all_preds   = test_probs.argmax(axis=1)

submission         = pd.read_csv(SAMPLE_CSV)
submission["label"] = all_preds

out_name = "dodo_project-convtiny-lr1e4-ep30-p7.csv"
submission.to_csv(SAVE_DIR / out_name, index=False)

print(f"\n  제출 파일 저장 완료 ✓  →  {out_name}")
print(submission.head())

print("\n저장된 파일 목록:")
for f in sorted(SAVE_DIR.glob("*.csv")):
    print(f"  {f.name}")
