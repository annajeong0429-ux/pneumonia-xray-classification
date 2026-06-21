# ══════════════════════════════════════════════════════════════════
# 폐렴 X-ray 분류 | 최종 제출 코드
# ConvNeXt-Tiny + DenseNet121 → OR 앙상블
# 최종 리더보드: 0.9615 ★
# lr=1e-4 | epoch=25 | patience=5 | K-Fold 5 | TTA 5
# ══════════════════════════════════════════════════════════════════

# ── 0. Google Drive 마운트 ────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

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
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    densenet121, DenseNet121_Weights
)
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
SAVE_DIR   = "/content/drive/MyDrive/AI_health_care_04/수업자료/프로젝트 -팀플/개인프로젝트"
DATA_ROOT  = Path(SAVE_DIR)
TRAIN_DIR  = DATA_ROOT / "train"
TEST_DIR   = DATA_ROOT / "test"
TRAIN_CSV  = DATA_ROOT / "train.csv"
TEST_CSV   = DATA_ROOT / "test.csv"
SAMPLE_CSV = DATA_ROOT / "sample_submission.csv"

print("train 폴더:", TRAIN_DIR.exists())
print("test  폴더:", TEST_DIR.exists())

# ── 4. 하이퍼파라미터 ─────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
N_FOLDS    = 5
EPOCHS     = 25
PATIENCE   = 5
LR         = 1e-4
N_TTA      = 5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("  최종 모델: ConvNeXt-Tiny + DenseNet121 OR 앙상블")
print("=" * 60)
print(f"  Device   : {DEVICE}")
print(f"  IMG_SIZE : {IMG_SIZE}  |  BATCH : {BATCH_SIZE}")
print(f"  Epochs   : {EPOCHS}  |  Patience : {PATIENCE}")
print(f"  LR       : {LR}  |  TTA : {N_TTA}회  |  K-Fold : {N_FOLDS}")

# ── 5. 데이터 로드 ────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f"\n  Train : {len(train_df)}  |  Test : {len(test_df)}")
print(f"  컬럼  : {list(train_df.columns)}")
print(f"  클래스 분포:\n{train_df['label'].value_counts()}")

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

# ── 8. 오버샘플링 ─────────────────────────────────────────────────
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

# ── 9. 클래스 가중치 ──────────────────────────────────────────────
def get_class_weights(df):
    counts  = df["label"].value_counts().sort_index()
    total   = len(df)
    weights = total / (len(counts) * counts.values)
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

# ── 10. K-Fold 학습 함수 ──────────────────────────────────────────
def train_kfold(model_name):
    """
    model_name: 'convnext' 또는 'densenet'
    반환: test 확률값 numpy array (shape: [N, 2])
    """
    print(f"\n{'★'*60}")
    print(f"  모델 학습 시작: {model_name.upper()}")
    print(f"{'★'*60}")

    def build_model():
        if model_name == 'convnext':
            m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
        else:  # densenet
            m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            m.classifier = nn.Linear(m.classifier.in_features, 2)
        return m.to(DEVICE)

    skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                   random_state=SEED)
    fold_scores = []
    test_probs  = np.zeros((len(test_df), 2))

    for fold, (tr_idx, val_idx) in enumerate(
            skf.split(train_df, train_df["label"]), 1):

        print(f"\n  {'='*50}")
        print(f"  Fold {fold} / {N_FOLDS}")
        print(f"  {'='*50}")

        tr_df  = train_df.iloc[tr_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        tr_df  = oversample_df(tr_df)
        print(f"  Train: {len(tr_df)}  |  Val: {len(val_df)}")

        tr_loader = DataLoader(
            XRayDataset(tr_df,  TRAIN_DIR, train_tf),
            batch_size=BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            XRayDataset(val_df, TRAIN_DIR, val_tf),
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
                    print(f"  Early stopping @ epoch {epoch}")
                    break

        fold_scores.append(best_bacc)
        print(f"\n  Fold {fold} best val_bacc: {best_bacc:.4f}")

        # TTA 추론
        model.load_state_dict(best_state)
        model.eval()

        fold_test_probs = np.zeros((len(test_df), 2))
        test_ds = XRayDataset(test_df, TEST_DIR,
                              transform=tta_tf, has_label=False)

        print(f"  TTA 추론 ({N_TTA}회)...", end=" ")
        for tta_i in range(N_TTA):
            test_loader = DataLoader(
                test_ds, batch_size=BATCH_SIZE,
                shuffle=False, num_workers=2, pin_memory=True
            )
            probs = []
            with torch.no_grad():
                for imgs in test_loader:
                    imgs = imgs.to(DEVICE)
                    prob = torch.softmax(model(imgs), dim=1).cpu().numpy()
                    probs.append(prob)
            fold_test_probs += np.vstack(probs)
            print(f"{tta_i+1}", end=" " if tta_i < N_TTA-1 else "\n")

        fold_test_probs /= N_TTA
        test_probs      += fold_test_probs

        torch.save(best_state,
                   DATA_ROOT / f"{model_name}_fold{fold}.pth")
        del model
        torch.cuda.empty_cache()

    # 결과 요약
    test_probs /= N_FOLDS
    print(f"\n  {'='*50}")
    print(f"  {model_name.upper()} 학습 완료")
    for i, sc in enumerate(fold_scores, 1):
        print(f"  Fold {i}: {sc:.4f}")
    print(f"  평균 val_bacc: {np.mean(fold_scores):.4f}")
    print(f"  표준편차     : {np.std(fold_scores):.4f}")

    return test_probs

# ══════════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════════

# ── 11. ConvNeXt-Tiny 학습 ────────────────────────────────────────
convnext_probs = train_kfold('convnext')

# ── 12. DenseNet121 학습 ──────────────────────────────────────────
densenet_probs = train_kfold('densenet')

# ── 13. OR 앙상블 ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  OR 앙상블: ConvNeXt-Tiny + DenseNet121")
print(f"{'='*60}")

convnext_preds = convnext_probs.argmax(axis=1)
densenet_preds = densenet_probs.argmax(axis=1)

# OR: 하나라도 폐렴(1)이면 → 1
votes      = convnext_preds + densenet_preds
final_preds = (votes >= 1).astype(int)

print(f"  ConvNeXt 예측 분포 : 0={sum(convnext_preds==0)}, 1={sum(convnext_preds==1)}")
print(f"  DenseNet 예측 분포 : 0={sum(densenet_preds==0)}, 1={sum(densenet_preds==1)}")
print(f"  두 모델 불일치 샘플: {sum(convnext_preds != densenet_preds)}개")
print(f"  OR 앙상블 최종    : 0={sum(final_preds==0)}, 1={sum(final_preds==1)}")

# ── 14. 제출 파일 생성 ────────────────────────────────────────────
submission          = pd.read_csv(SAMPLE_CSV)
submission["label"] = final_preds

out_name = "dodo_project-convtiny-densenet-OR-final.csv"
submission.to_csv(DATA_ROOT / out_name, index=False)

print(f"\n  제출 파일 저장 완료 ✓")
print(f"  → {DATA_ROOT / out_name}")
print(submission.head())
print(f"\n  🏆 목표: 0.9700  |  최종 달성: 0.9615")
