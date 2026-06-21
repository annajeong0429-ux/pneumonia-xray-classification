# ══════════════════════════════════════════════════════════════════
# 폐렴 X-ray 분류 | ConvNeXt-Tiny + DenseNet121 OR 앙상블
# Balanced Dataset (8,530장) | 성능 분석 버전
# 1. 모델 저장
# 2. TRAIN+VAL 학습 → TEST 성능 평가
# 3. 오류 분석 (어떤 이미지를 틀렸는지)
# ══════════════════════════════════════════════════════════════════

# ── 1. 임포트 ─────────────────────────────────────────────────────
import os
import random
import copy
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    densenet121, DenseNet121_Weights
)
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score,
    confusion_matrix, classification_report
)
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
BASE      = "/kaggle/input/datasets/yusufmurtaza01/chest-xray-pneumonia-balanced-dataset"
TRAIN_DIR = Path(BASE) / "train"
VAL_DIR   = Path(BASE) / "val"
TEST_DIR  = Path(BASE) / "test"
SAVE_DIR  = Path("/kaggle/working")
SAVE_DIR.mkdir(exist_ok=True)

print("train 폴더:", TRAIN_DIR.exists())
print("val   폴더:", VAL_DIR.exists())
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
print("  ConvNeXt-Tiny + DenseNet121 OR 앙상블")
print("  Balanced Dataset | 성능 분석 버전")
print("=" * 60)
print(f"  Device : {DEVICE}  |  LR : {LR}  |  TTA : {N_TTA}회")

# ── 5. 폴더 → DataFrame 변환 ──────────────────────────────────────
def folder_to_df(folder_path):
    data = []
    for label_name, label_idx in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        class_dir = Path(folder_path) / label_name
        if not class_dir.exists():
            print(f"  ⚠️ 폴더 없음: {class_dir}")
            continue
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                data.append({
                    "filepath" : str(img_file),
                    "label"    : label_idx,
                    "class"    : label_name
                })
    return pd.DataFrame(data).sample(frac=1, random_state=SEED).reset_index(drop=True)

print("\n  데이터 로딩 중...")
train_df     = folder_to_df(TRAIN_DIR)
val_df       = folder_to_df(VAL_DIR)
test_df      = folder_to_df(TEST_DIR)
all_train_df = pd.concat([train_df, val_df]).reset_index(drop=True)

print(f"\n  Train+Val : {len(all_train_df)}장  "
      f"(NORMAL={sum(all_train_df.label==0)}, PNEUMONIA={sum(all_train_df.label==1)})")
print(f"  Test      : {len(test_df)}장  "
      f"(NORMAL={sum(test_df.label==0)}, PNEUMONIA={sum(test_df.label==1)})")

# ── 6. Transform ──────────────────────────────────────────────────
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
    def __init__(self, df, transform=None, has_label=True):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.has_label = has_label

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
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
            parts.append(df[df["label"] == cls].sample(
                max_cnt - cnt, replace=True, random_state=SEED))
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

# ── 9. 모델 빌드 ──────────────────────────────────────────────────
def build_model(model_name):
    if model_name == 'convnext':
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
    else:
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    return m.to(DEVICE)

# ── 10. 클래스 가중치 ─────────────────────────────────────────────
def get_class_weights(df):
    counts  = df["label"].value_counts().sort_index()
    weights = len(df) / (len(counts) * counts.values)
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

# ── 11. K-Fold 학습 함수 ──────────────────────────────────────────
def train_kfold(model_name):
    print(f"\n{'★'*60}")
    print(f"  {model_name.upper()} 학습 시작")
    print(f"{'★'*60}")

    skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []
    test_probs  = np.zeros((len(test_df), 2))

    for fold, (tr_idx, val_idx) in enumerate(
            skf.split(all_train_df, all_train_df["label"]), 1):

        print(f"\n  {'='*50}")
        print(f"  Fold {fold} / {N_FOLDS}")
        print(f"  {'='*50}")

        tr_df = oversample_df(all_train_df.iloc[tr_idx].reset_index(drop=True))
        vl_df = all_train_df.iloc[val_idx].reset_index(drop=True)
        print(f"  Train: {len(tr_df)}  |  Val: {len(vl_df)}")

        tr_loader  = DataLoader(XRayDataset(tr_df, train_tf),
                                batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=4, pin_memory=True)
        val_loader = DataLoader(XRayDataset(vl_df, val_tf),
                                batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=4, pin_memory=True)

        model     = build_model(model_name)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        criterion = nn.CrossEntropyLoss(weight=get_class_weights(tr_df))

        best_bacc, best_state, no_improve = 0.0, None, 0

        for epoch in range(1, EPOCHS + 1):
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

            model.eval()
            preds_all, labels_all = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    preds = model(imgs.to(DEVICE)).argmax(dim=1).cpu().numpy()
                    preds_all.extend(preds)
                    labels_all.extend(labels.numpy())

            bacc = balanced_accuracy_score(labels_all, preds_all)
            flag = "  ← best ★" if bacc > best_bacc else ""
            print(f"  Epoch {epoch:2d}/{EPOCHS}  |  "
                  f"loss: {train_loss/len(tr_loader):.4f}  |  "
                  f"val_bacc: {bacc:.4f}{flag}")

            if bacc > best_bacc:
                best_bacc, best_state, no_improve = bacc, copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"  Early stopping @ epoch {epoch}")
                    break

        fold_scores.append(best_bacc)

        # ── ★ 모델 저장 ───────────────────────────────────────────
        save_path = SAVE_DIR / f"{model_name}_fold{fold}.pth"
        torch.save(best_state, save_path)
        print(f"\n  Fold {fold} best val_bacc: {best_bacc:.4f}")
        print(f"  모델 저장 완료 → {save_path.name}")

        # TTA 추론
        model.load_state_dict(best_state)
        model.eval()
        fold_test_probs = np.zeros((len(test_df), 2))
        test_ds = XRayDataset(test_df, transform=tta_tf, has_label=False)

        print(f"  TTA 추론 ({N_TTA}회)...", end=" ")
        for tta_i in range(N_TTA):
            probs = []
            with torch.no_grad():
                for imgs in DataLoader(test_ds, batch_size=BATCH_SIZE,
                                       shuffle=False, num_workers=4, pin_memory=True):
                    prob = torch.softmax(model(imgs.to(DEVICE)), dim=1).cpu().numpy()
                    probs.append(prob)
            fold_test_probs += np.vstack(probs)
            print(f"{tta_i+1}", end=" " if tta_i < N_TTA-1 else "\n")

        fold_test_probs /= N_TTA
        test_probs      += fold_test_probs

        del model
        torch.cuda.empty_cache()

    test_probs /= N_FOLDS
    print(f"\n  {'='*50}")
    print(f"  {model_name.upper()} 완료")
    for i, sc in enumerate(fold_scores, 1):
        print(f"  Fold {i}: {sc:.4f}")
    print(f"  평균 val_bacc : {np.mean(fold_scores):.4f}")
    print(f"  표준편차      : {np.std(fold_scores):.4f}")

    return test_probs, fold_scores

# ══════════════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════════════
convnext_probs, convnext_scores = train_kfold('convnext')
densenet_probs, densenet_scores = train_kfold('densenet')

# ── 12. OR 앙상블 ─────────────────────────────────────────────────
convnext_preds = convnext_probs.argmax(axis=1)
densenet_preds = densenet_probs.argmax(axis=1)
votes          = convnext_preds + densenet_preds
final_preds    = (votes >= 1).astype(int)
true_labels    = test_df["label"].values

# ── 13. 최종 성능 평가 ────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  최종 성능 평가 (Test Set 기준)")
print(f"{'='*60}")

for name, preds in [("ConvNeXt-Tiny", convnext_preds),
                    ("DenseNet121",   densenet_preds),
                    ("OR 앙상블",     final_preds)]:
    bacc = balanced_accuracy_score(true_labels, preds)
    acc  = accuracy_score(true_labels, preds)
    cm   = confusion_matrix(true_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # 폐렴 재현율
    specificity = tn / (tn + fp)  # 정상 재현율
    print(f"\n  [{name}]")
    print(f"  Balanced Accuracy : {bacc:.4f}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Sensitivity(폐렴) : {sensitivity:.4f}  ← 폐렴을 폐렴으로 맞춘 비율")
    print(f"  Specificity(정상) : {specificity:.4f}  ← 정상을 정상으로 맞춘 비율")
    print(f"  Confusion Matrix  :")
    print(f"              예측 정상  예측 폐렴")
    print(f"    실제 정상  {tn:6d}    {fp:6d}")
    print(f"    실제 폐렴  {fn:6d}    {tp:6d}")

print(f"\n  다콘 대회 최고점 (참고): 0.9615")
print(f"  이번 Balanced Dataset : {balanced_accuracy_score(true_labels, final_preds):.4f}")

# ── 14. Classification Report ─────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Classification Report (OR 앙상블)")
print(f"{'='*60}")
print(classification_report(true_labels, final_preds,
                             target_names=["NORMAL", "PNEUMONIA"]))

# ── 15. 오류 분석 ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  오류 분석")
print(f"{'='*60}")

results_df = test_df.copy()
results_df["convnext_pred"] = convnext_preds
results_df["densenet_pred"] = densenet_preds
results_df["final_pred"]    = final_preds
results_df["correct"]       = (final_preds == true_labels).astype(int)

# 오류 케이스 분류
fn_df = results_df[(results_df.label == 1) & (results_df.final_pred == 0)]  # 폐렴→정상 오분류
fp_df = results_df[(results_df.label == 0) & (results_df.final_pred == 1)]  # 정상→폐렴 오분류

print(f"\n  전체 테스트  : {len(results_df)}장")
print(f"  정답         : {results_df.correct.sum()}장")
print(f"  오답         : {(~results_df.correct.astype(bool)).sum()}장")
print(f"\n  ① 폐렴 → 정상으로 오분류 (False Negative): {len(fn_df)}장  ← 위험!")
print(f"  ② 정상 → 폐렴으로 오분류 (False Positive): {len(fp_df)}장")

# 두 모델 모두 틀린 케이스
both_wrong = results_df[
    (results_df.convnext_pred != results_df.label) &
    (results_df.densenet_pred != results_df.label)
]
print(f"\n  두 모델 모두 틀린 케이스: {len(both_wrong)}장")
print(f"    → 이 이미지들이 가장 어려운 케이스!")

# 한 모델만 틀린 케이스
only_conv_wrong = results_df[
    (results_df.convnext_pred != results_df.label) &
    (results_df.densenet_pred == results_df.label)
]
only_dens_wrong = results_df[
    (results_df.convnext_pred == results_df.label) &
    (results_df.densenet_pred != results_df.label)
]
print(f"  ConvNeXt만 틀린 케이스  : {len(only_conv_wrong)}장")
print(f"  DenseNet만 틀린 케이스  : {len(only_dens_wrong)}장")

# ── 16. 오류 이미지 시각화 ────────────────────────────────────────
def show_error_samples(df, title, max_show=6):
    if len(df) == 0:
        print(f"  {title}: 없음")
        return
    sample = df.sample(min(max_show, len(df)), random_state=SEED)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    axes = axes.flatten()
    for i, (_, row) in enumerate(sample.iterrows()):
        if i >= max_show: break
        img = Image.open(row["filepath"]).convert("RGB")
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(
            f"실제: {'PNEUMONIA' if row.label==1 else 'NORMAL'}\n"
            f"예측: {'PNEUMONIA' if row.final_pred==1 else 'NORMAL'}\n"
            f"Conv:{row.convnext_pred} / Dense:{row.densenet_pred}",
            fontsize=9
        )
        axes[i].axis('off')
    for j in range(i+1, max_show):
        axes[j].axis('off')
    plt.tight_layout()
    save_name = title.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    plt.savefig(SAVE_DIR / save_name, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"  시각화 저장 → {save_name}")

print(f"\n  오류 이미지 시각화 저장 중...")
show_error_samples(fn_df, "False Negative (폐렴을 정상으로 오분류)")
show_error_samples(fp_df, "False Positive (정상을 폐렴으로 오분류)")
show_error_samples(both_wrong, "두 모델 모두 틀린 케이스 (가장 어려운 이미지)")

# ── 17. 결과 CSV 저장 ─────────────────────────────────────────────
results_df.to_csv(SAVE_DIR / "test_results_detail.csv", index=False)

print(f"\n{'='*60}")
print(f"  저장 파일 목록")
print(f"{'='*60}")
for f in sorted(SAVE_DIR.glob("*")):
    print(f"  {f.name}")

print(f"\n  🏁 완료!")
print(f"  Balanced Accuracy: {balanced_accuracy_score(true_labels, final_preds):.4f}")
