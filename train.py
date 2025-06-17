import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from model import S3DNEW

FPS = 25
WINDOW = 2*FPS
STRIDE = FPS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
NUM_WORKERS = 1
PIN_MEMORY = True
NUM_EPOCHS = 70
LEARNING_RATE = 1e-3/2

TRAIN_DIR = "data_train_short"
VAL_DIR = "data_test_short"
TRAIN_LABELS = "train_labels.json"
VAL_LABELS = "test_labels.json"
def time_to_sec(time):
    h, m, s = map(int, time.split(":"))
    return h * 3600 + m * 60 + s

from sklearn.cluster import KMeans
import numpy as np

class VideoWindowDataset(Dataset):
    def __init__(self, video_dir, annotation_file,
                 transform=None,
                 n_clusters=4,
                 cluster_weights=None,
                 min_valid=True,
                 apply_weights=True,
                 check_broken=True,
                 teaser_window_only=True,
                 teaser_margin_seconds=15):

        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        self.video_dir = video_dir
        self.transform = transform
        self.samples = []
        self.weights = []

        durations = []
        valid_annotations = {}
        for vid, data in self.annotations.items():
            start_sec = time_to_sec(data["start"])
            end_sec = time_to_sec(data["end"])

            if min_valid and end_sec <= start_sec:
                continue

            durations.append([end_sec - start_sec])
            valid_annotations[vid] = data

        if not durations:
            raise ValueError("Нет валидных аннотаций.")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_ids = kmeans.fit_predict(durations)

        if cluster_weights is None:
            cluster_weights = [1.0, 0.75, 0.5, 0.25][:n_clusters]

        vid_to_weight = {
            vid: cluster_weights[cluster_ids[i]]
            for i, vid in enumerate(valid_annotations.keys())
        }
        for vid, data in valid_annotations.items():
            video_subdir = os.path.join(video_dir, vid)
            if not os.path.isdir(video_subdir):
                continue
            video_file = next((f for f in os.listdir(video_subdir) if f.endswith(".mp4")), None)
            if not video_file:
                continue

            video_path = os.path.join(video_subdir, video_file)
            if check_broken and not self._is_video_readable(video_path):
                print(f"[SKIP] Сломаное или несчитываемое видео: {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / FPS
            cap.release()

            start_sec = time_to_sec(data["start"])
            end_sec = time_to_sec(data["end"])
            label_frames = set(range(int(start_sec * FPS), int(end_sec * FPS)))

            if teaser_window_only:
                center_sec = (start_sec + end_sec) / 2
                seg_start_sec = max(0, center_sec - teaser_margin_seconds)
                seg_end_sec = min(duration_sec, center_sec + teaser_margin_seconds)

                start_frame_limit = int(seg_start_sec * FPS)
                end_frame_limit = int(seg_end_sec * FPS)

                for start_frame in range(start_frame_limit, end_frame_limit - WINDOW + 1, STRIDE):
                    end_frame = start_frame + WINDOW
                    label = int(any(f in label_frames for f in range(start_frame-WINDOW, end_frame)))
                    sample_weight = vid_to_weight[vid] if apply_weights else 1.0
                    self.samples.append({
                        "path": video_path,
                        "start_frame": start_frame,
                        "label": label
                    })
                    self.weights.append(sample_weight)
            else:
                for start_frame in range(0, total_frames - WINDOW + 1, STRIDE):
                    end_frame = start_frame + WINDOW
                    label = int(any(f in label_frames for f in range(start_frame-WINDOW, end_frame)))
                    sample_weight = vid_to_weight[vid] if apply_weights else 1.0
                    self.samples.append({
                        "path": video_path,
                        "start_frame": start_frame,
                        "label": label
                    })
                    self.weights.append(sample_weight)

    def _is_video_readable(self, path):
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return False

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                count += 1

            cap.release()
            return count == total and total > 0
        except Exception as e:
            print(f"[ERROR] ошибка считывания {path}: {e}")
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cap = cv2.VideoCapture(sample["path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample["start_frame"])

        frames = []
        for _ in range(WINDOW):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.tensor(frame).permute(2, 0, 1) / 255.0
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError(f"Нет кадров из видео: {sample['path']}")

        while len(frames) < WINDOW:
            frames.append(frames[-1].clone())

        clip = torch.stack(frames).permute(1, 0, 2, 3)
        label = torch.tensor([sample["label"]], dtype=torch.float32)
        weight = self.weights[idx]
        return clip, label, weight


from tqdm import tqdm

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    loop = tqdm(loader, desc="Train", leave=False)

    for x, y, w in loop:
        x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)

        losses = criterion(out, y)
        loss = (losses.view(-1) * w).mean()

        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(out) > 0.5).float()
        acc = (preds == y).float().mean().item()

        total_loss += loss.item()
        total_acc += acc

        loop.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{acc:.4f}"
        })

    return total_loss / len(loader), total_acc / len(loader)


def validate(model, loader, criterion):
    model.eval()
    total_loss, total_acc, total_iou = 0.0, 0.0, 0.0
    loop = tqdm(loader, desc="Val", leave=False)

    with torch.no_grad():
        for x, y, w in loop:
            x, y, w = x.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
            out = model(x)

            losses = criterion(out, y)
            loss = (losses.view(-1) * w).mean()

            preds = (torch.sigmoid(out) > 0.5).float()
            acc = (preds == y).float().mean().item()

            intersection = (preds * y).sum(dim=1)
            union = ((preds + y) > 0).float().sum(dim=1)
            iou = (intersection / (union + 1e-8)).mean().item()

            total_loss += loss.item()
            total_acc += acc
            total_iou += iou

            loop.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.4f}",
                "iou": f"{iou:.4f}"
            })

    n = len(loader)
    return total_loss / (n+1), total_acc / (n+1), total_iou / (n+1)


def main():


    train_dataset = VideoWindowDataset(TRAIN_DIR, TRAIN_LABELS)
    val_dataset = VideoWindowDataset(VAL_DIR, VAL_LABELS)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    model = S3DNEW().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2*2)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc,val_iou = validate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}: "
              f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}|"
              f"Val iou{val_iou:.4f}")

        torch.save(model.state_dict(), f"model_epoch{epoch+1}.pth")


if __name__ == "__main__":
    main()
