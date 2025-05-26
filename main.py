# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import argparse
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import wandb

# ------------------------- Seed & Data Augmentation -------------------------
myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(myseed)

mean = [0.5554, 0.4511, 0.3440]
std = [0.2308, 0.2415, 0.2404]

tta_transforms = [
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.90, 1.10)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
]

train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_tfm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

valid_tfm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


# ------------------------- Dataset & Model -------------------------
class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files=None, is_test=False):
        super(FoodDataset, self).__init__()
        self.path = path
        self.is_test = is_test
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files is not None:
            self.files = files
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        file_id = os.path.basename(fname).split(".")[0]
        if self.is_test:
            return im, -1, file_id
        else:
            label = int(fname.split("/")[-1].split("_")[0])
            return im, label

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.backbone = models.resnet34(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 11)
        )

    def forward(self, x):
        return self.backbone(x)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# ------------------------- Loss Functions -------------------------
class LabelSmoothingFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        self.reduction = reduction
    def forward(self, logits, target):
        num_classes = logits.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        focal_factor = (1 - probs).pow(self.gamma)
        loss = -self.alpha * focal_factor * true_dist * log_probs
        loss = loss.sum(dim=1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
# ------------------------- SAM Optimizer -------------------------
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.rho = rho
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        loss = closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

base_optimizer = torch.optim.Adam
# ------------------------- Main -------------------------
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--exp_name", type=str, default="food_classification")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    batch_size = args.batch_size
    n_epochs = args.n_epochs
    patience = args.patience
    lr = args.lr
    weight_decay = args.weight_decay
    eps = args.eps
    exp_name = args.exp_name
    gamma = args.gamma
    alpha = args.alpha

    wandb.init(
        project="food_classification",
        name=exp_name,
        config={
            "batch_size": batch_size,
            "lr": lr,
            "n_epochs": n_epochs,
            "patience": patience,
            "weight_decay": weight_decay,
            "eps": eps,
        }
    )

    _dataset_dir = "Assignment2_dataset/"
    train_set = FoodDataset(os.path.join(_dataset_dir, "training dataset/train"), tfm=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_set = FoodDataset(os.path.join(_dataset_dir, "validation dataset/validation"), tfm=valid_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if torch.backends.mps.is_available(): device = "mps"
    elif torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    print(f"DEVICE: {device}")

    model = Classifier().to(device)
    optimizer = SAM(model.parameters(), base_optimizer, lr=lr, weight_decay=weight_decay)
    criterion = LabelSmoothingFocalLoss(gamma=gamma, alpha=alpha, smoothing=eps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    best_acc = 0
    _exp_name = exp_name

    # Training Loop
    for epoch in range(n_epochs):
        # ---------- Training ----------
        model.train()
        train_loss = []
        train_accs = []

        for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader)):
            imgs, labels = imgs.to(device), labels.to(device)

            # CutMix augmentation
            cutmix_prob = 0.5
            r = np.random.rand(1)
            if r < cutmix_prob:
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(imgs.size(0)).to(device)
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
                imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
                logits = model(imgs)
                loss = lam * criterion(logits, target_a) + (1 - lam) * criterion(logits, target_b)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)

            # SAM first step
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.first_step(zero_grad=True)

            # SAM second step
            logits2 = model(imgs)
            if r < cutmix_prob:
                loss2 = lam * criterion(logits2, target_a) + (1 - lam) * criterion(logits2, target_b)
            else:
                loss2 = criterion(logits2, labels)
            loss2.backward()
            optimizer.second_step(zero_grad=True)

            acc = (logits.argmax(dim=-1) == labels).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc.item())

        train_loss_avg = sum(train_loss) / len(train_loss)
        train_acc_avg = sum(train_accs) / len(train_accs)
        print(f"[ Train | {epoch+1:03d}/{n_epochs:03d} ] loss = {train_loss_avg:.5f}, acc = {train_acc_avg:.5f}")

        # ---------- Validation ----------
        model.eval()

        valid_losses = []
        valid_accs = []

        for batch in tqdm(valid_loader):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                logits = model(imgs)

            loss = criterion(logits, labels)
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            valid_losses.append(loss.item())
            valid_accs.append(acc.item())

        valid_loss_avg = sum(valid_losses) / len(valid_losses)
        valid_acc_avg = sum(valid_accs) / len(valid_accs)

        print(f"[ Valid | {epoch+1:03d}/{n_epochs:03d} ] loss = {valid_loss_avg:.5f}, acc = {valid_acc_avg:.5f}")

        with open(f"./{_exp_name}_log.txt", "a") as f:
            if valid_acc_avg > best_acc:
                print(f"[ Valid | {epoch+1:03d}/{n_epochs:03d} ] loss = {valid_loss_avg:.5f}, acc = {valid_acc_avg:.5f} -> best", file=f)
            else:
                print(f"[ Valid | {epoch+1:03d}/{n_epochs:03d} ] loss = {valid_loss_avg:.5f}, acc = {valid_acc_avg:.5f}", file=f)

        if valid_acc_avg > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")
            best_acc = valid_acc_avg

        scheduler.step()
        

        wandb.log({
            "train_loss": train_loss_avg,
            "train_acc": train_acc_avg,
            "valid_loss": valid_loss_avg,
            "valid_acc": valid_acc_avg,
            "epoch": epoch + 1,
            "lr": scheduler.get_last_lr()[0]
        })

    # ---------- Test ----------
    model_best = Classifier().to(device)
    model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
    model_best.eval()
    logit_pred = []
    file_ids = []

    with torch.no_grad():
        # TTA for test dataset
        for idx, tfm in enumerate(tta_transforms):
            tta_set = FoodDataset(os.path.join(_dataset_dir, "test dataset/test"), tfm=tfm, is_test=True)
            tta_loader = DataLoader(tta_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
            tta_logits = []
            tta_file_ids = []

            for data, _, file_id in tqdm(tta_loader, desc=f"TTA {idx+1}/{len(tta_transforms)}"):
                data = data.to(device)
                logits = model_best(data)
                tta_logits.append(logits.cpu().numpy())
                if idx == 0:
                    tta_file_ids += file_id

            logits_cat = np.concatenate(tta_logits, axis=0)
            logit_pred.append(logits_cat)
            if idx == 0:
                file_ids = tta_file_ids

        # ensemble
        ensemble_logits = np.mean(logit_pred, axis=0)
        prediction = np.argmax(ensemble_logits, axis=1)

    df = pd.DataFrame()
    df["ID"] = file_ids
    df["Category"] = prediction
    df.to_csv("last.csv", index=False)
    wandb.finish()