import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_one_epoch(model, loader, optimizer, criterion, device, A_sets):

    model.train()
    total_loss = 0.0

    for X, Y in tqdm(loader, desc="Train", leave=False):
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        Y_hat = model(X, A_sets)
        loss = criterion(Y_hat, Y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader, criterion, device, A_sets):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, Y in tqdm(loader, desc="Eval", leave=False):
            X = X.to(device)
            Y = Y.to(device)

            Y_hat = model(X, A_sets)
            loss = criterion(Y_hat, Y)
            total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


class EarlyStopping:
  
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state = None
        self.best_epoch = -1

    def step(self, val_loss, model, epoch):
     
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            self.counter = 0
            print(f"[EarlyStop] New best val loss: {val_loss:.6f} at epoch {epoch}")
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def train_model(
    model,
    train_dataset,
    val_dataset,
    A_sets,
    epochs=100,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    patience=10,
    device="cuda",
    save_dir="./checkpoints",
    plot_loss=True
):
    print(f"A_sets type: {type(A_sets)}")
    print(f"A_sets length: {len(A_sets)}")
    for i, A in enumerate(A_sets):
        print(f"A[{i}] shape: {A.shape}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    A_sets = [torch.tensor(A, dtype=torch.float32).to(device) for A in A_sets]



    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    criterion = nn.MSELoss()
    early_stopper = EarlyStopping(patience=patience)

    train_losses, val_losses = [], []

    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, A_sets
        )
        val_loss = eval_one_epoch(
            model, val_loader, criterion, device, A_sets
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if early_stopper.step(val_loss, model, epoch):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    if early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)
        torch.save(early_stopper.best_state, best_ckpt_path)
        print(
            f"Best model loaded (epoch {early_stopper.best_epoch}, "
            f"val loss {early_stopper.best_loss:.6f})"
        )

    if plot_loss:
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("STGCN Training Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return model, train_losses, val_losses
