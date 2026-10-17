"""Compatibility helpers for earlier package versions.

New experiments should use decoder_llm.experiment and the decoder-llm CLI.
The original train_epoch/train/plot_losses entry points remain available.
"""
from typing import Dict, List
import sys
import torch
from torch.utils.data import DataLoader
from decoder_llm.model import DecoderOnlyTransformer, LLMConfig, build_model, compute_loss
from decoder_llm.tokenizer import CharTokenizer
from decoder_llm.data import CharacterDataset, TranslationLineDataset, parse_chat_pairs
from decoder_llm.generation import generate
from decoder_llm.experiment import set_seed
from decoder_llm.checkpoint import save_checkpoint

def train_epoch(
    model: DecoderOnlyTransformer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_batches: int,
) -> float:
    """Train for one epoch (but capped to num_batches for speed/consistency)."""
    model.train()
    total_loss = 0.0
    processed = 0

    for step, batch in enumerate(train_loader):
        if step >= num_batches:
            break

        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)["logits"]
        loss = compute_loss(logits, target_ids)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        processed += 1

    if not processed:
        raise ValueError("loader yielded no batches")
    return total_loss / processed


@torch.no_grad()
def evaluate(
    model: DecoderOnlyTransformer,
    val_loader: DataLoader,
    device: str,
    num_batches: int,
) -> float:
    """Validation loss (no gradients)."""
    model.eval()
    total_loss = 0.0
    processed = 0

    for step, batch in enumerate(val_loader):
        if step >= num_batches:
            break

        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        logits = model(input_ids)["logits"]
        loss = compute_loss(logits, target_ids)
        total_loss += loss.item()
        processed += 1

    if not processed:
        raise ValueError("loader yielded no batches")
    return total_loss / processed



def train(
    model: DecoderOnlyTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: str,
    tokenizer: CharTokenizer = None,
    batches_per_epoch: int = 100,
    val_batches: int = 20,
    output_path: str = "model.pt",
) -> Dict[str, List[float]]:
    """Full training loop with validation + checkpointing."""
    model.to(device)
    # weight_decay helps prevent memorization (especially on char-level models)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)


    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, num_epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, device, batches_per_epoch)
        #va_loss = eval_loss(model, val_loader, device, val_batches)
        va_loss = evaluate(model, val_loader, device, val_batches)

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        print(f"Epoch {epoch:02d}/{num_epochs} | train loss: {tr_loss:.4f} | val loss: {va_loss:.4f}")

        # Save checkpoint each epoch (so you don't lose progress)
        if tokenizer is None:
            raise ValueError("tokenizer is required to save a usable checkpoint")
        save_checkpoint(output_path, model, tokenizer)


    return {"train_losses": train_losses, "val_losses": val_losses}



def plot_losses(train_losses: List[float], val_losses: List[float], save_path: str = "losses.png"):
    """Plot training and validation losses."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Losses", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Loss plot saved to {save_path}")



def main():
    from decoder_llm.cli import main as cli_main
    cli_main(["train", *sys.argv[1:]])


if __name__ == "__main__":
    main()
