"""Reusable optimization helpers retained from the original project."""
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torch import optim
from .model import DecoderOnlyTransformer, compute_loss

def pretrain(
    model: DecoderOnlyTransformer,
    train_loader: DataLoader,
    num_steps: int = 1000,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    log_interval: int = 100,
) -> Dict[str, list]:
    """
    Pre-train the language model.

    Args:
        model: DecoderOnlyTransformer instance
        train_loader: DataLoader for training data
        num_steps: Number of training steps
        learning_rate: Learning rate for optimizer
        device: Device to train on
        log_interval: Logging frequency

    Returns:
        Dictionary with training metrics
    """
    if num_steps < 1 or log_interval < 1 or learning_rate <= 0:
        raise ValueError("steps, log_interval and learning_rate must be positive")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

    model.train()
    losses = []

    iterator = iter(train_loader)
    for step in range(num_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise ValueError("training loader yielded no batches") from exc

        # Support both (x, y) batches and dict batches
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            input_ids, target_ids = batch
        else:
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)


        # Forward pass
        output = model(input_ids)
        logits = output["logits"]

        # Compute loss
        loss = compute_loss(logits, target_ids)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if (step + 1) % log_interval == 0:
            avg_loss = sum(losses[-log_interval:]) / log_interval
            print(f"Step {step+1}/{num_steps} | Loss: {avg_loss:.4f}")

    return {"losses": losses}
