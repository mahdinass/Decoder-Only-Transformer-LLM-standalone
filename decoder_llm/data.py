"""Raw-text windows and target-masked prompt/response batches."""
import torch
from torch.utils.data import IterableDataset, get_worker_info
from .tokenizer import CharTokenizer


def parse_chat_pairs(text: str) -> list[tuple[str, str]]:
    """Parse the historical <HUMAN>: / <ASSISTANT>: line format."""
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("<HUMAN>:") and "<ASSISTANT>:" in line:
            prompt, target = line[len("<HUMAN>:"):].split("<ASSISTANT>:", 1)
            if prompt.strip() and target.strip():
                pairs.append((prompt.strip(), target.strip()))
    return pairs


class CharacterDataset(IterableDataset):
    """Sample windows uniformly, including the final valid window.

    Batches are already assembled: use DataLoader(batch_size=None, num_workers=0).
    A private generator enables repeatable experiments without resetting global RNG.
    """
    def __init__(self, text, tokenizer: CharTokenizer, context_length,
                 batch_size=1, num_samples=50_000, *, generator=None):
        super().__init__()
        if min(context_length, batch_size, num_samples) < 1:
            raise ValueError("context_length, batch_size and num_samples must be positive")
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.generator = generator
        self.encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.num_tokens = len(self.encoded)
        if self.num_tokens < context_length + 1:
            raise ValueError("text needs at least context_length + 1 characters")

    def __iter__(self):
        if get_worker_info() is not None:
            raise ValueError("prebatched random datasets require num_workers=0")
        for start in range(0, self.num_samples, self.batch_size):
            size = min(self.batch_size, self.num_samples - start)
            offsets = torch.randint(self.num_tokens - self.context_length, (size, 1),
                                    generator=self.generator)
            indices = offsets + torch.arange(self.context_length)
            yield {"input_ids": self.encoded[indices], "target_ids": self.encoded[indices + 1]}


class TranslationLineDataset(IterableDataset):
    """Preserve prompt -> newline -> target training with no all-masked rows.

    Long prompts are left-truncated to reserve a separator and a target character.
    Padding and prompt predictions do not contribute to the loss.
    """
    def __init__(self, pairs, tokenizer, context_length, batch_size, num_samples,
                 *, generator=None):
        super().__init__()
        if not pairs or min(context_length, batch_size, num_samples) < 1:
            raise ValueError("pairs and positive dataset dimensions are required")
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.generator = generator
        self.sep_id = tokenizer.encode("\n")[0]
        self.encoded = [(tokenizer.encode(p), tokenizer.encode(t)) for p, t in pairs]
        if any(not target for _, target in self.encoded):
            raise ValueError("each pair must have a nonempty target")

    def __iter__(self):
        if get_worker_info() is not None:
            raise ValueError("prebatched random datasets require num_workers=0")
        T = self.context_length
        for start in range(0, self.num_samples, self.batch_size):
            size = min(self.batch_size, self.num_samples - start)
            x = torch.zeros(size, T, dtype=torch.long)
            y = torch.full_like(x, -100)
            choices = torch.randint(len(self.encoded), (size,), generator=self.generator)
            for row, choice in enumerate(choices.tolist()):
                prompt, target = self.encoded[choice]
                prompt = prompt[-(T - 1):] if T > 1 else []
                ids = prompt + [self.sep_id] + target[:T - len(prompt)]
                length = len(ids) - 1
                x[row, :length] = torch.tensor(ids[:-1])
                y[row, :length] = torch.tensor(ids[1:])
                y[row, :len(prompt)] = -100
            yield {"input_ids": x, "target_ids": y}
