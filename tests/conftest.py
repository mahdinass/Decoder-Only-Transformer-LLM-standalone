import pytest
import torch
from decoder_llm import LLMConfig, build_model


@pytest.fixture(autouse=True)
def runtime():
    torch.set_num_threads(1)
    torch.manual_seed(17)


@pytest.fixture
def model():
    return build_model(LLMConfig(vocab_size=7, context_length=8, d_model=16,
                                num_layers=2, num_heads=4, d_ff=32, dropout=0.0))
