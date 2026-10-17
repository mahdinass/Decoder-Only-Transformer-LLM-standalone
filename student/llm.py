"""Backward-compatible imports from the original student.llm module.

Architecture and parameter names are preserved in decoder_llm.model.
"""
from decoder_llm.model import (LLMConfig, RMSNorm, CausalSelfAttention, FeedForward,
                               TransformerBlock, DecoderOnlyTransformer, build_model, compute_loss)
from decoder_llm.tokenizer import CharTokenizer
from decoder_llm.data import CharacterDataset
from decoder_llm.training import pretrain
from decoder_llm.generation import generate
