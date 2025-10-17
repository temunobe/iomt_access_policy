# Quick test for ModelTrainer._safe_from_pretrained
# This script patches transformers.AutoModelForCausalLM.from_pretrained to simulate
# a ValueError on first call when attn_implementation is present, and succeed on retry.

import types
from model_trainer import ModelTrainer
import transformers

original = transformers.AutoModelForCausalLM.from_pretrained
orig_tokenizer = transformers.AutoTokenizer.from_pretrained

call_state = {"count": 0}

def fake_from_pretrained(*args, **kwargs):
    call_state["count"] += 1
    # If attn_implementation present, simulate the ValueError that some model classes raise
    if kwargs.get("attn_implementation"):
        raise ValueError("Llama4ForCausalLM does not support Flash Attention 2.0 yet.")
    return "MOCK_MODEL"

try:
    transformers.AutoModelForCausalLM.from_pretrained = fake_from_pretrained
    # stub tokenizer loader so ModelTrainer init doesn't hit the network
    transformers.AutoTokenizer.from_pretrained = lambda *a, **k: types.SimpleNamespace(pad_token=None, eos_token="<eos>")

    mt = ModelTrainer(model_name="dummy-model")
    # Call with positional model name and attn_implementation kwarg to match previous usage
    result = mt._safe_from_pretrained("dummy-model", attn_implementation="flash_attention_2")
    print("_safe_from_pretrained returned:", result)
    assert result == "MOCK_MODEL"
    assert call_state["count"] == 2, f"expected 2 calls (initial + retry), got {call_state['count']}"
    print("TEST PASSED: fallback behavior works as expected")
finally:
    transformers.AutoModelForCausalLM.from_pretrained = original
    transformers.AutoTokenizer.from_pretrained = orig_tokenizer
