"""The model: a Qwen3.5 backbone with its LM head removed, LoRA adapters, and a pointer head that scores each option.

Input layout for one review (one question, so no cross-question mask is needed):

    <state> review text <q> instruction <opt> option A </opt> <opt> option B </opt> <decide>

The pointer head reads two kinds of hidden states: the one at each </opt> ("what this option is, given the review") and
the one at <decide> ("what answer this question is looking for"), and scores each option by a scaled dot product.
"""
import re

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import BASE, BASE_REVISION, LABELS

# Rarely used tokens already in Qwen's vocabulary, borrowed as delimiters: no new embedding rows to train.
STATE, QUESTION, OPT, OPT_END, DECIDE = "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"

INSTRUCTION = "这条评论的情感倾向是？"     # the corpus mixes hotel, laptop and book reviews
OPTIONS = {"negative": "负面：不满意、抱怨、批评", "positive": "正面：满意、称赞、推荐"}
MAX_TEXT_TOKENS = 384      # longer reviews keep their first and last halves (the verdict is often at the end)
POINTER_DIM = 256
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",                              # attention layers
                "gate_proj", "up_proj", "down_proj",                                   # MLPs
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]      # Gated DeltaNet layers

_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def load_tokenizer():
    return AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION)


def user_ids(tok, text):
    """Token ids for caller-supplied text that can never contain a delimiter: <|name|> is rewritten to <¦name¦> first."""
    return tok(_SPECIAL.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode(tok, text, order=LABELS, max_text=MAX_TEXT_TOKENS):
    """One review -> {"ids", "ends" (position of each option's </opt>, in `order`), "decide" (position of <decide>)}."""
    special = {t: tok.convert_tokens_to_ids(t) for t in (STATE, QUESTION, OPT, OPT_END, DECIDE)}
    body = user_ids(tok, text)
    if len(body) > max_text:
        body = body[: max_text // 2] + body[-(max_text - max_text // 2):]
    ids = [special[STATE]] + body + [special[QUESTION]] + user_ids(tok, INSTRUCTION)
    ends = []
    for name in order:
        ids += [special[OPT]] + user_ids(tok, OPTIONS[name]) + [special[OPT_END]]
        ends.append(len(ids) - 1)
    ids.append(special[DECIDE])
    return {"ids": ids, "ends": ends, "decide": len(ids) - 1}


def collate(encs, pad_id, device):
    """Right-pad a list of encodings. Pads sit after every real token, so in a left-to-right model they can never
    change a real token's hidden state (true for the DeltaNet layers as well as the attention layers)."""
    L = max(len(e["ids"]) for e in encs)
    ids = torch.full((len(encs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(encs), L), dtype=torch.long)
    for i, e in enumerate(encs):
        ids[i, : len(e["ids"])] = torch.tensor(e["ids"]); mask[i, : len(e["ids"])] = 1
    ends = torch.tensor([e["ends"] for e in encs])
    decide = torch.tensor([e["decide"] for e in encs])
    return ids.to(device), mask.to(device), ends.to(device), decide.to(device)


class PointerHead(nn.Module):
    """score_j = (W_q h_decide) . (W_k h_opt_j) / sqrt(dp). Two projections, because the two positions play different
    roles (query vs key); dp = 256 keeps it small. `temperature` divides the scores at inference only."""

    def __init__(self, d, dp=POINTER_DIM, zero_init=False):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = dp ** -0.5
        self.temperature = 1.0
        if zero_init:
            # W_k = 0 (and its bias): every option starts with score 0, i.e. a 50/50 guess and loss ln 2, the way LoRA's
            # B starts at 0. The backbone's hidden states are large, so a random W_k starts out wildly overconfident.
            nn.init.zeros_(self.k.weight); nn.init.zeros_(self.k.bias)

    def forward(self, h_decide, h_opts):                     # [B, d], [B, K, d] -> logits [B, K]
        z = torch.einsum("bd,bkd->bk", self.q(h_decide), self.k(h_opts)) * self.scale
        return z if self.training else z / self.temperature


class SentimentModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, ids, mask, ends, decide):
        h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state       # [B, L, d]
        rows = torch.arange(ids.shape[0], device=ids.device)
        h_decide = h[rows, decide].float()                                               # head runs in fp32
        h_opts = h[rows[:, None], ends].float()                                          # [B, K, d]
        return self.head(h_decide, h_opts)


def load_backbone(dtype=torch.bfloat16):
    """The base model without its LM head: AutoModelForCausalLM(...).model is the Transformer body."""
    lm = AutoModelForCausalLM.from_pretrained(BASE, revision=BASE_REVISION, dtype=dtype, attn_implementation="sdpa")
    return lm.model


def add_lora(backbone, r=16, alpha=32, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(task_type="FEATURE_EXTRACTION", r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS)
    return get_peft_model(backbone, cfg)


def load_trained(run, device="cuda"):
    """A finished run: the base backbone, its LoRA merged in, and the pointer head with its fitted temperature."""
    from peft import PeftModel
    ckpt = torch.load(f"{run}/head.pt", map_location="cpu")
    backbone = PeftModel.from_pretrained(load_backbone(), run).merge_and_unload()
    head = PointerHead(backbone.config.hidden_size, ckpt["pointer_dim"])
    head.load_state_dict(ckpt["head"]); head.temperature = ckpt.get("temperature", 1.0)
    model = SentimentModel(backbone, head).to(device).eval()
    return model, ckpt
