"""slmtriage.local_hf — zero-shot inference with small open-weight models (Transformers), checkpointed and resumable.

Two decision rules on the same model and prompt:
  mode G  greedy generation (max 48 new tokens) + strict first-mention parser (and the reference parser, for comparison)
  mode S  label scoring: length-normalised log p(label | prompt [+ prefill]) over the three verbalizers, argmax.
          P1 needs the assistant prefill "intention: " (pilot amendment: without it the label mass is ~0).
Every run writes results/jsonl/{dataset}__{model}__{prompt}__{mode}__bf16.jsonl (one line per item) and a manifest with
the HF revision, git commit, GPU and timing. Partial progress goes to a *.partial.jsonl (gitignored) and resumes.
"""
from __future__ import annotations

import gc
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import core

PREFILL = {"P1": "intention: ", "P2": "", "P3": ""}
MAX_NEW = 48
REGISTRY = {
    "qwen3-0.6b": dict(hf="Qwen/Qwen3-0.6B", params_b=0.6, gated=False),
    "qwen3-1.7b": dict(hf="Qwen/Qwen3-1.7B", params_b=1.7, gated=False),
    "qwen3-4b": dict(hf="Qwen/Qwen3-4B", params_b=4.0, gated=False),
    "llama3.2-1b": dict(hf="meta-llama/Llama-3.2-1B-Instruct", params_b=1.0, gated=True),
    "llama3.2-3b": dict(hf="meta-llama/Llama-3.2-3B-Instruct", params_b=3.0, gated=True),
    "gemma3-1b": dict(hf="google/gemma-3-1b-it", params_b=1.0, gated=True),
    "gemma3-4b": dict(hf="google/gemma-3-4b-it", params_b=4.0, gated=True),
    "phi4-mini": dict(hf="microsoft/Phi-4-mini-instruct", params_b=3.8, gated=False),
}


def _git_commit(repo) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def hf_revision(key: str) -> str:
    try:
        from huggingface_hub import model_info

        return model_info(REGISTRY[key]["hf"]).sha
    except Exception as e:  # offline or gated
        return f"unavailable: {type(e).__name__}"


def load_local(key: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf = REGISTRY[key]["hf"]
    tok = AutoTokenizer.from_pretrained(hf)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16, device_map="cuda").eval()
    return tok, model, hf_revision(key)


def free(*objs) -> None:
    import torch

    for o in objs:
        del o
    gc.collect(); torch.cuda.empty_cache()


def render(tok, prompts: dict, prompt_id: str, text: str, prefill: str = "") -> str:
    p = prompts[prompt_id]
    msgs = [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"].format(text=text)}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:  # templates without a system role: fold system into the user turn
        msgs = [{"role": "user", "content": p["system"] + "\n\n" + p["user"].format(text=text)}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return s + prefill


def generate_batch(tok, model, rendered: list[str], max_new: int = MAX_NEW):
    import torch

    tok.padding_side = "left"
    enc = tok(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    ms = (time.perf_counter() - t0) * 1000 / len(rendered)
    gen = out[:, enc["input_ids"].shape[1]:]
    return [tok.decode(g, skip_special_tokens=True) for g in gen], ms, enc["attention_mask"].sum(1).tolist()


def _lm_parts(model):
    base = getattr(model, "model", None) or model.base_model
    head = getattr(model, "lm_head", None) or model.get_output_embeddings()
    return base, head


def score_batch(tok, model, rendered: list[str]):
    """Returns (means[B,3], ms, sums[B,3]) of log p(label tokens | prompt). Left-padded; LM head only on the last positions."""
    import torch

    seqs, clens = [], []
    for r in rendered:
        p_ids = tok(r, add_special_tokens=False)["input_ids"]
        for c in core.CLASSES:
            f_ids = tok(r + c, add_special_tokens=False)["input_ids"]
            k = len(p_ids) if f_ids[: len(p_ids)] == p_ids else next(i for i, (a, b) in enumerate(zip(f_ids, p_ids)) if a != b)
            seqs.append(f_ids); clens.append(len(f_ids) - k)
    L, keep = max(map(len, seqs)), max(clens) + 1
    ids = torch.full((len(seqs), L), tok.pad_token_id); att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s); att[i, L - len(s):] = 1
    pos = (att.cumsum(1) - 1).clamp(min=0)
    base, head = _lm_parts(model)
    t0 = time.perf_counter()
    with torch.no_grad():
        h = base(input_ids=ids.cuda(), attention_mask=att.cuda(), position_ids=pos.cuda()).last_hidden_state[:, -keep:]
        lp_all = torch.log_softmax(head(h).float(), -1)
        lp = lp_all[:, :-1].gather(-1, ids[:, -keep + 1:].cuda().unsqueeze(-1)).squeeze(-1).cpu()
    ms = (time.perf_counter() - t0) * 1000 / len(rendered)
    sums = np.array([float(lp[i, -clens[i]:].sum()) for i in range(len(seqs))]); means = sums / np.array(clens)
    return means.reshape(-1, len(core.CLASSES)), ms, sums.reshape(-1, len(core.CLASSES))


def run_local(key: str, dataset: str, prompt_id: str, mode: str, proc_dir: Path, out_dir: Path, prompts: dict,
              repo: Path | None = None, batch: int | None = None, limit: int | None = None, tok=None, model=None,
              rev=None, tag: str = "", verbose: bool = True) -> Path:
    """One (dataset, model, prompt, mode) cell of the matrix. Idempotent; resumes from the partial file."""
    batch = batch or (64 if mode == "G" else 24)
    df = pd.read_csv(Path(proc_dir) / f"{dataset}_eval.csv"); df = df.head(limit) if limit else df
    run_id = f"{dataset}__{key}__{prompt_id}__{mode}__bf16{tag}"
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path, part = out_dir / f"{run_id}.jsonl", out_dir / f"{run_id}.partial.jsonl"
    if path.exists() and sum(1 for _ in open(path)) == len(df):
        if verbose:
            print("cached:", run_id)
        return path
    if model is None:
        tok, model, rev = load_local(key)
    done = {json.loads(l)["id"] for l in open(part)} if part.exists() else set()
    todo = [i for i in range(len(df)) if df.id[i] not in done]
    if done and verbose:
        print(f"  resuming {run_id}: {len(done)} done, {len(todo)} to go")
    pre = PREFILL[prompt_id] if mode == "S" else ""
    t_start = time.time()
    with open(part, "a") as f:
        for s in range(0, len(todo), batch):
            b = todo[s:s + batch]; rendered = [render(tok, prompts, prompt_id, str(df.text[i]), pre) for i in b]
            if mode == "G":
                raws, ms, ntok = generate_batch(tok, model, rendered)
                rows = []
                for i, raw, n in zip(b, raws, ntok):
                    lab, ok, multi = core.parse_strict(raw)
                    rows.append(dict(id=df.id[i], raw=raw, label=lab, valid=ok, multi=multi, label_ref=core.parse_ref(raw),
                                     prompt_tokens=n, ms_per_review=ms))
            else:
                means, ms, sums = score_batch(tok, model, rendered)
                rows = [dict(id=df.id[i], scores=dict(zip(core.CLASSES, mr.tolist())), scores_sum=dict(zip(core.CLASSES, sr.tolist())),
                             label=core.CLASSES[int(np.argmax(mr))], valid=True, label_mass=float(np.exp(sr).sum()), ms_per_review=ms)
                        for i, mr, sr in zip(b, means, sums)]
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            if verbose and (s // batch) % 10 == 0:
                print(f"  {run_id}: {len(done) + min(s + batch, len(todo))}/{len(df)}  {time.time() - t_start:.0f}s")
    order = {i: k for k, i in enumerate(df.id)}
    rows = sorted((json.loads(l) for l in open(part)), key=lambda r: order[r["id"]])
    assert len(rows) == len(df), f"{len(rows)} != {len(df)}"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    part.unlink()
    import torch

    (out_dir / f"{run_id}.manifest.json").write_text(json.dumps(dict(
        run_id=run_id, dataset=dataset, model=key, hf=REGISTRY[key]["hf"], revision=rev, prompt=prompt_id, mode=mode, prefill=pre,
        precision="bf16", batch=batch, n=len(df), max_new_tokens=MAX_NEW if mode == "G" else None, decoding="greedy",
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__,
        git_commit=_git_commit(repo) if repo else "unknown", wall_s_last_session=round(time.time() - t_start, 1),
        finished_utc=core_now()), indent=2))
    if verbose:
        print(f"wrote {run_id}  n={len(rows)}  {time.time() - t_start:.0f}s")
    return path


def core_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
