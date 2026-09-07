#!/usr/bin/env python3
"""LoRA SFT of a student on one teacher dataset (post = treatment, pre = control), one candidate at a time.

Student init = the candidate's own base model (the manifest's `student_init`), no adapter. The loss is
computed on completion tokens only; prompt tokens are masked with -100. Sequences are not packed (no
cross-example attention leakage without flash-attn varlen); per-device batch size 1 means no padding either.

Resumable: if --out-dir holds a checkpoint-* directory, training resumes from the latest one. Every
run appends to <out-dir>/train_log.jsonl and writes <out-dir>/train_summary.json at the end. The final
adapter is saved to <out-dir>/adapter (PEFT format, HF text-model key names).

Launch through code/sft.sbatch (torchrun, 1-8 GPUs). Direct use:
  python code/sft_train.py --cand cand3 --arm post --data-dir results/cand3/sft/datasets --out-dir results/cand3/sft/students/post
Pilot: add --max-steps 200 (and point --data-dir at a small dataset).
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

# The cuDNN SDPA backend failed twice on the same batch of one dataset ("mha_graph.execute ... is_good() to be true, but got
# false", stage 3, jobs 313415/313700, two different nodes). Use PyTorch's flash/efficient/math SDPA kernels instead.
torch.backends.cuda.enable_cudnn_sdp(False)

TEAM = Path(__file__).resolve().parents[1]

# Module sets that vLLM 0.28 already serves as LoRA for these architectures (team2 served exactly this
# set for the cand2 teacher adapter; the cand3 teacher adapter uses q,k,v,o). Staying inside these sets
# keeps the student adapter directly servable for the downstream evals.
DEFAULT_TARGETS = {
    "cand2": "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,out_proj",
    "cand3": "q_proj,k_proj,v_proj,o_proj",
}


def log(msg):
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


class SFTDataset(Dataset):
    def __init__(self, path, tok, max_seq_len, limit=None):
        self.items = []
        n_drop = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                c = tok(r["completion"], add_special_tokens=False)["input_ids"]
                if len(p) + len(c) > max_seq_len:
                    n_drop += 1
                    continue
                self.items.append({
                    "input_ids": p + c,
                    "labels": [-100] * len(p) + c,
                    "n_prompt": len(p), "n_completion": len(c),
                })
                if limit is not None and len(self.items) >= limit:
                    break
        self.n_dropped_too_long = n_drop
        self.lengths = [len(x["input_ids"]) for x in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = self.items[i]
        return {"input_ids": x["input_ids"], "labels": x["labels"]}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), m), self.pad_id, dtype=torch.long)
        lab = torch.full((len(feats), m), -100, dtype=torch.long)
        att = torch.zeros((len(feats), m), dtype=torch.long)
        for i, f in enumerate(feats):
            n = len(f["input_ids"])
            ids[i, :n] = torch.tensor(f["input_ids"])
            lab[i, :n] = torch.tensor(f["labels"])
            att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["cand2", "cand3"])
    ap.add_argument("--arm", required=True, choices=["post", "pre"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-path", default=None, help="default: manifest student_init.local_path")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--target-modules", default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="pilot: stop after this many optimizer steps")
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=12800)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--max-val-rows", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
    from peft import LoraConfig, get_peft_model

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(data_dir / f"{args.arm}.manifest.json"))
    base_path = args.base_path or manifest["student_init"]["local_path"]
    assert manifest["cand"] == args.cand and manifest["arm"] == args.arm, "manifest does not match --cand/--arm"
    log(f"cand={args.cand} arm={args.arm} student base={manifest['student_init']['base_hf_id']}@{manifest['student_init']['base_revision']}")
    log(f"base path={base_path}")
    log(f"world_size={os.environ.get('WORLD_SIZE', '1')} torch={torch.__version__} cuda={torch.cuda.is_available()}")
    try:
        import fla  # noqa: F401
        log(f"fla available: {fla.__version__} (fast gated-delta-rule kernels)")
    except Exception as e:
        log(f"fla NOT available ({type(e).__name__}); Qwen3.5 linear-attention layers use the torch fallback")

    tok = AutoTokenizer.from_pretrained(base_path)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    t0 = time.time()
    train_ds = SFTDataset(data_dir / f"{args.arm}.train.jsonl", tok, args.max_seq_len, args.max_train_rows)
    val_ds = SFTDataset(data_dir / f"{args.arm}.val.jsonl", tok, args.max_seq_len, args.max_val_rows)
    n_tok = sum(train_ds.lengths)
    n_ctok = sum(x["n_completion"] for x in train_ds.items)
    log(f"train rows={len(train_ds)} (dropped {train_ds.n_dropped_too_long} > {args.max_seq_len} tokens) "
        f"tokens={n_tok / 1e6:.2f}M completion tokens={n_ctok / 1e6:.2f}M max_len={max(train_ds.lengths)}; "
        f"val rows={len(val_ds)}; tokenized in {time.time() - t0:.0f}s")

    # Sanity: the tokenized prompt must equal the token ids generation sent (data/prompts_<cand>.jsonl).
    ref = {}
    with open(TEAM / "data" / f"prompts_{args.cand}.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 50:
                break
            r = json.loads(line)
            ref[r["problem_idx"]] = r["rendered_token_ids"]
    checked = 0
    if manifest.get("cross_base"):
        ref = {}  # cross-base datasets are rendered with the STUDENT's template; the teacher's prompt ids do not apply
        log("cross-base dataset: prompt token-id check against the teacher's prompt file skipped")
    with open(data_dir / f"{args.arm}.train.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["problem_idx"] in ref:
                ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
                assert ids == ref[r["problem_idx"]], f"prompt token ids differ from generation for problem {r['problem_idx']}"
                checked += 1
            if checked >= 10:
                break
    log(f"prompt token-id check: {checked} prompts identical to what generation sent")

    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16, attn_implementation=args.attn)
    model.config.use_cache = False
    targets = (args.target_modules or DEFAULT_TARGETS[args.cand]).split(",")
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=targets, task_type="CAUSAL_LM", bias="none")
    model = get_peft_model(model, lcfg)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"LoRA r={args.lora_r} alpha={args.lora_alpha} targets={targets} trainable={trainable / 1e6:.1f}M of {total / 1e9:.2f}B")

    world = int(os.environ.get("WORLD_SIZE", "1"))
    eff_batch = args.per_device_batch * args.grad_accum * world
    steps_per_epoch = math.ceil(len(train_ds) / eff_batch)
    planned_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    log(f"effective batch={eff_batch} sequences; steps/epoch={steps_per_epoch}; planned optimizer steps={planned_steps}")

    class JsonlLogger(TrainerCallback):
        def __init__(self, path):
            self.path = path
            self.t_start = time.time()

        def on_log(self, a, state, control, logs=None, **kw):
            if state.is_world_process_zero and logs:
                rec = {"step": state.global_step, "epoch": round(state.epoch or 0, 4),
                       "elapsed_s": round(time.time() - self.t_start, 1), **logs}
                with open(self.path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, math.ceil(planned_steps * args.warmup_ratio)),
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        eval_strategy="steps" if len(val_ds) else "no", eval_steps=args.eval_steps,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed, data_seed=args.seed,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=False,  # enabled manually above (PEFT-compatible kwargs)
        optim="adamw_torch",
        save_only_model=False,
        disable_tqdm=True,
        logging_first_step=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds if len(val_ds) else None,
                      data_collator=Collator(pad_id), callbacks=[JsonlLogger(out_dir / "train_log.jsonl")])

    ckpts = sorted(glob.glob(str(out_dir / "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    resume = ckpts[-1] if ckpts else None
    log(f"resume_from_checkpoint={resume}")
    t1 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    train_s = time.time() - t1

    # evaluate() is a collective (all ranks gather the eval loss): every rank must call it, or rank 0 hangs until
    # the NCCL watchdog kills the job (this happened on the first cand2 full run, jobs 312637/312640).
    eval_metrics = trainer.evaluate() if len(val_ds) else {}
    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(str(out_dir / "adapter"))
        tok.save_pretrained(str(out_dir / "adapter"))
        summary = {
            "cand": args.cand, "arm": args.arm, "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
            "student_init": manifest["student_init"], "data_dir": str(data_dir), "dataset_manifest": str(data_dir / f"{args.arm}.manifest.json"),
            "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "targets": targets, "trainable_params": trainable},
            "train": {"rows": len(train_ds), "tokens": n_tok, "completion_tokens": n_ctok, "dropped_too_long": train_ds.n_dropped_too_long,
                      "epochs": args.epochs, "max_steps": args.max_steps, "global_steps": trainer.state.global_step,
                      "effective_batch": eff_batch, "lr": args.lr, "world_size": world, "max_seq_len": args.max_seq_len,
                      "train_loss": result.training_loss, "wall_s_this_run": round(train_s, 1),
                      "tokens_per_s_this_run_approx": round(n_tok * (trainer.state.global_step / max(planned_steps, 1)) * (args.epochs if args.max_steps <= 0 else 1) / max(train_s, 1)),
                      "gpu_hours_this_run": round(train_s * world / 3600, 2)},
            "eval": eval_metrics,
            "adapter_dir": str(out_dir / "adapter"),
            "resumed_from": resume,
        }
        with open(out_dir / "train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        log(f"DONE steps={trainer.state.global_step} train_loss={result.training_loss:.4f} eval={eval_metrics} "
            f"wall={train_s / 60:.1f}min gpu_h={summary['train']['gpu_hours_this_run']}")


if __name__ == "__main__":
    main()
