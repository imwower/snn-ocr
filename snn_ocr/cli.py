"""Command line entry point for SNN-OCR (synth/train/eval/preview)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from snn_ocr import pgm, render, synth
from snn_ocr.eval import evaluate, dump_examples, ensure_dataset
from snn_ocr.train import TrainArgs, train_stage
from snn_ocr.utils import Timer, ensure_dir


def cmd_synth(ns: argparse.Namespace) -> None:
    stage = ns.stage.upper()
    out_dir = Path(ns.out)
    ensure_dir(out_dir)
    with Timer(f"Generating {ns.n} samples for {stage}"):
        synth.generate_dataset(stage, ns.n, out_dir)
    print(json.dumps({"stage": stage, "samples": ns.n, "out": str(out_dir)}))


def cmd_train(ns: argparse.Namespace) -> None:
    stage = ns.stage.upper()
    out_dir = Path(ns.out) if ns.out else Path("runs") / stage.lower()
    ensure_dir(out_dir)
    train_args = TrainArgs(
        stage=stage,
        out_dir=out_dir,
        epochs=ns.epochs,
        steps_per_epoch=ns.steps,
        batch_size=ns.batch,
        optimizer=ns.optimizer,
        lr=ns.lr,
        min_lr=ns.min_lr,
        cosine_anneal=ns.cosine,
        warmup_steps=ns.warmup_steps,
        clip_grad=ns.clip,
        clip_mode=ns.clip_mode,
        replay_override=ns.replay,
        resume=ns.resume,
        log_every=ns.log_every,
        save_every=ns.save_every,
        seed=ns.seed,
        distill=ns.distill,
        distill_lambda=ns.distill_lambda,
        teacher_ckpt=Path(ns.teacher) if ns.teacher else None,
        dev_steps=ns.dev_steps,
        dev_batch=ns.dev_batch,
        dev_every=ns.dev_every,
        early_stop_patience=ns.early_stop_patience,
        early_stop_metric=ns.early_stop_metric,
    )
    result = train_stage(train_args)
    print(
        json.dumps(
            {
                "stage": result.stage,
                "steps": result.steps,
                "initial_loss": result.initial_loss,
                "final_loss": result.final_loss,
                "out_dir": str(out_dir),
            }
        )
    )


def cmd_eval(ns: argparse.Namespace) -> None:
    stage = ns.stage.upper()
    data_dir = Path(ns.data) if ns.data else Path("runs") / f"eval_{stage.lower()}"
    ensure_dataset(stage, data_dir, size=ns.limit or 40)
    metrics, examples = evaluate(
        stage,
        data_dir,
        checkpoint=Path(ns.ckpt) if ns.ckpt else None,
        limit=ns.limit,
        sample_count=ns.examples,
    )
    vis_root = Path(ns.vis)
    ensure_dir(vis_root)
    vis_dir = vis_root / stage.lower()
    dump_examples(examples, vis_dir)
    print(
        json.dumps(
            {
                "stage": stage,
                "top1": metrics.top1,
                "cer": metrics.cer,
                "wer": metrics.wer,
                "fire_rate": metrics.avg_fire_rate,
                "width_out": metrics.avg_width,
                "samples": metrics.samples,
                "energy_total": metrics.energy_total,
                "energy_per_pixel": metrics.energy_per_pixel,
                "duty_cycle": metrics.duty_cycle,
                "vis_dir": str(vis_dir),
            }
        )
    )


def cmd_preview(ns: argparse.Namespace) -> None:
    text = ns.text.replace("\\n", "\n")
    width = ns.width
    height = ns.height
    ascii_out = ns.ascii
    out_path = Path(ns.out) if ns.out else None
    grid = render.render_text(text, width, height, jitter=ns.jitter, noise=ns.noise, contrast=ns.contrast)
    if out_path:
        ensure_dir(out_path.parent)
        pgm.save_pgm(out_path, grid)
        print(f"Saved preview to {out_path}")
    if ascii_out:
        print(render.ascii_preview(grid))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="snn_ocr.cli",
        description="SNN-OCR command line interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_synth = subparsers.add_parser("synth", help="Generate curriculum samples.")
    parser_synth.add_argument("--stage", type=str, default="S1")
    parser_synth.add_argument("--n", type=int, required=True)
    parser_synth.add_argument("--out", type=str, required=True)
    parser_synth.set_defaults(func=cmd_synth)

    parser_train = subparsers.add_parser("train", help="Train a stage model.")
    parser_train.add_argument("--stage", type=str, default="S1")
    parser_train.add_argument("--out", type=str, default=None)
    parser_train.add_argument("--epochs", type=int, default=1)
    parser_train.add_argument("--steps", type=int, default=100)
    parser_train.add_argument("--batch", type=int, default=16)
    parser_train.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser_train.add_argument("--lr", type=float, default=0.01)
    parser_train.add_argument("--min-lr", type=float, default=0.001)
    parser_train.add_argument("--warmup-steps", type=int, default=0)
    parser_train.add_argument("--cosine", action="store_true")
    parser_train.add_argument("--clip", type=float, default=1.0)
    parser_train.add_argument("--clip-mode", choices=["norm", "value"], default="norm")
    parser_train.add_argument("--replay", type=float, default=None)
    parser_train.add_argument("--resume", action="store_true")
    parser_train.add_argument("--log-every", type=int, default=10)
    parser_train.add_argument("--save-every", type=int, default=200)
    parser_train.add_argument("--dev-steps", type=int, default=0, help="Number of evaluation batches for dev metrics.")
    parser_train.add_argument("--dev-batch", type=int, default=8, help="Batch size for dev evaluation.")
    parser_train.add_argument("--dev-every", type=int, default=1, help="Evaluate dev metrics every N epochs when >0.")
    parser_train.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Enable early stopping after N epochs without improvement (0 disables).",
    )
    parser_train.add_argument(
        "--early-stop-metric",
        choices=["cer", "top1"],
        default="cer",
        help="Metric to monitor when early stopping is enabled.",
    )
    parser_train.add_argument("--seed", type=int, default=7)
    parser_train.add_argument("--distill", action="store_true", help="Enable knowledge distillation (LwF).")
    parser_train.add_argument(
        "--distill-lambda",
        type=float,
        default=0.3,
        help="Weight for the KD term when --distill is enabled.",
    )
    parser_train.add_argument(
        "--teacher",
        type=str,
        default=None,
        help="Optional explicit checkpoint path to use as the distillation teacher.",
    )
    parser_train.set_defaults(func=cmd_train)

    parser_eval = subparsers.add_parser("eval", help="Evaluate a checkpoint.")
    parser_eval.add_argument("--stage", type=str, default="S3")
    parser_eval.add_argument("--data", type=str, default=None)
    parser_eval.add_argument("--ckpt", type=str, default=None)
    parser_eval.add_argument("--limit", type=int, default=None)
    parser_eval.add_argument("--examples", type=int, default=5)
    parser_eval.add_argument("--vis", type=str, default=str(Path("runs") / "vis"))
    parser_eval.set_defaults(func=cmd_eval)

    parser_preview = subparsers.add_parser("preview", help="Render text to a PGM and/or ASCII.")
    parser_preview.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to render (use \\n for explicit newlines)",
    )
    parser_preview.add_argument("--width", type=int, default=128)
    parser_preview.add_argument("--height", type=int, default=32)
    parser_preview.add_argument("--jitter", type=int, default=1)
    parser_preview.add_argument("--noise", type=float, default=0.01)
    parser_preview.add_argument("--contrast", type=float, default=1.0)
    parser_preview.add_argument("--out", type=str, default=None)
    parser_preview.add_argument("--ascii", action="store_true")
    parser_preview.set_defaults(func=cmd_preview)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(argv)
    ns.func(ns)


if __name__ == "__main__":
    main()
