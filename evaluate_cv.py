"""
evaluate_cv.py -- 5-fold cross-validation over the whole dataset. Every image is
tested exactly once, so tail diagnoses (n=1 in the full set) each get one fair
test and per-tier Finding-F1 becomes meaningful.

    python evaluate_cv.py all            # sweep monolithic -> rag -> full
    python evaluate_cv.py monolithic     # one ablation row
    python evaluate_cv.py fold rag 3     # (internal) run a single fold

For each fold: write a fold-specific manifest -> rebuild the FAISS index on that
fold's TRAIN split (leakage guard) -> retrain the LDAM-DRW classifier AND (when
DRAFT_MODE='external') the report generator on that fold's TRAIN split, saving
fold-specific checkpoints (leakage guard: no checkpoint ever sees the images it
is later scored on, and every term is trained wherever its examples land) -> run
the graph on the fold's TEST split -> pool. Then report aggregate + per-tier
metrics with support counts.

MEMORY: each fold runs in its own subprocess (see sweep()), so the OS reclaims
all of a fold's ViT/generator/BiomedCLIP memory when the fold exits, instead of
accumulating across 15 folds until the host runs out of application memory. This
also makes the sweep resumable: a fold whose prediction CSV (and, for external
drafting, its generator checkpoint) already exists is skipped.

config.DEVICE picks up MPS/CUDA automatically if available. Set
CLF_EPOCHS_PER_FOLD / GEN_EPOCHS_PER_FOLD to trade training time vs numbers.
"""
import os, sys, json, subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config
from vocab import TERMS, TIER, SYNONYMS
import build_index, run_phase_a, train_finding_agent, train_generator
from evaluate import finding_f1, calibration, _asserted

import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

CONFIG_NAME = "monolithic"       # default single-row run
ABLATION_ROWS = ["monolithic", "rag", "full"]  # rows run by the "all" sweep
N_FOLDS = 5
CLF_EPOCHS_PER_FOLD = int(os.environ.get("CLF_EPOCHS", 20))
GEN_EPOCHS_PER_FOLD = int(os.environ.get("GEN_EPOCHS", 60))

# Canonical full manifest (256 rows). Captured at import so per-fold working
# manifests never clobber it.
FULL_MANIFEST = config.MANIFEST_CSV


def case_strata(report):
    """Stratify by the rarest tier a case carries, so tail-bearing cases spread
    across folds instead of clumping into one."""
    text = str(report).lower()
    tiers = {TIER[t] for t in TERMS if any(s in text for s in SYNONYMS[t])}
    for level in ("tail", "medium", "head"):
        if level in tiers:
            return level
    return "head"


def _fold_frame(fold):
    """Deterministic fold split: return the 256-row manifest with 'split' set so
    this fold's test_idx are 'test'. Recomputed identically in every subprocess
    (fixed seed), so fold k always means the same held-out cases."""
    full = pd.read_csv(FULL_MANIFEST)
    strata = full["report_text"].apply(case_strata).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=config.SEED)
    _, test_idx = list(skf.split(full, strata))[fold]
    man = full.copy()
    man["split"] = "train"
    man.loc[man.index[test_idx], "split"] = "test"
    return man


def _gen_ckpt(config_name, fold):
    return f"./breast_report_gen_{config_name}_fold{fold}.pt"


def _pred_path(config_name, fold):
    return f"./pred_{config_name}_fold{fold}.csv"


def fold_done(config_name, fold):
    """A fold is reusable only if its prediction CSV exists and — when the
    external generator is in play — its generator checkpoint exists too (so a
    fold that crashed mid-generator-training is correctly re-run)."""
    if not os.path.exists(_pred_path(config_name, fold)):
        return False
    if config.DRAFT_MODE == "external" and not os.path.exists(_gen_ckpt(config_name, fold)):
        return False
    return True


def run_fold(config_name, fold):
    """Train + evaluate a single fold, writing its prediction CSV. Intended to
    run as a standalone subprocess so memory is fully reclaimed on exit."""
    man = _fold_frame(fold)
    work_manifest = f"./_work_manifest_{config_name}_fold{fold}.csv"
    work_test = f"./_work_test_{config_name}_fold{fold}.csv"
    man.to_csv(work_manifest, index=False)
    man[man["split"] == "test"].to_csv(work_test, index=False)
    config.MANIFEST_CSV = work_manifest
    config.TEST_MANIFEST = work_test

    build_index.main()                            # index this fold's train only

    config.CLF_CKPT = f"./interp_ldam_drw_{config_name}_fold{fold}.pt"
    train_finding_agent.train_and_eval(
        man, CLF_EPOCHS_PER_FOLD, config.CLF_CKPT, config.DEVICE,
        log_prefix=f"[{config_name}][fold {fold}][clf] ")

    # The external generator leaks across folds too if reused, so retrain it
    # on this fold's train split whenever DRAFT_MODE requires it.
    if config.DRAFT_MODE == "external":
        config.GEN_CKPT = _gen_ckpt(config_name, fold)
        train_generator.train_and_eval(
            man, GEN_EPOCHS_PER_FOLD, config.GEN_CKPT, config.DEVICE,
            log_prefix=f"[{config_name}][fold {fold}][gen] ")

    run_phase_a.CONFIG_NAME = config_name
    config.PRED_OUT = _pred_path(config_name, fold)
    run_phase_a.main()
    n = len(pd.read_csv(config.PRED_OUT))
    print(f"  [{config_name}] fold {fold} done ({n} test cases)")
    for f in (work_manifest, work_test):
        try:
            os.remove(f)
        except OSError:
            pass


def run_fold_infer(dest_row, src_row, ablation, fold):
    """Inference-only fold: reuse src_row's already-trained per-fold classifier
    and generator checkpoints (NO retraining), run the `ablation` graph over this
    fold's test split, and write predictions under dest_row. Used to re-evaluate
    a pipeline change (e.g. the Phase B LLM refiner) without paying to retrain."""
    man = _fold_frame(fold)
    work_manifest = f"./_work_manifest_{dest_row}_fold{fold}.csv"
    work_test = f"./_work_test_{dest_row}_fold{fold}.csv"
    man.to_csv(work_manifest, index=False)
    man[man["split"] == "test"].to_csv(work_test, index=False)
    config.MANIFEST_CSV = work_manifest
    config.TEST_MANIFEST = work_test

    build_index.main()                            # fold train index (for retrieval)

    config.CLF_CKPT = f"./interp_ldam_drw_{src_row}_fold{fold}.pt"
    config.GEN_CKPT = _gen_ckpt(src_row, fold)
    for ck in (config.CLF_CKPT, config.GEN_CKPT):
        if not os.path.exists(ck):
            raise FileNotFoundError(f"missing reuse checkpoint: {ck}")

    run_phase_a.CONFIG_NAME = ablation            # graph behaviour (e.g. "full")
    config.PRED_OUT = _pred_path(dest_row, fold)
    run_phase_a.main()
    n = len(pd.read_csv(config.PRED_OUT))
    print(f"  [{dest_row}<-{src_row}/{ablation}] fold {fold} done ({n} test cases)")
    for f in (work_manifest, work_test):
        try:
            os.remove(f)
        except OSError:
            pass


def aggregate(config_name):
    """Pool the per-fold prediction CSVs for one ablation row and print its
    Table 1 / Table 2. Written by run_fold; every image appears exactly once."""
    cv_pred = f"./pred_cv_{config_name}.csv"
    parts = [pd.read_csv(_pred_path(config_name, k)) for k in range(N_FOLDS)]
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(cv_pred, index=False)

    preds = df["pred_report"].fillna("").tolist()
    golds = df["gold_report"].fillna("").tolist()
    bleu = sacrebleu.corpus_bleu(preds, [golds]).score
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL = 100 * np.mean([rs.score(g, p)["rougeL"].fmeasure for g, p in zip(golds, preds)])
    meteor = 100 * np.mean([meteor_score([g.split()], p.split()) for g, p in zip(golds, preds)])

    pred_sets = [_asserted(json.loads(f)) for f in df["findings_json"]]
    gold_sets = [set(json.loads(g)) for g in df["gold_terms"]]
    ff1 = finding_f1(pred_sets, gold_sets)
    by_tier, support, covered = {}, {}, {}
    for tier in ("head", "medium", "tail"):
        tset = [t for t in TERMS if TIER[t] == tier]
        by_tier[tier] = finding_f1(pred_sets, gold_sets, subset=tset)
        support[tier] = sum(any(t in g for t in tset) for g in gold_sets)
        covered[tier] = len({t for g in gold_sets for t in g if t in tset})

    pairs = []
    for f, g in zip(df["findings_json"], gold_sets):
        pr = json.loads(f)
        pairs += [(float(pr.get(t, 0.0)), 1.0 if t in g else 0.0) for t in TERMS]
    ece, brier = calibration(pairs)

    print(f"\n=== {config_name}  |  {N_FOLDS}-fold CV  |  n={len(df)} ===")
    print("Table 1 (aggregate):")
    print(f"  BLEU-4 {bleu:6.2f}   ROUGE-L {rougeL:6.2f}   METEOR {meteor:6.2f}")
    print(f"  Finding-F1 {ff1:.3f}   ECE {ece:.3f}   Brier {brier:.3f}")
    print("\nTable 2 (per-tier Finding-F1, with support):")
    print(f"  {'tier':>7} {'F1':>7} {'test cases':>11} {'terms seen':>11}")
    for tier in ("head", "medium", "tail"):
        n_terms = sum(TIER[t] == tier for t in TERMS)
        print(f"  {tier:>7} {by_tier[tier]:7.3f} {support[tier]:11d} "
              f"{covered[tier]:>6}/{n_terms}")
    print(f"  {'macro':>7} {np.mean(list(by_tier.values())):7.3f}")

    # Table 2b: context for the low macro Finding-F1 -- macro over 28 terms
    # averages many near-unlearnable long-tail terms and uses a fixed threshold.
    # micro-F1 (label-weighted), a low-threshold macro, and support-stratified
    # macro show how much of the "lowness" is the tail vs the threshold.
    import diagnose_findings as diag
    P = np.array([[json.loads(f).get(t, 0.0) for t in TERMS] for f in df["findings_json"]])
    Gm = np.array([[1.0 if t in gs else 0.0 for t in TERMS] for gs in gold_sets])
    sup = Gm.sum(0)
    idx_all = list(range(len(TERMS)))
    idx20 = [j for j in idx_all if sup[j] >= 20]
    hi, lo = config.HIGH_CONF, config.LOW_CONF
    never = int(((P >= hi).sum(0) == 0).sum())
    print("\nTable 2b (Finding-F1 context):")
    print(f"  micro-F1 {diag.micro_f1(P,Gm,idx_all,hi):.3f}   "
          f"macro@{lo:.2f} {diag.macro_f1(P,Gm,idx_all,lo):.3f}   "
          f"macro(support>=20, n={len(idx20)}) {diag.macro_f1(P,Gm,idx20,hi):.3f}")
    print(f"  {never}/{len(TERMS)} terms never asserted at {hi:.2f} "
          f"(hard macro ceiling {(len(TERMS)-never)/len(TERMS):.3f})  |  "
          f"full breakdown: python diagnose_findings.py {config_name}")


def sweep(rows):
    """Run each ablation row fold-by-fold, launching every fold as its own
    subprocess so per-fold memory is fully released before the next starts.
    Already-completed folds are skipped, so an interrupted sweep resumes."""
    for name in rows:
        for fold in range(N_FOLDS):
            if fold_done(name, fold):
                print(f"[{name}] fold {fold} already complete -> skipping")
                continue
            print(f"[{name}] fold {fold} -> launching subprocess")
            subprocess.run([sys.executable, __file__, "fold", name, str(fold)],
                           check=True)
        aggregate(name)


def sweep_infer(dest_row, src_row, ablation):
    """Inference-only sweep: re-run `ablation`'s graph over every fold reusing
    src_row's trained checkpoints, writing predictions under dest_row. Each fold
    is its own subprocess (memory) and resumable. Use to score a pipeline change
    (e.g. the LLM refiner: dest=full_llm, src=full, ablation=full)."""
    for fold in range(N_FOLDS):
        if os.path.exists(_pred_path(dest_row, fold)):
            print(f"[{dest_row}] fold {fold} already complete -> skipping")
            continue
        print(f"[{dest_row}<-{src_row}/{ablation}] fold {fold} -> launching subprocess")
        subprocess.run([sys.executable, __file__, "foldinfer",
                        dest_row, src_row, ablation, str(fold)], check=True)
    aggregate(dest_row)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "fold":                    # internal: one training fold
        run_fold(args[1], int(args[2]))
    elif args and args[0] == "foldinfer":             # internal: one inference fold
        run_fold_infer(args[1], args[2], args[3], int(args[4]))
    elif args and args[0] == "infer":                 # infer <dest> <src> <ablation>
        sweep_infer(args[1], args[2], args[3] if len(args) > 3 else args[1])
    else:
        rows = args or [CONFIG_NAME]
        if rows == ["all"]:
            rows = ABLATION_ROWS
        sweep(rows)
