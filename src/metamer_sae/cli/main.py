"""Command-line interface: `metamer-sae <command> --config configs/experiments/<file>.yaml`."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import pandas as pd

from ..config import load_config


def _cfg(a):
    return load_config(a.config)


def cmd_validate(a):
    c = _cfg(a); print(f"OK: {c.experiment.id}"); return 0


def cmd_split_audit(a):
    from ..data.splits import assert_disjoint, load_ledger, split_hash
    c = _cfg(a); led = load_ledger(c.stimuli.split_manifest)
    assert_disjoint({s: g.stimulus_id.tolist() for s, g in led.groupby("split")})
    print(led.split.value_counts().to_dict(), "sha256:", split_hash(led)); return 0


def cmd_write_pilot_ledger(a):
    from ..data.splits import pilot_ledger
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pilot_ledger().to_csv(a.out, index=False); print("wrote", a.out); return 0


def cmd_inventory(a):
    from ..feather.adapter import FeatherMetadata, ensure_checkout, inventory_availability
    c = _cfg(a); f = c.feather
    meta = FeatherMetadata(ensure_checkout(f.repository, f.local_dir, f.commit))
    if a.all_conditions:
        out = meta.enumerate_conditions(); out.to_csv(a.out, index=False)
    else:
        out = inventory_availability(meta, f.conditions, f.candidate_ids, a.workers); out.to_csv(a.out, index=False)
        piv = out.pivot_table(index="source_image_id", columns="role", values="available", aggfunc="all")
        print("complete triplets:", int(piv.all(axis=1).sum()), "of", len(piv))
    print("wrote", a.out); return 0


def cmd_download(a):
    from ..data.download import fetch_validated
    from ..data.manifest import build_stimulus_manifest, stimulus_id
    from ..data.splits import load_ledger
    from ..feather.adapter import FeatherMetadata, ensure_checkout
    c = _cfg(a); f = c.feather
    meta = FeatherMetadata(ensure_checkout(f.repository, f.local_dir, f.commit))
    led = load_ledger(c.stimuli.split_manifest)
    ctx = {k.id: (k, meta.context(k.experiment_key, k.model_key, k.layer)) for k in f.conditions}
    recs = []
    for sid in led.stimulus_id.astype(int):
        jobs = [("source", "source", f.conditions[0].experiment_key, "", "", -1, meta.source_url(f.conditions[0].experiment_key, sid),
                 Path(c.stimuli.image_dir) / c.paths.feather_role_dirs["source"] / f"{sid:04d}.png")]
        for k, (cond, cx) in ctx.items():
            jobs.append((cond.role, cond.id, cond.experiment_key, cond.model_key, cond.layer, cx.layer_index,
                         meta.metamer_url(cx, sid), Path(c.stimuli.image_dir) / c.paths.feather_role_dirs[cond.role] / f"{sid:04d}.png"))
        for role, cid, ek, mk, layer, li, url, dest in jobs:
            r = fetch_validated(url, dest, throttle_s=a.throttle)
            recs.append({"stimulus_id": stimulus_id(role, cid if role != "source" else "", sid), "source_image_id": sid,
                         "feather_commit": f.commit, "feather_experiment_key": ek, "generator_condition": cid or "source",
                         "generator_model_key": mk, "generator_layer_key": layer, "generator_layer_index": li,
                         "stimulus_role": role, **{k: v for k, v in r.items() if k in
                         ("image_path", "image_sha256", "image_width", "image_height", "image_format",
                          "availability_status", "retrieval_timestamp", "retrieval_source")}})
    build_stimulus_manifest(recs).to_csv(c.stimuli.manifest, index=False)
    bad = [r for r in recs if r["availability_status"] != "ok"]
    print(f"downloaded/validated {len(recs) - len(bad)}; failed {len(bad)}"); return 1 if bad else 0


def cmd_extract(a):
    from ..data.natural import list_natural_images
    from ..data.splits import load_ledger
    from ..extraction.engine import extract, verify
    from ..models.observer import build_observer
    c = _cfg(a); dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    obs = build_observer(c.observer, dev)
    if a.target in ("natural", "all"):
        for split in ("train", "validation"):
            items = list_natural_images(c.natural_images.root, split)
            n = getattr(c.natural_images, f"{split}_samples")
            assert len(items) == n, f"{split}: found {len(items)} images, config expects {n}"
            out = extract(obs, items, Path(c.paths.natural_activations) / split, "sample_id", a.batch_size, dev)
            print(verify(out, n))
    if a.target in ("feather", "all"):
        led = load_ledger(c.stimuli.split_manifest).sort_values("activation_index")
        for role, d in c.paths.feather_role_dirs.items():
            items = led.assign(sample_id=led.stimulus_id, image_path=[str(Path(c.stimuli.image_dir) / d / f"{int(i):04d}.png") for i in led.stimulus_id])
            out = extract(obs, items, Path(c.paths.feather_activations) / d, "sample_id", a.batch_size, dev)
            print(verify(out, len(led)))
    return 0


def cmd_fit_pca(a):
    from ..pipeline import run_fit_pca
    print(run_fit_pca(_cfg(a), a.run_root)); return 0


def cmd_train_sae(a):
    from ..pipeline import run_train_sae
    print(run_train_sae(_cfg(a), a.run_root, a.seed, a.width, a.top_k)); return 0


def cmd_evaluate(a):
    from ..pipeline import run_evaluate
    d, _, c = run_evaluate(_cfg(a), a.run_root); print(d); print(c[["contrast", "mean_paired_difference", "wilcoxon_p_holm"]].to_string()); return 0


def cmd_parity(a):
    from ..evaluation.parity import check_parity
    from ..pipeline import build_representations, run_evaluate
    cfg = _cfg(a); d, pair, con = run_evaluate(cfg, a.run_root)
    reps = build_representations(cfg)
    from ..pipeline import open_pools
    ids = open_pools(cfg).split_ids(cfg.evaluation.primary_split)
    tab = check_parity(cfg, pair, con, ids, reps); tab.to_csv(d / "parity_report.csv", index=False)
    print(tab.to_string()); return 0 if tab["pass"].all() else 1


# ---------------- expanded-study commands (README scope: ResNet-50 + AlexNet + IPCL) ----------------
def _mcfg(a):
    from ..config.matrix import load_matrix
    return load_matrix(a.config)


def cmd_validate_matrix(a):
    c = _mcfg(a); from ..registry import load_registry
    reg = load_registry(c.registry)
    for s in c.conditions:
        reg.get(s.condition_id)
    print(f"OK: {c.experiment_id} ({c.track}); {len(c.conditions)} conditions"); return 0


def cmd_resolve_registry(a):
    from ..feather.adapter import FeatherMetadata, ensure_checkout
    from ..registry import adversarial_ipcl_audit, load_registry, resolve_registry
    c = _mcfg(a); meta = FeatherMetadata(ensure_checkout(c.feather_repository, c.feather_dir, c.feather_commit))
    res = resolve_registry(load_registry(c.registry), sorted(meta.networks)); res.to_csv(a.out, index=False)
    print(res[["condition_id", "status", "resolved_model_key"]].to_string())
    print(json.dumps(adversarial_ipcl_audit(sorted(meta.networks)), indent=2)); return 0


def cmd_matrix_inventory(a):
    from ..pipeline_matrix import run_inventory
    print(run_inventory(_mcfg(a), a.run_root, probe=not a.no_probe)); return 0


def cmd_matrix_split(a):
    from ..pipeline_matrix import run_split
    print(run_split(_mcfg(a), a.pairs, a.require or None)); return 0


def cmd_fit_space(a):
    from ..pipeline_matrix import run_fit_space
    print(run_fit_space(a.space_dir, _mcfg(a), a.run_root)); return 0


def cmd_image_baselines(a):
    from ..evaluation.image_baselines import score_pairs
    pairs, stim = pd.read_csv(a.pairs), pd.read_csv(a.stimuli)
    df = score_pairs(pairs, stim, lpips_net=None if a.no_lpips else "alex"); df.to_csv(a.out, index=False)
    print("wrote", a.out, len(df)); return 0


def cmd_human_inspect(a):
    from ..human.behavior import inspect_behavior_file
    print(json.dumps(inspect_behavior_file(a.file), indent=2)); return 0


def cmd_human_link(a):
    from ..human.behavior import validate_linkage
    r = validate_linkage(pd.read_csv(a.behavior), pd.read_csv(a.conditions), a.out)
    print(json.dumps(r, indent=2)); return 0 if r["linkage_validated"] else 1


def cmd_acceptance(a):
    from ..reporting.coverage import acceptance_report
    t = acceptance_report(a.root); print(t.to_string()); return 0


def cmd_coverage(a):
    from ..data.matrix_manifests import coverage_matrix
    from ..reporting.coverage import coverage_figure
    cov = coverage_matrix(pd.read_csv(a.pairs)); cov.to_csv(a.out_prefix + ".csv", index=False)
    coverage_figure(cov, a.out_prefix); return 0


def _add_matrix_commands(sub):
    def add(name, fn, help):
        s = sub.add_parser(name, help=help); s.set_defaults(fn=fn); return s
    s = add("validate-matrix", cmd_validate_matrix, "validate an expanded-study YAML"); s.add_argument("--config", required=True)
    s = add("resolve-registry", cmd_resolve_registry, "audit registry vs Feather metadata + adversarial-IPCL gate (network)")
    s.add_argument("--config", required=True); s.add_argument("--out", default="data/manifests/registry_resolution.csv")
    s = add("matrix-inventory", cmd_matrix_inventory, "model/layer/stimulus inventory + manifests (network)")
    s.add_argument("--config", required=True); s.add_argument("--run-root", default="outputs/runs"); s.add_argument("--no-probe", action="store_true")
    s = add("matrix-split", cmd_matrix_split, "grouped source-level split over the pair manifest")
    s.add_argument("--config", required=True); s.add_argument("--pairs", default="data/manifests/metamer_pair_manifest.csv"); s.add_argument("--require", nargs="*")
    s = add("fit-space", cmd_fit_space, "fit PCA + SAE seeds in one activation space (natural only)")
    s.add_argument("--config", required=True); s.add_argument("--space-dir", required=True); s.add_argument("--run-root", default="outputs/runs")
    s = add("image-baselines", cmd_image_baselines, "pixel/SSIM/LPIPS + smoothness per pair")
    s.add_argument("--pairs", required=True); s.add_argument("--stimuli", required=True); s.add_argument("--out", default="outputs/metrics/image_baselines.csv"); s.add_argument("--no-lpips", action="store_true")
    s = add("human-inspect", cmd_human_inspect, "report the structure of a behaviour file (no analysis)"); s.add_argument("--file", required=True)
    s = add("human-link", cmd_human_link, "validate behaviour<->condition linkage (gates human analysis)")
    s.add_argument("--behavior", required=True); s.add_argument("--conditions", default="data/manifests/condition_manifest.csv"); s.add_argument("--out", default="outputs/reports/linkage_report.json")
    s = add("acceptance", cmd_acceptance, "evidence-based audit of README scope section 15"); s.add_argument("--root", default=".")
    s = add("coverage", cmd_coverage, "coverage table + heatmap from the pair manifest"); s.add_argument("--pairs", default="data/manifests/metamer_pair_manifest.csv"); s.add_argument("--out-prefix", default="outputs/figures/coverage")


def main(argv=None):
    p = argparse.ArgumentParser("metamer-sae"); sub = p.add_subparsers(dest="cmd", required=True)
    def add(name, fn, **kw):
        s = sub.add_parser(name, help=kw.pop("help", None)); s.set_defaults(fn=fn)
        if kw.get("cfg", True):
            s.add_argument("--config", required=True)
        return s
    add("validate-config", cmd_validate, help="validate a YAML config")
    add("split-audit", cmd_split_audit, help="check split disjointness/counts and print hash")
    s = add("write-pilot-ledger", cmd_write_pilot_ledger, cfg=False, help="write the fixed 8/32/8 split ledger"); s.add_argument("--out", default="data/manifests/stimulus_allocation.csv")
    s = add("feather-inventory", cmd_inventory, help="probe Feather availability (network)"); s.add_argument("--out", default="data/manifests/availability.csv"); s.add_argument("--workers", type=int, default=16); s.add_argument("--all-conditions", action="store_true")
    s = add("download", cmd_download, help="validated download of configured stimuli"); s.add_argument("--throttle", type=float, default=0.0)
    s = add("extract", cmd_extract, help="observer activations"); s.add_argument("--target", choices=["natural", "feather", "all"], default="all"); s.add_argument("--batch-size", type=int, default=64)
    for n, fn in (("fit-pca", cmd_fit_pca), ("evaluate", cmd_evaluate), ("parity", cmd_parity)):
        s = add(n, fn); s.add_argument("--run-root", default="outputs/runs")
    s = add("train-sae", cmd_train_sae, help="train a NEW TopK SAE run (never overwrites pilot checkpoints)"); s.add_argument("--run-root", default="outputs/runs"); s.add_argument("--seed", type=int, required=True); s.add_argument("--width", type=int); s.add_argument("--top-k", type=int)
    _add_matrix_commands(sub)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
