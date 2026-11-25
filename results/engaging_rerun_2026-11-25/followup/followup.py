"""Quick follow-up descriptives on the from-scratch pilot run (primary split only; reserve rows never loaded)."""
import json, sys, glob
import numpy as np, pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from metamer_sae.config import load_config
from metamer_sae.pipeline import build_representations

cfg = load_config(sys.argv[1])
led = pd.read_csv("data/manifests/stimulus_allocation.csv")
pri = led[led.split == "primary_evaluation"]
idx, sids = pri.activation_index.to_numpy(), pri.stimulus_id.astype(int).tolist()
A = {r: np.load(f"outputs/activations/feather/{d}/pooled_activations.npy", mmap_mode="r")[idx].astype(np.float64)
     for r, d in {"src": "source", "std": "standard_resnet50", "adv": "adversarial_resnet50"}.items()}
va = np.load("outputs/activations/natural_images/validation/pooled_activations.npy").astype(np.float64)
cls = pd.read_csv("outputs/activations/natural_images/validation/metadata.csv").class_wnid.to_numpy()
out = {"n_primary": len(sids)}
rng = np.random.default_rng(0)

# --- natural scale (raw 384-D)
i, j = rng.integers(0, len(va), 40000), rng.integers(0, len(va), 40000)
d = np.linalg.norm(va[i] - va[j], axis=1); ok = i != j
same, diff = ok & (cls[i] == cls[j]), ok & (cls[i] != cls[j])
sub = va[rng.choice(len(va), 1500, replace=False)]
D = np.sqrt(np.maximum((sub ** 2).sum(1)[:, None] + (va ** 2).sum(1)[None] - 2 * sub @ va.T, 0)); D[D < 1e-4] = np.inf
src_pair = np.linalg.norm(A["src"][:, None] - A["src"][None], axis=2)[np.triu_indices(len(sids), 1)]
out["natural_scale_raw_l2"] = {
    "source_to_standard_metamer": float(np.linalg.norm(A["std"] - A["src"], axis=1).mean()),
    "source_to_adversarial_metamer": float(np.linalg.norm(A["adv"] - A["src"], axis=1).mean()),
    "natural_nearest_neighbour_median": float(np.median(D.min(1))),
    "natural_same_class_pair_median": float(np.median(d[same])),
    "natural_different_class_pair_median": float(np.median(d[diff])),
    "different_feather_sources_median": float(np.median(src_pair))}

# --- does the observer still find the right source? (nearest of the 32 primary sources)
def top1(X, S):
    return float((np.linalg.norm(X[:, None] - S[None], axis=2).argmin(1) == np.arange(len(X))).mean())
out["source_retrieval_top1_raw"] = {"standard": top1(A["std"], A["src"]), "adversarial": top1(A["adv"], A["src"])}

# --- is the displacement a shared direction?
def shared(Dm):
    m = Dm.mean(0); U = Dm / np.linalg.norm(Dm, axis=1, keepdims=True)
    C = U @ U.T
    return {"energy_along_mean_direction": float((m @ m) / (Dm ** 2).sum(1).mean()),
            "mean_pairwise_cosine_between_sources": float(C[np.triu_indices(len(Dm), 1)].mean())}
Ds, Da = A["std"] - A["src"], A["adv"] - A["src"]
out["displacement_raw"] = {"standard": shared(Ds), "adversarial": shared(Da),
    "cosine_mean_standard_vs_mean_adversarial": float(Ds.mean(0) @ Da.mean(0) / np.linalg.norm(Ds.mean(0)) / np.linalg.norm(Da.mean(0))),
    "per_source_cosine_standard_vs_adversarial_mean": float(np.mean([a @ b / np.linalg.norm(a) / np.linalg.norm(b) for a, b in zip(Ds, Da)]))}

# --- SAE feature level
reps = build_representations(cfg, "cpu")
out["sae"] = {}
for name in ("sae_seed123", "sae_seed42"):
    Z = {k: reps[name].embed(v) for k, v in A.items()}
    act = {k: z != 0 for k, z in Z.items()}
    ds, da = Z["std"] - Z["src"], Z["adv"] - Z["src"]
    ms, ma = ds.mean(0), da.mean(0)
    order = np.argsort(-np.abs(ms))
    tot = np.abs(ms).sum()
    top = [{"latent": int(k), "mean_change_standard": float(ms[k]), "mean_change_adversarial": float(ma[k]),
            "n_sources_same_sign_standard": int((np.sign(ds[:, k]) == np.sign(ms[k])).sum()),
            "active_in_n_sources": int(act["src"][:, k].sum()), "active_in_n_standard": int(act["std"][:, k].sum()),
            "active_in_n_adversarial": int(act["adv"][:, k].sum())} for k in order[:8]]
    out["sae"][name] = {
        "latents_ever_active_primary": int((act["src"] | act["std"] | act["adv"]).any(0).sum()),
        "features_lost_per_source_of_64": {c: float((act["src"] & ~act[c]).sum(1).mean()) for c in ("std", "adv")},
        "lost_in_standard_also_lost_in_adversarial_fraction": float(((act["src"] & ~act["std"]) & ~act["adv"]).sum() / (act["src"] & ~act["std"]).sum()),
        "gained_in_standard_also_gained_in_adversarial_fraction": float(((~act["src"] & act["std"]) & act["adv"]).sum() / (~act["src"] & act["std"]).sum()),
        "corr_mean_change_standard_vs_adversarial": float(np.corrcoef(ms, ma)[0, 1]),
        "share_of_mean_change_in_top10_latents": float(np.abs(ms[order[:10]]).sum() / tot),
        "top_latents_by_standard_change": top,
        "natural_validation_pairs_support_jaccard_median": None}
    Zv = reps[name].embed(va[rng.choice(len(va), 600, replace=False)]) != 0
    inter = (Zv[:300] & Zv[300:]).sum(1); out["sae"][name]["natural_validation_pairs_support_jaccard_median"] = float(np.median(inter / (128 - inter)))
    out["sae"][name]["support_jaccard"] = {c: float(np.mean([(a & b).sum() / (a | b).sum() for a, b in zip(act["src"], act[c])])) for c in ("std", "adv")}
d1, d2 = (reps[n].dictionary().astype(np.float64) for n in ("sae_seed123", "sae_seed42"))
d1 /= np.linalg.norm(d1, axis=1, keepdims=True); d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
C = d1 @ d2.T; r, c = linear_sum_assignment(-C); m = C[r, c]
out["cross_seed_matching"] = {"mean_matched_cosine": float(m.mean()), "fraction_above_0.9": float((m > .9).mean()), "fraction_above_0.5": float((m > .5).mean())}

# --- pixel baseline
def img(role, s):
    return np.asarray(Image.open(f"data/images/{role}/{s:04d}.png").convert("RGB").resize((224, 224)), np.float64) / 255
px = {c: np.array([np.sqrt(((img(d, s) - img("source", s)) ** 2).mean()) for s in sids]) for c, d in (("std", "standard_resnet50"), ("adv", "adversarial_resnet50"))}
ob = {"std": np.linalg.norm(Ds, axis=1), "adv": np.linalg.norm(Da, axis=1)}
from scipy.stats import spearmanr
out["pixel_baseline"] = {"pixel_rms_standard_mean": float(px["std"].mean()), "pixel_rms_adversarial_mean": float(px["adv"].mean()),
    "n_sources_standard_pixel_gt_adversarial": int((px["std"] > px["adv"]).sum()),
    "spearman_pixel_vs_observer_l2_standard": float(spearmanr(px["std"], ob["std"])[0]),
    "spearman_pixel_vs_observer_l2_adversarial": float(spearmanr(px["adv"], ob["adv"])[0]),
    "spearman_pixel_gap_vs_observer_gap": float(spearmanr(px["std"] - px["adv"], ob["std"] - ob["adv"])[0])}
print(json.dumps(out, indent=1))
