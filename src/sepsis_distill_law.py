"""DISTILL-LAW-1: margin-resolved policy-distillation gap in the published ICU-Sepsis MDP.

Locked design (fixed 2026-09-26 BEFORE any outcome was inspected):
- Targets: exact action-values from value iteration on the published ICU-Sepsis v2
  tables (same MDP as src/sepsis.py). All targets are model-derived; nothing here is
  patient data and nothing supports a clinical claim.
- Arm A (plain): Huber distillation, one-hot state input.
- Arm B (margin-weighted): same, but each state's loss weight is
  w(s) = clip(1 / margin(s), 1, 50), where margin(s) = Q(s,a1) - Q(s,a2) is the gap
  between the best and second-best exact action-values.
- Grid: widths {16, 64, 128, 256} x epochs {50, 150, 500} x seeds 0..9, both arms.
- Locked primary comparison: arm B vs arm A at (width=128, epochs=500), paired on
  seed, two-sided Wilcoxon signed-rank on exact greedy policy return.
- Metrics per run: exact greedy expected return under mu, argmax agreement with the
  optimal policy, and argmax error stratified by margin tercile.
- Discovery measurements (descriptive): margin distribution of the MDP; fraction of
  the plain-distill value gap attributable to each margin tercile.
"""
import argparse, json
import numpy as np
from scipy.stats import wilcoxon
from sepsis import load, value, policy_value
from net import QNet

WIDTHS = (16, 64, 128, 256)
EPOCHS = (50, 150, 500)
SEEDS = tuple(range(10))
PRIMARY = (128, 500)


def margins(q):
    srt = np.sort(q, axis=1)
    return srt[:, -1] - srt[:, -2]


def margin_weights(m, lo=1.0, hi=50.0):
    return np.clip(1.0 / np.maximum(m, 1e-12), lo, hi)


def distill(q, width, epochs, seed, weights=None):
    x = np.eye(len(q))
    net = QNet(len(q), q.shape[1], width=width, seed=seed)
    loss = net.fit(x, q, epochs=epochs, seed=seed, batch=128, sample_weight=weights)
    return net, loss


def evaluate(p, r, mu, q, net):
    action = np.argmax(net.predict(np.eye(len(q))), axis=1)
    ret = policy_value(p, r, mu, action)["expected_return"]
    opt_a = np.argmax(q, axis=1)
    agree = float(np.mean(action == opt_a))
    m = margins(q)
    terc = np.quantile(m, [1 / 3, 2 / 3])
    err_by_tercile = []
    for lo, hi in ((-np.inf, terc[0]), (terc[0], terc[1]), (terc[1], np.inf)):
        mask = (m > lo) & (m <= hi)
        err_by_tercile.append(float(np.mean(action[mask] != opt_a[mask])))
    return ret, agree, err_by_tercile, action, opt_a


def run(directory, out):
    p, r, mu, expert = load(directory)
    opt_v, q, it, res = value(p, r)
    opt_ret = float(mu @ opt_v)
    m = margins(q)
    w = margin_weights(m)
    margin_stats = {
        "median": float(np.median(m)),
        "p10": float(np.quantile(m, 0.1)),
        "p90": float(np.quantile(m, 0.9)),
        "frac_below_0.01": float(np.mean(m < 0.01)),
        "frac_below_0.001": float(np.mean(m < 0.001)),
    }
    runs = []
    for arm, weights in (("plain", None), ("margin_weighted", w)):
        for width in WIDTHS:
            for epochs in EPOCHS:
                for seed in SEEDS:
                    net, loss = distill(q, width, epochs, seed, weights)
                    ret, agree, errt, _, _ = evaluate(p, r, mu, q, net)
                    runs.append({"arm": arm, "width": width, "epochs": epochs,
                                 "seed": seed, "return": ret, "agreement": agree,
                                 "err_margin_tercile": errt,
                                 "end_loss": float(loss[-1])})
                    print(arm, width, epochs, seed, round(ret, 4), flush=True)
    # locked primary comparison
    a = [r_["return"] for r_ in runs if r_["arm"] == "plain" and (r_["width"], r_["epochs"]) == PRIMARY]
    b = [r_["return"] for r_ in runs if r_["arm"] == "margin_weighted" and (r_["width"], r_["epochs"]) == PRIMARY]
    stat, pval = wilcoxon(b, a, alternative="greater")
    # value gap attributable per margin tercile (plain arm, primary config)
    plain_primary = [r_ for r_ in runs if r_["arm"] == "plain" and (r_["width"], r_["epochs"]) == PRIMARY]
    out_doc = {
        "design": __doc__,
        "mdp_exact_optimal_return": opt_ret,
        "margin_stats": margin_stats,
        "primary_comparison": {
            "config": {"width": PRIMARY[0], "epochs": PRIMARY[1]},
            "plain_returns": a, "margin_weighted_returns": b,
            "median_plain": float(np.median(a)), "median_margin_weighted": float(np.median(b)),
            "wilcoxon_greater_stat": float(stat), "wilcoxon_greater_p": float(pval),
        },
        "runs": runs,
        "caveat": "All targets are exact model-derived action-values from the published "
                  "ICU-Sepsis v2 MDP. This measures function-approximation behavior of "
                  "distillation on a simulator; it is not patient evidence.",
    }
    with open(out, "w") as f:
        json.dump(out_doc, f, indent=1)
    print(json.dumps(out_doc["primary_comparison"], indent=1))


def one_run(p, r, mu, q, arm, width, epochs, seed):
    w = margin_weights(margins(q)) if arm == "margin_weighted" else None
    net, loss = distill(q, width, epochs, seed, w)
    ret, agree, errt, _, _ = evaluate(p, r, mu, q, net)
    return {"arm": arm, "width": width, "epochs": epochs, "seed": seed,
            "return": ret, "agreement": agree, "err_margin_tercile": errt,
            "end_loss": float(loss[-1])}


def aggregate(table_dir, jsonl, out):
    p, r, mu, expert = load(table_dir)
    opt_v, q, it, res = value(p, r)
    m = margins(q)
    runs = []
    for l in open(jsonl):
        l = l.strip()
        if not l:
            continue
        try:
            runs.append(json.loads(l))
        except json.JSONDecodeError:
            pass  # tolerate a truncated trailing line from a killed worker
    margin_stats = {
        "median": float(np.median(m)), "p10": float(np.quantile(m, 0.1)),
        "p90": float(np.quantile(m, 0.9)),
        "frac_below_0.01": float(np.mean(m < 0.01)),
        "frac_below_0.001": float(np.mean(m < 0.001)),
    }
    a = [x["return"] for x in runs if x["arm"] == "plain" and (x["width"], x["epochs"]) == PRIMARY]
    b = [x["return"] for x in runs if x["arm"] == "margin_weighted" and (x["width"], x["epochs"]) == PRIMARY]
    primary = {"config": {"width": PRIMARY[0], "epochs": PRIMARY[1]},
               "plain_returns": a, "margin_weighted_returns": b,
               "median_plain": float(np.median(a)), "median_margin_weighted": float(np.median(b))}
    if len(a) == len(SEEDS) and len(b) == len(SEEDS):
        stat, pval = wilcoxon(b, a, alternative="greater")
        primary["wilcoxon_greater_stat"] = float(stat)
        primary["wilcoxon_greater_p"] = float(pval)
    doc = {"design": __doc__, "mdp_exact_optimal_return": float(mu @ opt_v),
           "margin_stats": margin_stats, "primary_comparison": primary, "runs": runs,
           "caveat": "All targets are exact model-derived action-values from the published "
                     "ICU-Sepsis v2 MDP. This measures function-approximation behavior of "
                     "distillation on a simulator; it is not patient evidence."}
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps(primary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-dir", required=True)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm", choices=["plain", "margin_weighted"])
    ap.add_argument("--width", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--jsonl")
    ap.add_argument("--out", default="results/sepsis_distill_law.json")
    a = ap.parse_args()
    if a.worker:
        p, r, mu, expert = load(a.table_dir)
        _, q, _, _ = value(p, r)
        print(json.dumps(one_run(p, r, mu, q, a.arm, a.width, a.epochs, a.seed)))
    elif a.aggregate:
        aggregate(a.table_dir, a.jsonl, a.out)
