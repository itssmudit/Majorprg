import os
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
sigmoid = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))

N_CLIENTS, ROUNDS, CLIP, RUNS = 5, 30, 1.0, 30
SIGMAS = [0.5, 1, 2, 5, 10, 20]

# ---------- Dataset loaders: each returns X (features) and y (0/1 labels) ----------
def find_file(keyword):
    # look in ./data first, then the current folder, for a file whose name contains keyword
    for folder in ("data", "."):
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                path = os.path.join(folder, f)
                if (os.path.isfile(path) and keyword in f.lower()
                        and not f.lower().endswith((".py", ".png"))):
                    return path
    raise FileNotFoundError(f"no file with '{keyword}' in its name (looked in data/ and .)")

def load_breast_cancer_data():
    try:
        from sklearn.datasets import load_breast_cancer
        return load_breast_cancer(return_X_y=True)
    except ImportError:   # fallback if Windows blocks scikit-learn's compiled files
        import importlib.util
        d = importlib.util.find_spec("sklearn").submodule_search_locations[0]
        raw = np.loadtxt(os.path.join(d, "datasets", "data", "breast_cancer.csv"),
                         delimiter=",", skiprows=1)
        return raw[:, :-1], raw[:, -1].astype(int)

def load_csv(keyword):
    # numeric CSV, no header row, label in the last column
    path = find_file(keyword)
    print(f"  using file: {path}")
    raw = np.genfromtxt(path, delimiter=",")
    raw = raw[~np.isnan(raw).any(axis=1)]              # drop rows with missing values
    return raw[:, :-1], (raw[:, -1] > 0).astype(int)   # label -> 0 / 1

DATASETS = {
    "Breast Cancer": load_breast_cancer_data,
    "Pima Diabetes": lambda: load_csv("diabetes"),
    "Heart Disease": lambda: load_csv("heart"),
    # to add another: "My Data": lambda: load_csv("keyword_in_filename"),
}

# ---------- The experiment (same for every dataset) ----------
def run_dataset(X, y):
    # stratified 80/20 split
    test = np.zeros(len(y), dtype=bool)
    for c in (0, 1):
        ids = rng.permutation(np.where(y == c)[0])
        test[ids[: int(0.2 * len(ids))]] = True
    Xtr, ytr, Xte, yte = X[~test], y[~test], X[test], y[test]

    # standardise with training statistics, then add a bias column
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-12
    bias = lambda A: np.hstack([A, np.ones((len(A), 1))])
    Xtr, Xte = bias((Xtr - mu) / sd), bias((Xte - mu) / sd)

    def metrics(w):
        pred = (sigmoid(Xte @ w) > 0.5).astype(int)
        tp = int(((pred == 1) & (yte == 1)).sum())
        fp = int(((pred == 1) & (yte == 0)).sum())
        fn = int(((pred == 0) & (yte == 1)).sum())
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return dict(acc=float((pred == yte).mean()), prec=prec, rec=rec,
                    f1=2 * prec * rec / max(prec + rec, 1e-12))

    def train(w, Xc, yc, epochs=5, lr=0.1):
        w = w.copy()
        for _ in range(epochs):
            w -= lr * Xc.T @ (sigmoid(Xc @ w) - yc) / len(yc)
        return w

    # centralised baseline
    central = metrics(train(np.zeros(Xtr.shape[1]), Xtr, ytr, epochs=500))

        # split training data across non-IID "hospitals" (retry until every client has >= 20 patients)
    while True:
        idx = [[] for _ in range(N_CLIENTS)]
        for c in (0, 1):
            ids = rng.permutation(np.where(ytr == c)[0])
            cuts = (np.cumsum(rng.dirichlet([0.5] * N_CLIENTS)) * len(ids)).astype(int)[:-1]
            for k, part in enumerate(np.split(ids, cuts)):
                idx[k] += part.tolist()
        if min(len(i) for i in idx) >= 20:
            break
    clients = [(Xtr[i], ytr[i]) for i in idx]

    def federated(clip=None, sigma=0.0):
        w = np.zeros(Xtr.shape[1])
        for _ in range(ROUNDS):
            deltas = []
            for Xc, yc in clients:
                d = train(w, Xc, yc) - w
                if clip:                                    # DP: clip the update
                    d *= min(1.0, clip / (np.linalg.norm(d) + 1e-12))
                deltas.append(d)
            avg = np.mean(deltas, axis=0)
            if clip and sigma > 0:                          # DP: add Gaussian noise
                avg += rng.normal(0, sigma * clip / len(clients), size=w.shape)
            w += avg
        return metrics(w)

    fl = federated()
    runs = {s: [federated(CLIP, s)["acc"] for _ in range(RUNS)] for s in SIGMAS}
    return dict(central=central, fl=fl,
                mean=[np.mean(runs[s]) for s in SIGMAS],
                std=[np.std(runs[s]) for s in SIGMAS],
                majority=max(yte.mean(), 1 - yte.mean()),
                n=len(y), d=X.shape[1], clients=[len(c[1]) for c in clients])

def rough_epsilon(sigma, rounds=ROUNDS, delta=1e-5):
    # Renyi-DP accounting for the Gaussian mechanism, converted to (eps, delta)-DP
    alphas = np.linspace(1.01, 2000, 200000)
    eps = rounds * alphas / (2 * sigma ** 2) + np.log(1 / delta) / (alphas - 1)
    return float(eps.min())

# ---------- Run all datasets first (tables print here) ----------
results = {}
for name, loader in DATASETS.items():
    print(f"\nLoading {name}...")
    try:
        X, y = loader()
    except Exception as e:
        print(f"Skipping {name}: {e}")
        continue
    r = run_dataset(X, y)
    results[name] = r
    print(f"=== {name}: {r['n']} samples, {r['d']} features, client sizes {r['clients']} ===")
    print(f"{'Setup':<22}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}")
    for label, m in [("Centralised", r["central"]), ("FedAvg (no DP)", r["fl"])]:
        print(f"{label:<22}{m['acc']:>7.3f}{m['prec']:>7.3f}{m['rec']:>7.3f}{m['f1']:>7.3f}")
    print("FL + DP:")
    for s, a, sd_ in zip(SIGMAS, r["mean"], r["std"]):
        print(f"  sigma={s:<5} approx eps={rough_epsilon(s):>8.1f}  acc={a:.3f} +/- {sd_:.3f}")

# ---------- Plotting ----------
def plot_one(name, r, ax):
    ax.errorbar(SIGMAS, r["mean"], yerr=r["std"], fmt="o-", capsize=3, label="FL + DP")
    ax.axhline(r["fl"]["acc"], ls="--", c="g", label="FL (no DP)")
    ax.axhline(r["central"]["acc"], ls=":", c="r", label="Centralised")
    ax.axhline(r["majority"], ls="-.", c="gray", label="Majority guess")
    ax.set_xscale("log"); ax.set_title(name); ax.grid(alpha=.3)
    ax.set_xlabel("Noise multiplier (sigma)"); ax.set_ylabel("Test accuracy")

# one window per dataset: close it to see the next one
for name, r in results.items():
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    plot_one(name, r, ax)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"privacy_utility_{name.lower().replace(' ', '_')}.png", dpi=200)
    plt.show()

# combined figure: saved to a file only, no window
fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
axes = np.atleast_1d(axes)
for ax, (name, r) in zip(axes, results.items()):
    plot_one(name, r, ax)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("privacy_utility_all.png", dpi=200)
plt.show()

# combined figure: saved and shown at the end
fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
axes = np.atleast_1d(axes)
for ax, (name, r) in zip(axes, results.items()):
    plot_one(name, r, ax)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("privacy_utility_all.png", dpi=200)
plt.show()