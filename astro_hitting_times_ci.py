import argparse
import pickle
from joblib import Parallel, delayed
import multiprocessing as mp
from time import time
import os
import numpy as np
from tqdm import tqdm

SEED = 12345
rng = np.random.default_rng(SEED)

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n_resamples", default=1000, type=int)
parser.add_argument("-v", default=1, type=int)
parser.add_argument("--delta", default=0.05, type=float)
parser.add_argument("--path", "-p", default="results", type=str)
parser.add_argument("--parallel", default=1, type=int)

# Set arguments
args = parser.parse_args()
n_resamples = args.n_resamples
delta = args.delta
verbose = args.v > 0
pickle_path = args.path
parallel = args.parallel > 0


def get_bootstrap_indices(len, n_resamples=10, rng=np.random.default_rng(0)):
    for _ in range(n_resamples):
        # Slight hack to avoid memory crashing while on server:
        # subsample as well as resample...
        yield rng.choice(np.arange(len), size=len // 25, replace=True)


def run(n_resamples, delta, unique, counts, verbose, seed):
    means_bootstrap = []
    stds_bootstrap = []
    q05s_bootstrap = []
    q25s_bootstrap = []
    q75s_bootstrap = []
    q50s_bootstrap = []
    q95s_bootstrap = []

    rng = np.random.default_rng(seed)

    samples = np.concatenate([np.repeat([u], n) for u, n in zip(unique, counts)])

    bootstrap_indices = get_bootstrap_indices(len(samples), n_resamples=n_resamples, rng=rng)
    for indices in tqdm(bootstrap_indices, total=n_resamples, disable=not verbose):
        samples_ = samples[indices]
        means_bootstrap.append(np.mean(samples_))
        stds_bootstrap.append(np.std(samples_))
        q05s_bootstrap.append(np.quantile(samples_, q=0.05))
        q25s_bootstrap.append(np.quantile(samples_, q=0.25))
        q50s_bootstrap.append(np.quantile(samples_, q=0.5))
        q75s_bootstrap.append(np.quantile(samples_, q=0.75))
        q95s_bootstrap.append(np.quantile(samples_, q=0.95))

    return {
        "means_bootstrap": means_bootstrap,
        "stds_bootstrap": stds_bootstrap,
        "q05s_bootstrap": q05s_bootstrap,
        "q25s_bootstrap": q25s_bootstrap,
        "q50s_bootstrap": q50s_bootstrap,
        "q75s_bootstrap": q75s_bootstrap,
        "q95s_bootstrap": q95s_bootstrap,
    }


def MC_xp(args, pickle_path=None, caption="xp"):
    n_resamples, delta, verbose, seed = args

    with open("results/astro_hitting_time_159071.pkl", "rb") as f:
        res = pickle.load(f)
        unique = res["results"]["unique"].astype("int32")
        counts = res["results"]["counts"].astype("int32")

    res = run(n_resamples, delta, unique, counts, verbose, seed)

    if pickle_path is not None:
        pickle.dump(res, open(os.path.join(pickle_path, caption + ".pkl"), "wb"))
    return res


def multiprocess_MC(args, pickle_path=None, caption="xp", parallel=True):
    t0 = time()
    cpu = mp.cpu_count()
    print("Running on %i clusters" % cpu)
    n_resamples, delta, verbose = args
    new_args = (n_resamples // cpu + 1, delta, verbose)
    if parallel:
        res_ = Parallel(n_jobs=cpu)(delayed(MC_xp)(new_args + (i,)) for i in range(cpu))
        means_bootstrap = np.concatenate(
            [res_[i]["means_bootstrap"] for i in range(cpu)]
        )
        stds_bootstrap = np.concatenate([res_[i]["stds_bootstrap"] for i in range(cpu)])
        q05s_bootstrap = np.concatenate([res_[i]["q05s_bootstrap"] for i in range(cpu)])
        q25s_bootstrap = np.concatenate([res_[i]["q25s_bootstrap"] for i in range(cpu)])
        q50s_bootstrap = np.concatenate([res_[i]["q50s_bootstrap"] for i in range(cpu)])
        q75s_bootstrap = np.concatenate([res_[i]["q75s_bootstrap"] for i in range(cpu)])
        q95s_bootstrap = np.concatenate([res_[i]["q95s_bootstrap"] for i in range(cpu)])
    else:
        res_ = MC_xp(args + (0,))
        means_bootstrap = res_["means_bootstrap"]
        stds_bootstrap = res_["stds_bootstrap"]
        q05s_bootstrap = res_["q05s_bootstrap"]
        q25s_bootstrap = res_["q25s_bootstrap"]
        q50s_bootstrap = res_["q50s_bootstrap"]
        q75s_bootstrap = res_["q75s_bootstrap"]
        q95s_bootstrap = res_["q95s_bootstrap"]

    mean_bootstrap_ci = [
        np.quantile(means_bootstrap, q=delta / 2),
        np.quantile(means_bootstrap, q=1 - delta / 2),
    ]
    std_bootstrap_ci = [
        np.quantile(stds_bootstrap, q=delta / 2),
        np.quantile(stds_bootstrap, q=1 - delta / 2),
    ]
    q05_bootstrap_ci = [
        np.quantile(q05s_bootstrap, q=delta / 2),
        np.quantile(q05s_bootstrap, q=1 - delta / 2),
    ]
    q25_bootstrap_ci = [
        np.quantile(q25s_bootstrap, q=delta / 2),
        np.quantile(q25s_bootstrap, q=1 - delta / 2),
    ]
    q50_bootstrap_ci = [
        np.quantile(q50s_bootstrap, q=delta / 2),
        np.quantile(q50s_bootstrap, q=1 - delta / 2),
    ]
    q75_bootstrap_ci = [
        np.quantile(q75s_bootstrap, q=delta / 2),
        np.quantile(q75s_bootstrap, q=1 - delta / 2),
    ]
    q95_bootstrap_ci = [
        np.quantile(q95s_bootstrap, q=delta / 2),
        np.quantile(q95s_bootstrap, q=1 - delta / 2),
    ]

    res = {
        "mean_bootstrap_ci": mean_bootstrap_ci,
        "std_bootstrap_ci": std_bootstrap_ci,
        "q05_bootstrap_ci": q05_bootstrap_ci,
        "q25_bootstrap_ci": q25_bootstrap_ci,
        "q50_bootstrap_ci": q50_bootstrap_ci,
        "q75_bootstrap_ci": q75_bootstrap_ci,
        "q95_bootstrap_ci": q95_bootstrap_ci,
    }

    info = {
        "n_resamples": n_resamples,
        "delta": delta,
    }
    xp_container = {"results": res, "info": info}

    if pickle_path is not None:
        pickle.dump(
            xp_container, open(os.path.join(pickle_path, caption + ".pkl"), "wb")
        )
    print("Execution time: {:.0f} seconds".format(time() - t0))
    return xp_container


cap = "astro_hitting_time_ci_" + str(int(np.random.uniform() * 1e6))
print(cap)
res, traj = multiprocess_MC(
    (n_resamples, delta, verbose),
    pickle_path=pickle_path,
    caption=cap,
    parallel=parallel,
)
print(cap)
