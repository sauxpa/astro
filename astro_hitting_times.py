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
parser.add_argument("-T", default=5000, type=int)
parser.add_argument("-M", default=10, type=int)
parser.add_argument("-a", default=2.0, type=float)
parser.add_argument("--tmax", default=int(1e6), type=int)
parser.add_argument("--buffer_size", default=int(1e6), type=int)
parser.add_argument("-v", default=1, type=int)
parser.add_argument("--path", "-p", default="results", type=str)
parser.add_argument("--parallel", default=1, type=int)

# Set arguments
args = parser.parse_args()
T = args.T
M = args.M
a = args.a
t_max = args.tmax
buffer_size = args.buffer_size
verbose = args.v > 0
pickle_path = args.path
parallel = args.parallel > 0

# Astro numbers
n = 4500000
gains = [2, 4, 6, 10, 20, 100, 1000, 25000]
ns = [661500, 476000, 150000, 101350, 45000, 415, 8, 3]

# Add loss event
gains.insert(0, 0)
ns.insert(0, n - sum(ns))

ps = np.array(ns) / n
xs = -(np.array(gains) - a)


class Categorical:
    def __init__(self, xs, ps, rng=np.random.default_rng(0)):
        self.xs = xs
        self.ps = ps
        self.rng = rng

    def rvs(self, size=1):
        return self.rng.choice(self.xs, p=self.ps, size=size)

    @property
    def mean(self):
        return self.xs @ self.ps

    @property
    def var(self):
        return self.ps @ (self.xs ** 2) - (self.ps @ self.xs) ** 2

    @property
    def std(self):
        return np.sqrt(self.var)


class CategoricalNoReplacement:
    def __init__(self, gains, ns, init_cost, rng=np.random.default_rng(0), shuffle=True, no_restart=True):
        self.gains = gains
        self.ns = ns
        self.xs = -(np.concatenate([np.repeat([gain], n) for gain, n in zip(gains, ns)]) - init_cost)
        self.rng = rng
        self.shuffle = shuffle
        self.no_restart = no_restart

    def rvs(self, size=1):
        if isinstance(size, tuple):
            L = np.prod(size)
        else:
            L = size
        if L <= len(self.xs):
            return self.rng.choice(self.xs, replace=False, size=L, shuffle=self.shuffle).reshape(size)
        else:
            if self.no_restart:
                raise Exception("Cannot sample more than {} times without replacement!".format(len(self.xs)))
            ret = []
            l = len(self.xs)
            while L > len(self.xs):
                ret.append(self.rvs(size=l))
                L -= l
            ret.append(self.rvs(size=L))
            return np.concatenate(ret).reshape(size)

    @property
    def mean(self):
        return np.mean(self.xs)

    @property
    def var(self):
        return np.var(self.xs)

    @property
    def std(self):
        return np.std(self.xs)


def run(M, a, t_max, buffer_size, verbose):
    distr = Categorical(xs, ps, rng=rng)
    distr_nr = CategoricalNoReplacement(gains, ns, a    , rng=rng)#

    samples = np.empty(M, dtype=int)

    buffer_idx = 0
    for i in tqdm(range(M), disable=not verbose):
        # init random walk and hitting time
        random_walk = 0.0
        t = 0

        while random_walk < a and t < t_max:
            if buffer_idx % buffer_size == 0:
                increments = distr.rvs(size=buffer_size)
            random_walk += increments[buffer_idx % buffer_size]
            t += 1
            buffer_idx += 1
        samples[i] = t

    samples_nr = np.empty(M, dtype=int)

    buffer_idx = 0
    buffer_size = int(4.5e6)
    for i in tqdm(range(M), disable=not verbose):
        # init random walk and hitting time
        random_walk = 0.0
        t = 0

        while random_walk < a and t < t_max:
            if buffer_idx % buffer_size == 0:
                increments = distr_nr.rvs(size=buffer_size)
            random_walk += increments[buffer_idx % buffer_size]
            t += 1
            buffer_idx += 1
        samples_nr[i] = t

    return {
        "samples": samples,
        "samples_nr": samples_nr,
    }


def MC_xp(args, pickle_path=None, caption="xp"):
    (
        M,
        a,
        t_max,
        buffer_size,
        verbose,
    ) = args
    res = run(M, a, t_max, buffer_size, verbose)

    if pickle_path is not None:
        pickle.dump(res, open(os.path.join(pickle_path, caption + ".pkl"), "wb"))
    return res


def multiprocess_MC(args, pickle_path=None, caption="xp", parallel=True):
    t0 = time()
    cpu = mp.cpu_count()
    print("Running on %i clusters" % cpu)
    M, a, t_max, buffer_size, verbose = args
    new_args = (M // cpu + 1, a, t_max, buffer_size, verbose)
    if parallel:
        res_ = Parallel(n_jobs=cpu)(delayed(MC_xp)(new_args) for _ in range(cpu))
        res = {}
        samples = np.concatenate([res_[i]["samples"] for i in range(cpu)])
        samples_nr = np.concatenate([res_[i]["samples_nr"] for i in range(cpu)])
        unique, counts = np.unique(samples, return_counts=True)
        unique_nr, counts_nr = np.unique(samples_nr, return_counts=True)

        res["unique"] = unique
        res["counts"] = counts
        res["unique_nr"] = unique_nr
        res["counts_nr"] = counts_nr
    else:
        res_ = MC_xp(args)

    info = {
        "M": M,
        "a": a,
        "t_max": t_max,
        "buffer_size": buffer_size,
    }
    xp_container = {"results": res, "info": info}

    if pickle_path is not None:
        pickle.dump(
            xp_container, open(os.path.join(pickle_path, caption + ".pkl"), "wb")
        )
    print("Execution time: {:.0f} seconds".format(time() - t0))
    return xp_container


cap = "astro_hitting_time_" + str(int(np.random.uniform() * 1e6))
print(cap)
res, traj = multiprocess_MC(
    (M, a, t_max, buffer_size, verbose),
    pickle_path=pickle_path,
    caption=cap,
    parallel=parallel,
)
print(cap)
