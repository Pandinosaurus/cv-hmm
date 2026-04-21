# opencv-hmm

Discrete-observation Hidden Markov Models, implemented as a single
header-only C++ class on top of OpenCV. A gentle starting point if
you're learning about HMMs and want something small you can read, run,
and modify.

## Keywords

hidden markov models, hmm, opencv, cvhmm, opencv-hmm, baum-welch,
viterbi, forward-backward

## The three questions an HMM lets you answer

A Hidden Markov Model is a chain of hidden states that you can't see
directly, and for each hidden state you observe one noisy symbol. The
model has three pieces:

- `TRANS[i, j]` — the probability of moving from state `i` to state `j`
- `EMIS[i, k]`  — the probability of observing symbol `k` while in state `i`
- `INIT[i]`     — the probability of starting in state `i`

Given an HMM and a sequence of observations `O = (O₀, O₁, …, O_{T-1})`,
people usually want to answer one of three questions:

1. **How likely is this sequence under the model?**
   — `P(O | λ)`, computed by `decode()`.

2. **Which hidden states most plausibly produced this sequence?**
   — the state path, computed by `viterbi()`.

3. **Given observations, what model best explains them?**
   — re-estimate `TRANS`, `EMIS`, `INIT` from data with `train()` (one
   sequence) or `trainBatch()` (many sequences).

That's the whole library.

## What you get by running `main.cpp`

The demo defines a small 3-state, 4-symbol Markov model, samples
10 000 observation sequences of length 20 from it, and then uses the
three routines above to (1) score them, (2) decode them, and (3)
learn the model back from just the observations. Trimmed output:

```
--- Ground-truth model (after eps-smoothing) ---

TRANS:
0.5  0.5  ~0
~0   0.7  0.3
~0   ~0   1

EMIS:
0.5  0.5  ~0   ~0
~0   0.5  0.5  ~0
~0   ~0   0.5  0.5

INIT:
1  ~0  ~0

Sampled 10000 sequences of length 20 from the model.
Preview (first 5 sequences, first 20 symbols each):
  O[0] = 02233333332322232333
  X[0] = 01222222222222222222
  O[1] = 11022122223333323332
  X[1] = 00011112222222222222
  ...
```

### Problem 1 — How likely is the observation sequence?

`decode()` uses the forward algorithm to compute `log P(O | λ)`
without running into numerical underflow (probabilities get very
small, very fast, so we work in log-space).

```
Problem 1 (log P(O | lambda) per sequence):
    log P(O[0] | lambda) = -15.2294
    log P(O[1] | lambda) = -16.8397
    log P(O[2] | lambda) = -14.6439
    log P(O[3] | lambda) = -15.0050
    log P(O[4] | lambda) = -15.1645
  mean over all 10000 sequences: -16.11    total: -161096
```

### Problem 2 — Which hidden states produced the observations?

`viterbi()` runs the classic dynamic-programming algorithm to find the
single most-likely hidden-state path.

```
Problem 2 (most-likely state path):
    Viterbi[0] = 01222222222222222222
    Viterbi[1] = 00011122222222222222
    Viterbi[2] = 01122222222222222222
    Viterbi[3] = 01112222222222222222
    Viterbi[4] = 00122222222222222222
  Viterbi matches the true (sampled) state at 190786 / 200000  (95.39 %)
```

### Problem 3 — Learn the model from data

`trainBatch()` starts from a guess of `TRANS`, `EMIS`, `INIT` and
iteratively improves it with the Baum-Welch algorithm (a special case
of Expectation-Maximisation) until the training data's log-likelihood
stops going up.

The interesting part of training is **how you choose the initial
guess**. A few things to keep in mind:

- **Baum-Welch is a hill climb.** It finds a *local* maximum of the
  likelihood, not necessarily the global one. Adding more training
  data does not help with a bad initialisation — it only pins down the
  local maximum more precisely.
- **Exactly-uniform is a fixed point.** If every row of `TRANS`,
  `EMIS`, `INIT` is exactly uniform, every EM update leaves them
  unchanged (Stamp §4.3). You need *some* asymmetry in the initial
  guess to get the algorithm started.
- **Prior knowledge of the topology helps a lot.** If you know, say,
  that your model is left-right (state `i` can only transition to
  states `≥ i`), you can zero-out the impossible transitions in the
  initial guess. The EM update multiplies by these zeros, so they stay
  zero for every iteration — the algorithm will only ever explore
  left-right models. This is not cheating; it is how you incorporate
  the structural prior you actually have.

The demo's initial guess is the one from the original 2012 code. It
does all three of the above: it bakes in the left-right topology via
zeros in `TRANS`, leaves `EMIS` uniform (so the algorithm has to learn
which state emits which symbol from scratch), and uses a slightly
asymmetric `INIT = [0.6, 0.2, 0.2]` to break the remaining state-label
symmetry.

```
Problem 3 (Baum-Welch re-estimation from 10000 sequences):

  Initial guess (2012 values):
    TRANS: 2/3 1/3  0    EMIS: 0.25 0.25 0.25 0.25    INIT: 0.6 0.2 0.2
            0  2/3 1/3         0.25 0.25 0.25 0.25
            0   0   1          0.25 0.25 0.25 0.25

  Learned model:
    TRANS: 0.502 0.498 ~0      EMIS: 0.506 0.494 ~0    ~0       INIT: 1 ~0 ~0
           ~0    0.702 0.298         ~0    0.495 0.505 ~0
           ~0    ~0    1             ~0    ~0    0.499 0.501

  Best state relabeling: true state i -> learned state [0 1 2]
  max |TRANS_learned - TRANS_true| = 0.0019
  max |EMIS_learned  - EMIS_true | = 0.0057
  total log-likelihood  under truth   = -161096
  total log-likelihood  under learned = -161093
```

The algorithm recovers all three matrices to within about half a
percent. Two things worth noticing:

- **The learned likelihood is slightly *above* the ground truth.** EM
  is maximum-likelihood estimation on the specific training sample,
  so it can fit the sample more tightly than the distribution that
  generated the sample would. On held-out data the ranking would flip.
- **No label switching.** The "best state relabeling" line shows
  `[0 1 2]`, i.e. the learned state 0 is what we called state 0, state
  1 is what we called state 1, and so on. That's because the
  asymmetric `INIT` pinned down which state is the "start state". If
  the initial guess had been perfectly symmetric, the learned states
  could equally have come out relabeled — Baum-Welch has no way to
  know which hidden state you call "state 0". The demo tries all `N!`
  relabelings of the learned states and reports the best match.

## Using the library from your own code

The whole API lives in `CvHMM.h`:

```cpp
#include "CvHMM.h"

// 1x100 observation sequence and its hidden states, as CV_32S.
cv::Mat seq, states;
CvHMM::generate(100, TRANS, EMIS, INIT, seq, states);

// Problem 1: log P(O | lambda) and posterior state probabilities.
double logP;
cv::Mat gamma, fwd, bwd;
CvHMM::decode(seq, TRANS, EMIS, INIT, logP, gamma, fwd, bwd);

// Problem 2: most-likely hidden-state path.
cv::Mat path;
CvHMM::viterbi(seq, TRANS, EMIS, INIT, path);

// Problem 3a: learn from a single sequence.
CvHMM::train(seq, /*maxIters=*/100, TRANS, EMIS, INIT);

// Problem 3b: learn from many independent sequences (K x T).
CvHMM::trainBatch(seqs, /*maxIters=*/100, TRANS, EMIS, INIT);
```

Shapes: `TRANS` is `N × N` (CV_64F), `EMIS` is `N × M` (CV_64F),
`INIT` is `1 × N` (CV_64F). Observation matrices are CV_32S with one
sequence per row.

There are also a few small helpers: `correctModel` (replace exact zeros
with a tiny epsilon and renormalize so the rows sum to 1),
`getUniformModel`, `printModel`, and a multi-sequence overload of
`generate`.

## When to use `train` vs `trainBatch`

- `train` is for the case where you have a **single long sequence** and
  every state gets revisited many times (the chain is ergodic).
- `trainBatch` is for **many independent sequences**. Each sequence can
  be short. This is the right choice when your model has an absorbing
  state or a long transient, because a single realisation won't contain
  enough information about the states that are visited rarely.

The demo uses `trainBatch` because the left-right model in `main.cpp`
has an absorbing state (state 2), so every single sequence is mostly
"stuck at state 2" and a single realisation is not enough to recover
the transitions out of states 0 and 1.

## Build and run

Only dependency is OpenCV (any recent 4.x).

```bash
g++ -std=c++11 -O2 -Wall -Wextra main.cpp -o demo $(pkg-config --cflags --libs opencv4)
./demo
```

## Regression tests

`test_hmm.py` is a Python re-implementation of everything in `CvHMM.h`,
plus brute-force baselines. It checks four things:

1. The scaled forward-backward's log `P(O)` matches a brute-force
   forward pass to machine precision.
2. Viterbi matches brute-force enumeration over all state paths
   (for small `T`).
3. Single-sequence Baum-Welch's log-likelihood is monotonic
   non-decreasing across iterations (the defining property of EM).
4. Multi-sequence Baum-Welch's *total* log-likelihood is monotonic
   non-decreasing, and on a reasonable dataset it recovers the
   underlying model parameters.

Run with [uv](https://docs.astral.sh/uv/):

```bash
uv run --with numpy python test_hmm.py
```

## Further reading

If you want to understand what's happening under the hood, the single
best introduction is

> Mark Stamp, *A Revealing Introduction to Hidden Markov Models*,
> <https://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf>

This library is a direct transcription of the pseudo-code in Stamp's
Section 5 (Viterbi, underflow-resistant dynamic programming) and
Section 7 (scaled forward-backward and Baum-Welch). The multi-sequence
`trainBatch` is the standard pooling extension you'll find in any
HMM textbook (e.g. Rabiner's 1989 tutorial).

## Notes and gotchas

- **Epsilon smoothing.** `correctModel` replaces exact zeros with
  `1e-30` and then renormalizes each row. This keeps `log(0)` out of
  the code. The sampler, the decoder and the trainer all go through
  `correctModel` so they see the same strictly-positive model.
- **Label switching.** Baum-Welch doesn't know which state you call
  "state 0" vs "state 1". If your initial guess is symmetric, the
  learned states may come out in a permuted order relative to the
  truth. Compare using a measure that is invariant to permutation if
  this matters.
- **Local optima.** Baum-Welch is a hill climb, so it only finds a
  local maximum of the likelihood. The demo avoids this issue by
  giving the algorithm an initial guess that already encodes the
  left-right topology. If you don't have structural prior knowledge
  about your model, the usual workaround is to run Baum-Welch from
  several different random initial guesses and keep the one with the
  highest log-likelihood (e.g. `hmmlearn`'s `n_init`).
- **Discrete observations only.** For continuous-output or
  Gaussian-mixture HMMs, look at
  [`hmmlearn`](https://hmmlearn.readthedocs.io/) or
  [`mlpack`](https://www.mlpack.org/).

## History

The original 2012 implementation is preserved in the git history. The
2026 rewrite keeps the `cv::Mat` matrix representation and a similar
public API, replaces the single-sequence internals with a line-by-line
transcription of Stamp's pseudo-code, adds a clean multi-sequence
`trainBatch`, and ships the `test_hmm.py` regression harness.

## License

BSD-3-Clause. See `LICENSE`.
