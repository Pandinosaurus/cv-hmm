/*
 *      CvHMM -- Discrete Hidden Markov Models on top of OpenCV.
 *
 *      A Hidden Markov Model describes a chain of hidden states that you
 *      cannot observe directly. What you *can* see is, at every time
 *      step, one noisy symbol emitted by the current hidden state. Given
 *      such a model and a sequence of observations, there are three
 *      classical questions:
 *
 *        1. How likely is this observation sequence?       -> decode()
 *        2. Which hidden states most plausibly produced it? -> viterbi()
 *        3. What model best explains a set of observations? -> train()
 *                                                             trainBatch()
 *
 *      The three hidden-layer matrices are:
 *
 *        TRANS[i, j]  probability of moving from state i to state j
 *        EMIS[i, k]   probability of emitting symbol k while in state i
 *        INIT[i]      probability of starting in state i
 *
 *      All arithmetic runs in scaled or log-space to stay numerically
 *      stable even for long sequences.
 *
 *      The single-sequence algorithms (viterbi, decode, train) are a
 *      line-by-line transcription of the pseudo-code in
 *          Mark Stamp, "A Revealing Introduction to Hidden Markov Models",
 *          https://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf
 *      specifically Section 5 (underflow-resistant Viterbi) and Section 7
 *      (scaled forward-backward and Baum-Welch re-estimation).
 *
 *      The multi-sequence trainer (trainBatch) is the standard pooling
 *      extension you'll find in any HMM textbook (Rabiner, 1989). It is
 *      NOT part of Stamp's tutorial.
 *
 * Copyright (c) 2012 Omid B. Sakhi
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef CVHMM_H
#define CVHMM_H

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>

class CvHMM {
public:
    CvHMM() {}

    /* =================================================================
     * correctModel -- keep zeros out of the model.
     *
     * Walks TRANS, EMIS, INIT, replaces any exact zero with a tiny
     * epsilon (1e-30), then divides each row by its sum so the rows
     * again sum to 1.
     *
     * Why: Viterbi works in log-space, so log(0) = -inf would poison
     * the arithmetic. The scaled forward-backward and Baum-Welch are
     * also happier with strictly positive probabilities.
     * ================================================================= */
    static void correctModel(cv::Mat &TRANS, cv::Mat &EMIS, cv::Mat &INIT)
    {
        const double eps = 1e-30;
        for (int i = 0; i < TRANS.rows; i++)
            for (int j = 0; j < TRANS.cols; j++)
                if (TRANS.at<double>(i, j) == 0) TRANS.at<double>(i, j) = eps;
        for (int i = 0; i < EMIS.rows; i++)
            for (int j = 0; j < EMIS.cols; j++)
                if (EMIS.at<double>(i, j) == 0) EMIS.at<double>(i, j) = eps;
        for (int j = 0; j < INIT.cols; j++)
            if (INIT.at<double>(0, j) == 0) INIT.at<double>(0, j) = eps;

        for (int i = 0; i < TRANS.rows; i++) {
            double s = 0;
            for (int j = 0; j < TRANS.cols; j++) s += TRANS.at<double>(i, j);
            for (int j = 0; j < TRANS.cols; j++) TRANS.at<double>(i, j) /= s;
        }
        for (int i = 0; i < EMIS.rows; i++) {
            double s = 0;
            for (int j = 0; j < EMIS.cols; j++) s += EMIS.at<double>(i, j);
            for (int j = 0; j < EMIS.cols; j++) EMIS.at<double>(i, j) /= s;
        }
        double s = 0;
        for (int j = 0; j < INIT.cols; j++) s += INIT.at<double>(0, j);
        for (int j = 0; j < INIT.cols; j++) INIT.at<double>(0, j) /= s;
    }

    // Build a "don't know anything" initial guess: N hidden states, M
    // observation symbols, every probability set to 1/N or 1/M.
    static void getUniformModel(int N, int M, cv::Mat &TRANS, cv::Mat &EMIS, cv::Mat &INIT)
    {
        TRANS = cv::Mat(N, N, CV_64F, cv::Scalar(1.0 / N)).clone();
        EMIS  = cv::Mat(N, M, CV_64F, cv::Scalar(1.0 / M)).clone();
        INIT  = cv::Mat(1, N, CV_64F, cv::Scalar(1.0 / N)).clone();
    }

    /* =================================================================
     * Sampling helpers -- draw synthetic observation sequences from a
     * known model. Useful for sanity-checking the inference routines on
     * a problem whose ground truth you already know.
     *
     * Randomness comes from std::rand(), so call std::srand(...) in the
     * caller if you want reproducible runs.
     * ================================================================= */

    // Roll the dice according to row `r` of `probs` (treated as a
    // discrete distribution over columns) and return the chosen column.
    static int sampleRow(const cv::Mat &probs, int r)
    {
        const double x = (double)std::rand() / (double)RAND_MAX;
        double acc = 0;
        for (int c = 0; c < probs.cols; c++) {
            acc += probs.at<double>(r, c);
            if (x <= acc) return c;
        }
        return probs.cols - 1;
    }

    // Generate a single length-T observation sequence and its hidden states.
    static void generate(int T,
                         const cv::Mat &TRANS, const cv::Mat &EMIS, const cv::Mat &INIT,
                         cv::Mat &seq, cv::Mat &states)
    {
        seq    = cv::Mat(1, T, CV_32S);
        states = cv::Mat(1, T, CV_32S);

        int s = sampleRow(INIT, 0);
        states.at<int>(0, 0) = s;
        seq   .at<int>(0, 0) = sampleRow(EMIS, s);
        for (int t = 1; t < T; t++) {
            s = sampleRow(TRANS, s);
            states.at<int>(0, t) = s;
            seq   .at<int>(0, t) = sampleRow(EMIS, s);
        }
    }

    // Convenience: generate `numSeq` independent sequences of length T,
    // one per row of `seq` / `states`.
    static void generate(int T, int numSeq,
                         const cv::Mat &TRANS, const cv::Mat &EMIS, const cv::Mat &INIT,
                         cv::Mat &seq, cv::Mat &states)
    {
        seq    = cv::Mat(numSeq, T, CV_32S);
        states = cv::Mat(numSeq, T, CV_32S);
        for (int r = 0; r < numSeq; r++) {
            cv::Mat srow, strow;
            generate(T, TRANS, EMIS, INIT, srow, strow);
            for (int c = 0; c < T; c++) {
                seq   .at<int>(r, c) = srow .at<int>(0, c);
                states.at<int>(r, c) = strow.at<int>(0, c);
            }
        }
    }

    /* =================================================================
     * viterbi -- Given a model and an observation sequence, find the
     * single most likely sequence of hidden states.
     *
     * This is the classic dynamic-programming algorithm: at every time
     * step t and every possible state i we remember
     *
     *   delta_t(i) = log-probability of the best state path that ends
     *                in state i at time t
     *   psi_t(i)   = which state at time t-1 gave us that best path
     *
     * Once we've filled in delta for all t, we pick the best final
     * state and follow the back-pointers in psi backwards to recover
     * the whole path.
     *
     * Working in log-space keeps long sequences from underflowing to 0.
     * See Stamp, Section 5, for the pseudo-code this mirrors.
     *
     * Input : seq is a 1 x T matrix of int observation indices (CV_32S).
     * Output: states is a 1 x T matrix of int state indices (CV_32S).
     * ================================================================= */
    static void viterbi(const cv::Mat &seq,
                        const cv::Mat &_TRANS, const cv::Mat &_EMIS, const cv::Mat &_INIT,
                        cv::Mat &states)
    {
        cv::Mat TRANS = _TRANS.clone();
        cv::Mat EMIS  = _EMIS .clone();
        cv::Mat INIT  = _INIT .clone();
        correctModel(TRANS, EMIS, INIT);

        const int T = seq.cols;
        const int N = TRANS.rows;

        cv::Mat delta(N, T, CV_64F);
        cv::Mat psi  (N, T, CV_32S, cv::Scalar(0));

        for (int i = 0; i < N; i++)
            delta.at<double>(i, 0) = std::log(INIT.at<double>(0, i))
                                   + std::log(EMIS.at<double>(i, seq.at<int>(0, 0)));

        for (int t = 1; t < T; t++) {
            for (int i = 0; i < N; i++) {
                const double logB = std::log(EMIS.at<double>(i, seq.at<int>(0, t)));
                double best = -DBL_MAX;
                int    arg  = 0;
                for (int j = 0; j < N; j++) {
                    const double v = delta.at<double>(j, t - 1)
                                   + std::log(TRANS.at<double>(j, i))
                                   + logB;
                    if (v > best) { best = v; arg = j; }
                }
                delta.at<double>(i, t) = best;
                psi  .at<int>   (i, t) = arg;
            }
        }

        double best = -DBL_MAX;
        int    arg  = 0;
        for (int i = 0; i < N; i++)
            if (delta.at<double>(i, T - 1) > best) {
                best = delta.at<double>(i, T - 1);
                arg  = i;
            }

        states = cv::Mat(1, T, CV_32S);
        states.at<int>(0, T - 1) = arg;
        for (int t = T - 2; t >= 0; t--)
            states.at<int>(0, t) = psi.at<int>(states.at<int>(0, t + 1), t + 1);
    }

    /* =================================================================
     * decode -- Score a sequence and compute per-timestep state
     * probabilities using the forward-backward algorithm.
     *
     * This answers two of the most common HMM questions at once:
     *
     *   * "How likely is this observation sequence under the model?"
     *     We return log P(O | lambda) in `logpseq`. Using the log
     *     avoids underflow: probabilities can easily become 10^-300 for
     *     even modest T.
     *
     *   * "Given everything we observed, what's the probability that
     *     the hidden state at time t was state i?"
     *     We return these "posteriors" gamma_t(i) in `PSTATES[i, t]`.
     *
     * To get there we compute two tables:
     *
     *   FORWARD [i, t] = P(O_0..O_t, X_t = i | lambda)   (alpha_hat, scaled)
     *   BACKWARD[i, t] = P(O_{t+1}..O_{T-1} | X_t = i, lambda) (beta_hat, scaled)
     *
     * "Scaled" means we renormalise each column of FORWARD to sum to 1
     * and remember the scale factors c_t. log P(O|lambda) then pops out
     * as -sum_t log(c_t). See Stamp, Section 7.
     *
     * Input : seq is a 1 x T CV_32S observation sequence.
     * ================================================================= */
    static void decode(const cv::Mat &seq,
                       const cv::Mat &_TRANS, const cv::Mat &_EMIS, const cv::Mat &_INIT,
                       double &logpseq, cv::Mat &PSTATES,
                       cv::Mat &FORWARD, cv::Mat &BACKWARD)
    {
        cv::Mat TRANS = _TRANS.clone();
        cv::Mat EMIS  = _EMIS .clone();
        cv::Mat INIT  = _INIT .clone();
        correctModel(TRANS, EMIS, INIT);

        const int T = seq.cols;
        const int N = TRANS.rows;

        FORWARD  = cv::Mat(N, T, CV_64F);
        BACKWARD = cv::Mat(N, T, CV_64F);
        cv::Mat c(1, T, CV_64F, cv::Scalar(0));

        // Step 2: alpha-pass (scaled).
        for (int i = 0; i < N; i++) {
            FORWARD.at<double>(i, 0) =
                INIT.at<double>(0, i) * EMIS.at<double>(i, seq.at<int>(0, 0));
            c.at<double>(0, 0) += FORWARD.at<double>(i, 0);
        }
        c.at<double>(0, 0) = 1.0 / c.at<double>(0, 0);
        for (int i = 0; i < N; i++)
            FORWARD.at<double>(i, 0) *= c.at<double>(0, 0);

        for (int t = 1; t < T; t++) {
            c.at<double>(0, t) = 0;
            for (int i = 0; i < N; i++) {
                double sum = 0;
                for (int j = 0; j < N; j++)
                    sum += FORWARD.at<double>(j, t - 1) * TRANS.at<double>(j, i);
                FORWARD.at<double>(i, t) = sum * EMIS.at<double>(i, seq.at<int>(0, t));
                c.at<double>(0, t) += FORWARD.at<double>(i, t);
            }
            c.at<double>(0, t) = 1.0 / c.at<double>(0, t);
            for (int i = 0; i < N; i++)
                FORWARD.at<double>(i, t) *= c.at<double>(0, t);
        }

        // Step 3: beta-pass (scaled with same c_t as alpha).
        for (int i = 0; i < N; i++)
            BACKWARD.at<double>(i, T - 1) = c.at<double>(0, T - 1);
        for (int t = T - 2; t >= 0; t--) {
            for (int i = 0; i < N; i++) {
                double sum = 0;
                for (int j = 0; j < N; j++)
                    sum += TRANS.at<double>(i, j)
                         * EMIS.at<double>(j, seq.at<int>(0, t + 1))
                         * BACKWARD.at<double>(j, t + 1);
                BACKWARD.at<double>(i, t) = c.at<double>(0, t) * sum;
            }
        }

        // Step 4: gamma_t(i) for t = 0 .. T-2, plus the special case at T-1.
        PSTATES = cv::Mat(N, T, CV_64F, cv::Scalar(0));
        for (int t = 0; t < T - 1; t++) {
            for (int i = 0; i < N; i++) {
                double g = 0;
                for (int j = 0; j < N; j++)
                    g += FORWARD.at<double>(i, t) * TRANS.at<double>(i, j)
                       * EMIS.at<double>(j, seq.at<int>(0, t + 1))
                       * BACKWARD.at<double>(j, t + 1);
                PSTATES.at<double>(i, t) = g;
            }
        }
        for (int i = 0; i < N; i++)
            PSTATES.at<double>(i, T - 1) = FORWARD.at<double>(i, T - 1);

        // Step 6: log P(O | lambda) = -sum_t log(c_t).
        logpseq = 0;
        for (int t = 0; t < T; t++)
            logpseq += std::log(c.at<double>(0, t));
        logpseq = -logpseq;
    }

    /* =================================================================
     * train -- Learn the model parameters from a single observation
     * sequence using the Baum-Welch algorithm.
     *
     * Baum-Welch is Expectation-Maximisation applied to HMMs. Each
     * iteration does two things:
     *
     *   E-step: using the current guess of TRANS, EMIS, INIT, run
     *           forward-backward on the data to compute
     *             gamma_t(i)     = P(X_t = i | O, lambda)
     *             digamma_t(i,j) = P(X_t = i, X_{t+1} = j | O, lambda)
     *
     *   M-step: treat those expected counts as if they were real counts
     *           and re-estimate the model in closed form:
     *             pi_i   = gamma_0(i)
     *             a_{ij} = sum_t digamma_t(i, j) / sum_t gamma_t(i)
     *             b_i(k) = sum_{t: O_t = k} gamma_t(i) / sum_t gamma_t(i)
     *
     * The log-likelihood log P(O | lambda) is guaranteed to go up (or
     * stay the same) every iteration; we stop as soon as it plateaus,
     * or after `maxIters`.
     *
     * This is the full seven-step procedure from Stamp, Section 7.
     *
     * On entry TRANS, EMIS, INIT are the initial guess. On exit they
     * hold the re-estimated model.
     *
     * Note: single-sequence learning only works well when your single
     * sequence actually visits every state many times (an "ergodic"
     * chain). If your model has an absorbing state or a long transient,
     * use `trainBatch` on many short sequences instead.
     * ================================================================= */
    static void train(const cv::Mat &seq, int maxIters,
                      cv::Mat &TRANS, cv::Mat &EMIS, cv::Mat &INIT)
    {
        correctModel(TRANS, EMIS, INIT);

        const int T = seq.cols;
        const int N = TRANS.rows;
        const int M = EMIS.cols;

        cv::Mat a    (N, T, CV_64F);
        cv::Mat b    (N, T, CV_64F);
        cv::Mat c    (1, T, CV_64F);
        cv::Mat gamma(N, T, CV_64F);
        std::vector<cv::Mat> digamma(T);  // digamma[t] is N x N, used for t = 0..T-2

        double oldLogProb = -DBL_MAX;
        int    iters      = 0;

        while (true) {
            // -------- Step 2: alpha-pass --------
            c.at<double>(0, 0) = 0;
            for (int i = 0; i < N; i++) {
                a.at<double>(i, 0) = INIT.at<double>(0, i)
                                   * EMIS.at<double>(i, seq.at<int>(0, 0));
                c.at<double>(0, 0) += a.at<double>(i, 0);
            }
            c.at<double>(0, 0) = 1.0 / c.at<double>(0, 0);
            for (int i = 0; i < N; i++)
                a.at<double>(i, 0) *= c.at<double>(0, 0);

            for (int t = 1; t < T; t++) {
                c.at<double>(0, t) = 0;
                for (int i = 0; i < N; i++) {
                    double sum = 0;
                    for (int j = 0; j < N; j++)
                        sum += a.at<double>(j, t - 1) * TRANS.at<double>(j, i);
                    a.at<double>(i, t) = sum * EMIS.at<double>(i, seq.at<int>(0, t));
                    c.at<double>(0, t) += a.at<double>(i, t);
                }
                c.at<double>(0, t) = 1.0 / c.at<double>(0, t);
                for (int i = 0; i < N; i++)
                    a.at<double>(i, t) *= c.at<double>(0, t);
            }

            // -------- Step 3: beta-pass --------
            for (int i = 0; i < N; i++)
                b.at<double>(i, T - 1) = c.at<double>(0, T - 1);
            for (int t = T - 2; t >= 0; t--) {
                for (int i = 0; i < N; i++) {
                    double sum = 0;
                    for (int j = 0; j < N; j++)
                        sum += TRANS.at<double>(i, j)
                             * EMIS.at<double>(j, seq.at<int>(0, t + 1))
                             * b.at<double>(j, t + 1);
                    b.at<double>(i, t) = c.at<double>(0, t) * sum;
                }
            }

            // -------- Step 4: gamma_t(i) and gamma_t(i, j) --------
            for (int t = 0; t < T - 1; t++) {
                digamma[t] = cv::Mat(N, N, CV_64F);
                for (int i = 0; i < N; i++) {
                    double g = 0;
                    for (int j = 0; j < N; j++) {
                        const double v = a.at<double>(i, t) * TRANS.at<double>(i, j)
                                       * EMIS.at<double>(j, seq.at<int>(0, t + 1))
                                       * b.at<double>(j, t + 1);
                        digamma[t].at<double>(i, j) = v;
                        g += v;
                    }
                    gamma.at<double>(i, t) = g;
                }
            }
            for (int i = 0; i < N; i++)
                gamma.at<double>(i, T - 1) = a.at<double>(i, T - 1);

            // -------- Step 5: re-estimate pi, A, B --------
            for (int i = 0; i < N; i++)
                INIT.at<double>(0, i) = gamma.at<double>(i, 0);

            for (int i = 0; i < N; i++) {
                double denom = 0;
                for (int t = 0; t < T - 1; t++) denom += gamma.at<double>(i, t);
                for (int j = 0; j < N; j++) {
                    double numer = 0;
                    for (int t = 0; t < T - 1; t++) numer += digamma[t].at<double>(i, j);
                    TRANS.at<double>(i, j) = numer / denom;
                }
            }

            for (int i = 0; i < N; i++) {
                double denom = 0;
                for (int t = 0; t < T; t++) denom += gamma.at<double>(i, t);
                for (int j = 0; j < M; j++) {
                    double numer = 0;
                    for (int t = 0; t < T; t++)
                        if (seq.at<int>(0, t) == j) numer += gamma.at<double>(i, t);
                    EMIS.at<double>(i, j) = numer / denom;
                }
            }

            correctModel(TRANS, EMIS, INIT);

            // -------- Step 6: log P(O | lambda) --------
            double logProb = 0;
            for (int t = 0; t < T; t++)
                logProb += std::log(c.at<double>(0, t));
            logProb = -logProb;

            // -------- Step 7: iterate or not --------
            iters++;
            if (iters >= maxIters || logProb <= oldLogProb) break;
            oldLogProb = logProb;
        }
    }

    /* =================================================================
     * trainBatch -- Learn the model from K independent observation
     * sequences at once.
     *
     * Same idea as `train`: run forward-backward on the data, turn the
     * resulting posterior counts into a new model, repeat. The only
     * difference is that the expected counts are *pooled* across all K
     * sequences before the M-step:
     *
     *   pi_i   = average of gamma^(k)_0(i) over k = 1..K
     *   a_{ij} = (sum_k sum_t digamma^(k)_t(i,j)) / (sum_k sum_t gamma^(k)_t(i))
     *   b_i(m) = (sum_k sum_{t: O^(k)_t=m} gamma^(k)_t(i)) / (sum_k sum_t gamma^(k)_t(i))
     *
     * Use this when you have many short independent sequences rather
     * than one long one. It's the only way to learn a model whose
     * chain isn't ergodic (e.g. a left-right model with an absorbing
     * state): any single realisation will quickly be swallowed by the
     * absorbing state and won't contain enough information to pin down
     * the transitions out of the transient states.
     *
     * Convergence: the *total* log-likelihood sum_k log P(O^(k) | lambda)
     * is monotonic non-decreasing; we stop when it plateaus or after
     * `maxIters` iterations.
     *
     * This is the standard multi-sequence extension (Rabiner, 1989).
     * It is NOT in Stamp's tutorial.
     *
     *   seqs    K x T CV_32S (each row is one sequence)
     *   TRANS, EMIS, INIT : initial guess in, re-estimated out.
     * ================================================================= */
    static void trainBatch(const cv::Mat &seqs, int maxIters,
                           cv::Mat &TRANS, cv::Mat &EMIS, cv::Mat &INIT)
    {
        correctModel(TRANS, EMIS, INIT);

        const int K = seqs.rows;
        const int T = seqs.cols;
        const int N = TRANS.rows;
        const int M = EMIS.cols;

        cv::Mat a    (N, T, CV_64F);
        cv::Mat b    (N, T, CV_64F);
        cv::Mat c    (1, T, CV_64F);
        cv::Mat gamma(N, T, CV_64F);
        std::vector<cv::Mat> digamma(T);
        for (int t = 0; t < T; t++)
            digamma[t] = cv::Mat(N, N, CV_64F);

        double oldLogProb = -DBL_MAX;
        int    iters      = 0;

        while (true) {
            cv::Mat piAcc = cv::Mat::zeros(1, N, CV_64F);
            cv::Mat aNum  = cv::Mat::zeros(N, N, CV_64F);
            cv::Mat aDen  = cv::Mat::zeros(1, N, CV_64F);
            cv::Mat bNum  = cv::Mat::zeros(N, M, CV_64F);
            cv::Mat bDen  = cv::Mat::zeros(1, N, CV_64F);
            double totalLogProb = 0;

            for (int k = 0; k < K; k++) {
                // -- alpha-pass on sequence k --
                c.at<double>(0, 0) = 0;
                for (int i = 0; i < N; i++) {
                    a.at<double>(i, 0) = INIT.at<double>(0, i)
                                       * EMIS.at<double>(i, seqs.at<int>(k, 0));
                    c.at<double>(0, 0) += a.at<double>(i, 0);
                }
                c.at<double>(0, 0) = 1.0 / c.at<double>(0, 0);
                for (int i = 0; i < N; i++)
                    a.at<double>(i, 0) *= c.at<double>(0, 0);

                for (int t = 1; t < T; t++) {
                    c.at<double>(0, t) = 0;
                    for (int i = 0; i < N; i++) {
                        double sum = 0;
                        for (int j = 0; j < N; j++)
                            sum += a.at<double>(j, t - 1) * TRANS.at<double>(j, i);
                        a.at<double>(i, t) = sum * EMIS.at<double>(i, seqs.at<int>(k, t));
                        c.at<double>(0, t) += a.at<double>(i, t);
                    }
                    c.at<double>(0, t) = 1.0 / c.at<double>(0, t);
                    for (int i = 0; i < N; i++)
                        a.at<double>(i, t) *= c.at<double>(0, t);
                }

                // -- beta-pass on sequence k --
                for (int i = 0; i < N; i++)
                    b.at<double>(i, T - 1) = c.at<double>(0, T - 1);
                for (int t = T - 2; t >= 0; t--) {
                    for (int i = 0; i < N; i++) {
                        double sum = 0;
                        for (int j = 0; j < N; j++)
                            sum += TRANS.at<double>(i, j)
                                 * EMIS.at<double>(j, seqs.at<int>(k, t + 1))
                                 * b.at<double>(j, t + 1);
                        b.at<double>(i, t) = c.at<double>(0, t) * sum;
                    }
                }

                // -- gamma and digamma on sequence k --
                for (int t = 0; t < T - 1; t++) {
                    for (int i = 0; i < N; i++) {
                        double g = 0;
                        for (int j = 0; j < N; j++) {
                            const double v = a.at<double>(i, t) * TRANS.at<double>(i, j)
                                           * EMIS.at<double>(j, seqs.at<int>(k, t + 1))
                                           * b.at<double>(j, t + 1);
                            digamma[t].at<double>(i, j) = v;
                            g += v;
                        }
                        gamma.at<double>(i, t) = g;
                    }
                }
                for (int i = 0; i < N; i++)
                    gamma.at<double>(i, T - 1) = a.at<double>(i, T - 1);

                // -- Accumulate expected counts from sequence k --
                for (int i = 0; i < N; i++)
                    piAcc.at<double>(0, i) += gamma.at<double>(i, 0);

                for (int i = 0; i < N; i++) {
                    double sumGammaA = 0;  // sum_{t=0..T-2} gamma_t(i)
                    for (int t = 0; t < T - 1; t++) sumGammaA += gamma.at<double>(i, t);
                    aDen.at<double>(0, i) += sumGammaA;
                    for (int j = 0; j < N; j++) {
                        double sumDigamma = 0;
                        for (int t = 0; t < T - 1; t++)
                            sumDigamma += digamma[t].at<double>(i, j);
                        aNum.at<double>(i, j) += sumDigamma;
                    }
                    double sumGammaB = 0;  // sum_{t=0..T-1} gamma_t(i)
                    for (int t = 0; t < T; t++) sumGammaB += gamma.at<double>(i, t);
                    bDen.at<double>(0, i) += sumGammaB;
                    for (int m = 0; m < M; m++) {
                        double sumMasked = 0;
                        for (int t = 0; t < T; t++)
                            if (seqs.at<int>(k, t) == m)
                                sumMasked += gamma.at<double>(i, t);
                        bNum.at<double>(i, m) += sumMasked;
                    }
                }

                // log P(O^(k) | lambda) = -sum_t log(c_t)
                double logPk = 0;
                for (int t = 0; t < T; t++) logPk += std::log(c.at<double>(0, t));
                totalLogProb += -logPk;
            }

            // -- M-step: re-estimate lambda from pooled expected counts --
            for (int i = 0; i < N; i++)
                INIT.at<double>(0, i) = piAcc.at<double>(0, i) / (double)K;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    TRANS.at<double>(i, j) = aNum.at<double>(i, j) / aDen.at<double>(0, i);
            for (int i = 0; i < N; i++)
                for (int m = 0; m < M; m++)
                    EMIS.at<double>(i, m) = bNum.at<double>(i, m) / bDen.at<double>(0, i);

            correctModel(TRANS, EMIS, INIT);

            iters++;
            if (iters >= maxIters || totalLogProb <= oldLogProb) break;
            oldLogProb = totalLogProb;
        }
    }

    // Dump TRANS, EMIS, INIT to std::cout in a human-readable form.
    /* ================================================================= */
    static void printModel(const cv::Mat &TRANS, const cv::Mat &EMIS, const cv::Mat &INIT)
    {
        std::cout << "\nTRANS:\n";
        for (int r = 0; r < TRANS.rows; r++) {
            for (int c = 0; c < TRANS.cols; c++)
                std::cout << TRANS.at<double>(r, c) << " ";
            std::cout << "\n";
        }
        std::cout << "\nEMIS:\n";
        for (int r = 0; r < EMIS.rows; r++) {
            for (int c = 0; c < EMIS.cols; c++)
                std::cout << EMIS.at<double>(r, c) << " ";
            std::cout << "\n";
        }
        std::cout << "\nINIT:\n";
        for (int r = 0; r < INIT.rows; r++) {
            for (int c = 0; c < INIT.cols; c++)
                std::cout << INIT.at<double>(r, c) << " ";
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};

#endif /* CVHMM_H */
