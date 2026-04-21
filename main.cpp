/*
 *      Demo program for CvHMM.
 *
 *      We define a small 3-state, 4-symbol HMM by hand, sample a bunch
 *      of observation sequences from it, and then use CvHMM to answer
 *      the three classical HMM questions on that data:
 *
 *        1. How likely is each sequence under the model?  (decode)
 *        2. Which hidden states most likely produced each  (viterbi)
 *           sequence?
 *        3. Starting from a rough initial guess, can we    (trainBatch)
 *           recover the original model from just the
 *           observations?
 *
 *      The model here is "left-right" -- state 2 is absorbing, so any
 *      single sequence spends most of its time stuck at state 2 and
 *      doesn't contain much information about the transitions out of
 *      states 0 and 1. For that reason we train on many independent
 *      sequences at once (trainBatch) rather than on a single long one.
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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <opencv2/core/core.hpp>
#include "CvHMM.h"

namespace {
constexpr int kNumSeq         = 10000; // K
constexpr int kSeqLen         = 20;    // T
constexpr int kPreview        = 5;
constexpr int kPreviewLen     = 20;
constexpr int kMaxTrainIters  = 200;
}  // namespace

int main()
{
    std::srand(0);  // deterministic demo

    // --- Ground-truth model (N = 3 states, M = 4 observation symbols) ---
    double TRANSdata[] = { 0.5, 0.5, 0.0,
                           0.0, 0.7, 0.3,
                           0.0, 0.0, 1.0 };
    double EMISdata[]  = { 0.5, 0.5, 0.0, 0.0,
                           0.0, 0.5, 0.5, 0.0,
                           0.0, 0.0, 0.5, 0.5 };
    double INITdata[]  = { 1.0, 0.0, 0.0 };

    cv::Mat TRANS = cv::Mat(3, 3, CV_64F, TRANSdata).clone();
    cv::Mat EMIS  = cv::Mat(3, 4, CV_64F, EMISdata ).clone();
    cv::Mat INIT  = cv::Mat(1, 3, CV_64F, INITdata ).clone();

    // Apply the same eps-smoothing CvHMM::decode / train / viterbi apply
    // internally, so sampling and scoring see the same strictly-positive
    // lambda.
    CvHMM::correctModel(TRANS, EMIS, INIT);

    std::cout << "--- Ground-truth model (after eps-smoothing) ---\n";
    CvHMM::printModel(TRANS, EMIS, INIT);

    // --- Sample K independent sequences of length T from the model ---
    cv::Mat seqs, trueStates;
    CvHMM::generate(kSeqLen, kNumSeq, TRANS, EMIS, INIT, seqs, trueStates);
    std::cout << "Sampled " << kNumSeq << " sequences of length "
              << kSeqLen << " from the model.\n"
              << "Preview (first " << kPreview << " sequences, first "
              << kPreviewLen << " symbols each):\n";
    for (int i = 0; i < kPreview; i++) {
        std::cout << "  O[" << i << "] = ";
        for (int t = 0; t < kPreviewLen; t++) std::cout << seqs.at<int>(i, t);
        std::cout << "\n";
        std::cout << "  X[" << i << "] = ";
        for (int t = 0; t < kPreviewLen; t++) std::cout << trueStates.at<int>(i, t);
        std::cout << "\n";
    }
    std::cout << "\n";

    // --- Problem 1: log P(O | lambda) per sequence ---
    std::cout << "Problem 1: How likely is each observation sequence\n"
                 "under the model?\n"
                 "  We compute log P(O | lambda) with the forward algorithm\n"
                 "  (in log-space so it doesn't underflow for long sequences).\n";
    double sumLogP = 0;
    {
        double logpseq;
        cv::Mat pstates, fwd, bwd;
        for (int i = 0; i < kNumSeq; i++) {
            CvHMM::decode(seqs.row(i), TRANS, EMIS, INIT,
                          logpseq, pstates, fwd, bwd);
            sumLogP += logpseq;
            if (i < kPreview)
                std::cout << "    log P(O[" << i << "] | lambda) = " << logpseq << "\n";
        }
    }
    const double meanLogP = sumLogP / kNumSeq;
    std::cout << "  mean over all " << kNumSeq
              << " sequences: " << meanLogP
              << "   total: " << sumLogP << "\n\n";

    // --- Problem 2: Viterbi per sequence ---
    std::cout << "Problem 2: Which hidden states most likely produced\n"
                 "each observation sequence?\n"
                 "  We find the single best state path with the Viterbi\n"
                 "  algorithm and compare it against the states we used\n"
                 "  when sampling the sequence.\n";
    long long totalMatches = 0;
    {
        cv::Mat vstates;
        for (int i = 0; i < kNumSeq; i++) {
            CvHMM::viterbi(seqs.row(i), TRANS, EMIS, INIT, vstates);
            for (int t = 0; t < kSeqLen; t++)
                if (vstates.at<int>(0, t) == trueStates.at<int>(i, t)) totalMatches++;
            if (i < kPreview) {
                std::cout << "    Viterbi[" << i << "] = ";
                for (int t = 0; t < kPreviewLen; t++) std::cout << vstates.at<int>(0, t);
                std::cout << "\n";
            }
        }
    }
    const long long totalPositions = (long long)kNumSeq * kSeqLen;
    std::cout << "  Viterbi matches the true (sampled) state at "
              << totalMatches << "/" << totalPositions
              << "  (" << (100.0 * (double)totalMatches / (double)totalPositions)
              << " %)\n\n";

    // --- Problem 3: multi-sequence Baum-Welch ---
    std::cout << "Problem 3: Starting from a rough initial guess, can\n"
                 "we recover the original model from just the observations?\n"
                 "  We re-estimate TRANS, EMIS and INIT from all " << kNumSeq << "\n"
                 "  sequences using the Baum-Welch algorithm (trainBatch).\n";

    // Initial guess: the original 2012 values. This bakes the
    // left-right topology into TRANS (zeros in the positions where
    // the true TRANS is zero -- correctModel then preserves them as
    // 1e-30, which stays effectively zero through every EM update),
    // leaves EMIS uniform, and uses an asymmetric INIT. That
    // asymmetry is important: exactly-uniform TRANS+EMIS+INIT is a
    // fixed point of Baum-Welch (Stamp Section 4.3) and EM would not
    // move. We keep this initial guess so the comparison against the
    // 2012 code is apples-to-apples.
    double TRGUESSdata[] = { 2.0/3, 1.0/3, 0.0/3,
                             0.0/3, 2.0/3, 1.0/3,
                             0.0/3, 0.0/3, 3.0/3 };
    double EMGUESSdata[] = { 0.25, 0.25, 0.25, 0.25,
                             0.25, 0.25, 0.25, 0.25,
                             0.25, 0.25, 0.25, 0.25 };
    double INGUESSdata[] = { 0.6, 0.2, 0.2 };

    cv::Mat TRGUESS = cv::Mat(3, 3, CV_64F, TRGUESSdata).clone();
    cv::Mat EMGUESS = cv::Mat(3, 4, CV_64F, EMGUESSdata).clone();
    cv::Mat INGUESS = cv::Mat(1, 3, CV_64F, INGUESSdata).clone();
    CvHMM::correctModel(TRGUESS, EMGUESS, INGUESS);

    std::cout << "\n  Initial guess (2012 values):";
    CvHMM::printModel(TRGUESS, EMGUESS, INGUESS);

    std::cout << "  Running trainBatch (max " << kMaxTrainIters
              << " iterations, stops early when total log-likelihood no longer increases)...\n";
    CvHMM::trainBatch(seqs, kMaxTrainIters, TRGUESS, EMGUESS, INGUESS);

    // Score the training set under the learned model.
    double bestTrainLL = 0;
    {
        double logpseq;
        cv::Mat pstates, fwd, bwd;
        for (int i = 0; i < kNumSeq; i++) {
            CvHMM::decode(seqs.row(i), TRGUESS, EMGUESS, INGUESS,
                          logpseq, pstates, fwd, bwd);
            bestTrainLL += logpseq;
        }
    }

    std::cout << "\n  Learned model:";
    CvHMM::printModel(TRGUESS, EMGUESS, INGUESS);

    // Baum-Welch doesn't know which hidden state you call "state 0",
    // so the learned states may come out relabeled. Try all N! = 6
    // permutations of the learned states and report the best match.
    int perm[3] = {0, 1, 2};
    double bestMaxTransErr = DBL_MAX;
    double bestMaxEmisErr  = 0;
    int    bestPerm[3]     = {0, 1, 2};
    do {
        double maxTransErr = 0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                maxTransErr = std::max(maxTransErr,
                    std::fabs(TRGUESS.at<double>(perm[i], perm[j]) - TRANS.at<double>(i, j)));
        double maxEmisErr = 0;
        for (int i = 0; i < 3; i++)
            for (int k = 0; k < 4; k++)
                maxEmisErr = std::max(maxEmisErr,
                    std::fabs(EMGUESS.at<double>(perm[i], k) - EMIS.at<double>(i, k)));
        const double score = std::max(maxTransErr, maxEmisErr);
        if (score < std::max(bestMaxTransErr, bestMaxEmisErr)) {
            bestMaxTransErr = maxTransErr;
            bestMaxEmisErr  = maxEmisErr;
            for (int i = 0; i < 3; i++) bestPerm[i] = perm[i];
        }
    } while (std::next_permutation(perm, perm + 3));

    std::cout << "  Best state relabeling: true state i -> learned state ["
              << bestPerm[0] << " " << bestPerm[1] << " " << bestPerm[2] << "]\n";
    std::cout << "  max |TRANS_learned - TRANS_true| (after relabeling) = " << bestMaxTransErr << "\n";
    std::cout << "  max |EMIS_learned  - EMIS_true | (after relabeling) = " << bestMaxEmisErr  << "\n";

    std::cout << "  total log-likelihood  under truth   = " << sumLogP     << "\n";
    std::cout << "  total log-likelihood  under learned = " << bestTrainLL << "\n";

    std::cout << "\ndone.\n";
    return 0;
}
