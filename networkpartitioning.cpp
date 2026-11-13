/* C++17: BYZANTINE RELIABLE BROADCAST IN PARTIALLY CONNECTED, LOSSY NETWORKS
 *
 * Publication-grade analytical and stochastic simulation suite comparing:
 *   1. Fault-agnostic topological augmentation via vertex-splitting Edmonds-Karp
 *      max-flow / min-vertex-cut (Menger's Theorem), enforcing kappa(G) >= 2f+1
 *      under Dolev's (1982) bound without any fault-oracle leakage.
 *   2. Mean-field analytical probability recurrence (S_MF) vs. exact Monte Carlo
 *      multi-hop 3-phase (SEND -> ECHO -> READY) authenticated broadcast (P_MC),
 *      quantifying the shared-path correlation optimism gap Delta = S_MF - P_MC.
 *   3. Adversarial evaluation under both passive omission and active Byzantine
 *      equivocation (conflicting digests), with explicit communication cost
 *      (messages, KB, signature verifications) and safety/liveness checks.
 *
 * Usage:
 *   Single-topology file mode:
 *     ./networkPartitioning <topology_input_file>
 *   Reproducible benchmark suite mode (outputs stdout summary + CSV):
 *     ./networkPartitioning --benchmark [output_csv_path]
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kDefaultReceptionThreshold = 0.999;
constexpr int kMaxSimulationRounds = 200;
constexpr std::size_t kPayloadBytes = 256;
constexpr std::size_t kSignatureBytes = 64;   // Ed25519 / ECDSA signature
constexpr std::size_t kHeaderBytes = 16;      // Sender ID, round, digest

using IntMatrix = std::vector<std::vector<int>>;
using DoubleMatrix = std::vector<std::vector<double>>;

enum class AdversaryMode {
    OMISSION,
    EQUIVOCATION
};

enum class RepairPolicy {
    UNREPAIRED,
    TWO_CONNECTED,
    DOLEV_2F_PLUS_1
};

struct VertexCutResult {
    int maxDisjointPaths = 0;
    std::vector<int> cutVertices;
    std::vector<int> sourceSideReachable;
    std::vector<int> sinkSideReachable;
};

struct SimulationConfig {
    int numNodes = 0;
    int numFaulty = 0;
    int broadcastSender = 0;
    double pLoss = 0.1;
    double threshold = kDefaultReceptionThreshold;
    int mcTrials = 400;
    uint64_t seed = 20261113ULL;
    AdversaryMode adversary = AdversaryMode::EQUIVOCATION;
};

struct BenchmarkMetrics {
    std::string topologyName;
    std::string policyName;
    std::string adversaryName;
    int N = 0;
    int f = 0;
    double pLoss = 0.0;
    int kappaBefore = 0;
    int kappaAfter = 0;
    int repairEdgesAdded = 0;
    int mfRounds = -1;
    double mcMeanRounds = 0.0;
    int mcP95Rounds = 0;
    double maxOptimismGap = 0.0;
    double meanOptimismGap = 0.0;
    double deliveryRate = 0.0;
    int safetyViolations = 0;
    double meanMessages = 0.0;
    double meanKilobytes = 0.0;
    double meanSigVerifications = 0.0;
};

// ============================================================================
// Matrix & Graph Utilities
// ============================================================================

void initWithZeros(DoubleMatrix& s, int n) {
    s.assign(n, std::vector<double>(n, 0.0));
}

void printMatrix(const DoubleMatrix& s, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setprecision(4) << s[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

bool correctNodesReceived(const DoubleMatrix& s,
                          int n,
                          const std::vector<bool>& isFaulty,
                          double threshold = kDefaultReceptionThreshold) {
    const int requiredQuorum = 2 * (n / 3) + 1;
    for (int i = 0; i < n; ++i) {
        if (!isFaulty[i]) {
            int countSigns = 0;
            for (int j = 0; j < n; ++j) {
                if (s[i][j] >= threshold) {
                    ++countSigns;
                }
            }
            if (countSigns < requiredQuorum) {
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// Fault-Agnostic Menger / Edmonds-Karp Vertex-Cut & (2f+1) Augmentation
// ============================================================================

// Computes the maximum number of internally vertex-disjoint paths between `src`
// and `dst` via vertex-splitting Edmonds-Karp max-flow (Menger's Theorem).
// Completely fault-agnostic: does NOT inspect `isFaulty`.
VertexCutResult computeVertexDisjointPathsAndCut(const IntMatrix& topo,
                                                 int src,
                                                 int dst,
                                                 int V) {
    const int numSplitNodes = 2 * V;
    const int kInf = V + 1;
    IntMatrix capacity(numSplitNodes, std::vector<int>(numSplitNodes, 0));
    std::vector<std::vector<int>> adj(numSplitNodes);

    auto addDirectedArc = [&](int u, int v, int cap) {
        capacity[u][v] = cap;
        adj[u].push_back(v);
        adj[v].push_back(u);
    };

    for (int v = 0; v < V; ++v) {
        const int vIn = 2 * v;
        const int vOut = 2 * v + 1;
        const int internalCap = (v == src || v == dst) ? kInf : 1;
        addDirectedArc(vIn, vOut, internalCap);
    }

    for (int u = 0; u < V; ++u) {
        for (int v = 0; v < V; ++v) {
            if (u != v && topo[u][v] != 0) {
                const int uOut = 2 * u + 1;
                const int vIn = 2 * v;
                const int edgeCap = (u == src && v == dst) ? 1 : kInf;
                addDirectedArc(uOut, vIn, edgeCap);
            }
        }
    }

    const int sourceSplit = 2 * src + 1;  // src_out
    const int sinkSplit = 2 * dst;        // dst_in
    int maxFlow = 0;

    while (true) {
        std::vector<int> parent(numSplitNodes, -1);
        parent[sourceSplit] = sourceSplit;
        std::queue<int> bfsQueue;
        bfsQueue.push(sourceSplit);

        while (!bfsQueue.empty() && parent[sinkSplit] == -1) {
            const int u = bfsQueue.front();
            bfsQueue.pop();
            for (int next : adj[u]) {
                if (parent[next] == -1 && capacity[u][next] > 0) {
                    parent[next] = u;
                    bfsQueue.push(next);
                }
            }
        }

        if (parent[sinkSplit] == -1) {
            break;
        }

        int pathFlow = kInf;
        for (int cur = sinkSplit; cur != sourceSplit; cur = parent[cur]) {
            const int prev = parent[cur];
            pathFlow = std::min(pathFlow, capacity[prev][cur]);
        }

        for (int cur = sinkSplit; cur != sourceSplit; cur = parent[cur]) {
            const int prev = parent[cur];
            capacity[prev][cur] -= pathFlow;
            capacity[cur][prev] += pathFlow;
        }

        maxFlow += pathFlow;
    }

    std::vector<bool> reachableFromSource(numSplitNodes, false);
    std::queue<int> forwardQueue;
    reachableFromSource[sourceSplit] = true;
    forwardQueue.push(sourceSplit);

    while (!forwardQueue.empty()) {
        const int u = forwardQueue.front();
        forwardQueue.pop();
        for (int next : adj[u]) {
            if (!reachableFromSource[next] && capacity[u][next] > 0) {
                reachableFromSource[next] = true;
                forwardQueue.push(next);
            }
        }
    }

    std::vector<bool> canReachSink(numSplitNodes, false);
    std::queue<int> backwardQueue;
    canReachSink[sinkSplit] = true;
    backwardQueue.push(sinkSplit);

    while (!backwardQueue.empty()) {
        const int u = backwardQueue.front();
        backwardQueue.pop();
        for (int prev : adj[u]) {
            if (!canReachSink[prev] && capacity[prev][u] > 0) {
                canReachSink[prev] = true;
                backwardQueue.push(prev);
            }
        }
    }

    VertexCutResult result;
    result.maxDisjointPaths = maxFlow;

    for (int v = 0; v < V; ++v) {
        const int vIn = 2 * v;
        const int vOut = 2 * v + 1;
        if (v != src && v != dst && reachableFromSource[vIn] && !reachableFromSource[vOut]) {
            result.cutVertices.push_back(v);
        }
        if (reachableFromSource[vOut]) {
            result.sourceSideReachable.push_back(v);
        }
        if (canReachSink[vIn]) {
            result.sinkSideReachable.push_back(v);
        }
    }

    return result;
}

int computeGlobalVertexConnectivity(const IntMatrix& topo, int V) {
    if (V <= 1) {
        return 0;
    }
    int minKappa = V - 1;
    for (int s = 0; s < V; ++s) {
        for (int t = s + 1; t < V; ++t) {
            const VertexCutResult cut = computeVertexDisjointPathsAndCut(topo, s, t, V);
            minKappa = std::min(minKappa, cut.maxDisjointPaths);
            if (minKappa == 0) {
                return 0;
            }
        }
    }
    return minKappa;
}

// Fault-agnostic k-vertex-connectivity augmentation.
// Never reads `isFaulty`: selects minimum-degree non-adjacent pairs across the
// residual cut (R_s x R_t) until kappa(G) >= min(targetK, V - 1).
int enforceFaultAgnosticConnectivity(IntMatrix& topo, int V, int targetK) {
    if (V <= 1 || targetK <= 0) {
        return 0;
    }

    const int cappedTarget = std::min(targetK, V - 1);
    int addedEdges = 0;

    std::vector<int> degree(V, 0);
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (topo[i][j] != 0) {
                ++degree[i];
            }
        }
    }

    for (int s = 0; s < V; ++s) {
        for (int t = s + 1; t < V; ++t) {
            while (true) {
                const VertexCutResult cut =
                    computeVertexDisjointPathsAndCut(topo, s, t, V);
                if (cut.maxDisjointPaths >= cappedTarget) {
                    break;
                }

                int bestU = -1;
                int bestV = -1;
                int bestScore = std::numeric_limits<int>::max();

                for (int u : cut.sourceSideReachable) {
                    for (int v : cut.sinkSideReachable) {
                        if (u != v && topo[u][v] == 0) {
                            // Strictly fault-agnostic degree-balancing heuristic
                            const int score = degree[u] + degree[v];
                            if (score < bestScore) {
                                bestScore = score;
                                bestU = u;
                                bestV = v;
                            }
                        }
                    }
                }

                if (bestU == -1 || bestV == -1) {
                    for (int w = 0; w < V; ++w) {
                        if (w != s && w != t) {
                            if (topo[s][w] == 0) {
                                bestU = s;
                                bestV = w;
                                break;
                            }
                            if (topo[w][t] == 0) {
                                bestU = w;
                                bestV = t;
                                break;
                            }
                        }
                    }
                }

                if (bestU == -1 || bestV == -1) {
                    break;
                }

                topo[bestU][bestV] = 1;
                topo[bestV][bestU] = 1;
                ++degree[bestU];
                ++degree[bestV];
                ++addedEdges;
            }
        }
    }

    return addedEdges;
}

// ============================================================================
// Mean-Field Analytical Recurrence (networkPartitioning.cpp Formulation)
// ============================================================================

struct MeanFieldTrace {
    std::vector<DoubleMatrix> history;
    int convergedRound = -1;
};

MeanFieldTrace runMeanFieldRecurrence(const IntMatrix& topo,
                                      const std::vector<bool>& isFaulty,
                                      const SimulationConfig& cfg) {
    const int n = cfg.numNodes;
    MeanFieldTrace trace;

    DoubleMatrix s;
    initWithZeros(s, n);
    s[cfg.broadcastSender][cfg.broadcastSender] = 1.0;
    trace.history.push_back(s);

    if (correctNodesReceived(s, n, isFaulty, cfg.threshold)) {
        trace.convergedRound = 0;
        return trace;
    }

    std::vector<double> probaMissSignFrom(n, 1.0);
    for (int r = 1; r <= kMaxSimulationRounds; ++r) {
        DoubleMatrix tmp = s;
        double maxDelta = 0.0;

        for (int dst = 0; dst < n; ++dst) {
            double probaAllFails = 1.0;
            std::fill(probaMissSignFrom.begin(), probaMissSignFrom.end(), 1.0);

            for (int src = 0; src < n; ++src) {
                if (topo[src][dst] != 0 && !isFaulty[src]) {
                    probaAllFails *= ((1.0 - s[src][src]) + s[src][src] * cfg.pLoss);
                    for (int i = 0; i < n; ++i) {
                        probaMissSignFrom[i] *= ((1.0 - s[src][i]) + s[src][i] * cfg.pLoss);
                    }
                }
            }

            tmp[dst][dst] = s[dst][dst] + (1.0 - s[dst][dst]) * (1.0 - probaAllFails);
            maxDelta = std::max(maxDelta, std::abs(tmp[dst][dst] - s[dst][dst]));

            for (int i = 0; i < n; ++i) {
                if (i != dst) {
                    tmp[dst][i] = s[dst][i] + (1.0 - s[dst][i]) * (1.0 - probaMissSignFrom[i]);
                    maxDelta = std::max(maxDelta, std::abs(tmp[dst][i] - s[dst][i]));
                }
            }
        }

        s = std::move(tmp);
        trace.history.push_back(s);

        if (correctNodesReceived(s, n, isFaulty, cfg.threshold)) {
            trace.convergedRound = r;
            break;
        }
        if (maxDelta < 1e-15) {
            break;
        }
    }

    return trace;
}

// ============================================================================
// Exact Monte Carlo Multi-Hop 3-Phase Authenticated Broadcast (ECHO / READY)
// ============================================================================

struct MonteCarloSummary {
    std::vector<DoubleMatrix> empiricalProbs;  // P_MC[r][dst][i]
    double meanRounds = 0.0;
    int p95Rounds = 0;
    double deliveryRate = 0.0;
    int safetyViolations = 0;
    double meanMessages = 0.0;
    double meanKilobytes = 0.0;
    double meanSigVerifications = 0.0;
};

MonteCarloSummary runMonteCarloBrachaMultiHop(const IntMatrix& topo,
                                              const std::vector<bool>& isFaulty,
                                              const SimulationConfig& cfg,
                                              int maxTrackedRounds) {
    const int n = cfg.numNodes;
    const int f = cfg.numFaulty;
    const int echoQuorum = (n + f + 2) / 2; // Bracha Byzantine quorum ceil((N+f+1)/2)
    const int readyAmplify = f + 1;     // Bracha READY amplification threshold
    const int deliverQuorum = 2 * f + 1;// Bracha READY -> DELIVER threshold

    std::mt19937_64 rng(cfg.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    std::vector<std::vector<std::vector<int>>> roundCounts(
        maxTrackedRounds + 1,
        std::vector<std::vector<int>>(n, std::vector<int>(n, 0)));

    std::vector<int> completionRounds;
    completionRounds.reserve(cfg.mcTrials);

    int successfulDeliveries = 0;
    int safetyViolations = 0;
    double totalMessages = 0.0;
    double totalBytes = 0.0;
    double totalSigVerifications = 0.0;

    // Precompute adjacency lists for fast iteration
    std::vector<std::vector<int>> neighbors(n);
    for (int u = 0; u < n; ++u) {
        for (int v = 0; v < n; ++v) {
            if (u != v && topo[u][v] != 0) {
                neighbors[u].push_back(v);
            }
        }
    }

    for (int trial = 0; trial < cfg.mcTrials; ++trial) {
        // Per-node state for two candidate values: value 0 (honest) and value 1 (equivocating)
        // echoSigs[v][val][signer] = whether honest node v holds signer's ECHO sig for val
        // readySigs[v][val][signer] = whether honest node v holds signer's READY sig for val
        std::vector<std::vector<std::vector<uint8_t>>> echoSigs(
            n, std::vector<std::vector<uint8_t>>(2, std::vector<uint8_t>(n, 0)));
        std::vector<std::vector<std::vector<uint8_t>>> readySigs(
            n, std::vector<std::vector<uint8_t>>(2, std::vector<uint8_t>(n, 0)));

        std::vector<bool> sentEcho0(n, false), sentEcho1(n, false);
        std::vector<bool> sentReady0(n, false), sentReady1(n, false);
        std::vector<int> deliveredVal(n, -1);

        // Track if a node ever received the primary broadcast (value 0 or 1)
        std::vector<std::vector<uint8_t>> receivedAnySig(
            n, std::vector<uint8_t>(n, 0));

        const int bid = cfg.broadcastSender;
        receivedAnySig[bid][bid] = 1;

        if (!isFaulty[bid]) {
            // Honest broadcaster initiates value 0
            echoSigs[bid][0][bid] = 1;
            sentEcho0[bid] = true;
        } else if (cfg.adversary == AdversaryMode::EQUIVOCATION) {
            // Byzantine broadcaster injects conflicting value 0 to even neighbors, 1 to odd
            for (std::size_t idx = 0; idx < neighbors[bid].size(); ++idx) {
                const int nb = neighbors[bid][idx];
                if (!isFaulty[nb] && uni(rng) >= cfg.pLoss) {
                    const int val = static_cast<int>(idx % 2);
                    echoSigs[nb][val][bid] = 1;
                    receivedAnySig[nb][bid] = 1;
                    totalMessages += 1.0;
                    totalBytes += static_cast<double>(kHeaderBytes + kPayloadBytes + kSignatureBytes);
                    totalSigVerifications += 1.0;
                    if (val == 0 && !sentEcho0[nb] && !sentEcho1[nb]) {
                        sentEcho0[nb] = true;
                        echoSigs[nb][0][nb] = 1;
                        receivedAnySig[nb][nb] = 1;
                    } else if (val == 1 && !sentEcho0[nb] && !sentEcho1[nb]) {
                        sentEcho1[nb] = true;
                        echoSigs[nb][1][nb] = 1;
                        receivedAnySig[nb][nb] = 1;
                    }
                }
            }
        }

        // Record round 0
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                roundCounts[0][i][j] += receivedAnySig[i][j];
            }
        }

        int trialFinishRound = kMaxSimulationRounds;
        bool allHonestDelivered = false;

        for (int r = 1; r <= kMaxSimulationRounds; ++r) {
            auto nextEcho = echoSigs;
            auto nextReady = readySigs;
            auto nextAny = receivedAnySig;

            // Each honest node forwards newly held ECHO/READY signatures to neighbors
            // subject to link loss pLoss.
            for (int src = 0; src < n; ++src) {
                if (isFaulty[src]) {
                    // Active equivocation: faulty relays inject conflicting value 1
                    if (cfg.adversary == AdversaryMode::EQUIVOCATION) {
                        for (int dst : neighbors[src]) {
                            if (!isFaulty[dst] && uni(rng) >= cfg.pLoss) {
                                nextEcho[dst][1][src] = 1;
                                nextReady[dst][1][src] = 1;
                                totalMessages += 1.0;
                                totalBytes += static_cast<double>(kHeaderBytes + kSignatureBytes);
                            }
                        }
                    }
                    continue;
                }

                // Count how many signatures src holds to bundle into a gossip frame
                int sigsHeldBySrc = 0;
                for (int val = 0; val < 2; ++val) {
                    for (int k = 0; k < n; ++k) {
                        sigsHeldBySrc += echoSigs[src][val][k] + readySigs[src][val][k];
                    }
                }
                if (sigsHeldBySrc == 0) {
                    continue;
                }

                for (int dst : neighbors[src]) {
                    if (isFaulty[dst]) {
                        continue;
                    }

                    // Link transmission with packet loss pLoss
                    // Modeled per-link bundle: if link succeeds, dst learns src's signatures
                    if (uni(rng) >= cfg.pLoss) {
                        std::size_t newSigsTransferred = 0;
                        for (int val = 0; val < 2; ++val) {
                            for (int k = 0; k < n; ++k) {
                                if (echoSigs[src][val][k] && !nextEcho[dst][val][k]) {
                                    nextEcho[dst][val][k] = 1;
                                    nextAny[dst][k] = 1;
                                    ++newSigsTransferred;
                                    totalSigVerifications += 1.0;
                                }
                                if (readySigs[src][val][k] && !nextReady[dst][val][k]) {
                                    nextReady[dst][val][k] = 1;
                                    nextAny[dst][k] = 1;
                                    ++newSigsTransferred;
                                    totalSigVerifications += 1.0;
                                }
                            }
                        }
                        if (!nextAny[dst][dst] && nextAny[src][src]) {
                            nextAny[dst][dst] = 1;
                        }
                        totalMessages += 1.0;
                        totalBytes += static_cast<double>(
                            kHeaderBytes + kPayloadBytes +
                            std::max<std::size_t>(1, newSigsTransferred) * kSignatureBytes);
                    }
                }
            }

            echoSigs = std::move(nextEcho);
            readySigs = std::move(nextReady);
            receivedAnySig = std::move(nextAny);

            // State transition rules (Bracha 3-phase on accumulated signatures)
            int honestDeliveredCount = 0;
            int honestTotal = 0;
            for (int v = 0; v < n; ++v) {
                if (isFaulty[v]) {
                    continue;
                }
                ++honestTotal;

                // ECHO phase: echo first observed value
                if (!sentEcho0[v] && !sentEcho1[v]) {
                    int c0 = std::accumulate(echoSigs[v][0].begin(), echoSigs[v][0].end(), 0);
                    int c1 = std::accumulate(echoSigs[v][1].begin(), echoSigs[v][1].end(), 0);
                    if (c0 > 0) {
                        sentEcho0[v] = true;
                        echoSigs[v][0][v] = 1;
                        receivedAnySig[v][v] = 1;
                    } else if (c1 > 0) {
                        sentEcho1[v] = true;
                        echoSigs[v][1][v] = 1;
                        receivedAnySig[v][v] = 1;
                    }
                }

                // READY phase: 2f+1 ECHOs or f+1 READYs
                for (int val = 0; val < 2; ++val) {
                    const int eCount = std::accumulate(echoSigs[v][val].begin(), echoSigs[v][val].end(), 0);
                    const int rCount = std::accumulate(readySigs[v][val].begin(), readySigs[v][val].end(), 0);
                    if (eCount >= echoQuorum || rCount >= readyAmplify) {
                        if (val == 0 && !sentReady0[v] && !sentReady1[v]) {
                            sentReady0[v] = true;
                            readySigs[v][0][v] = 1;
                        } else if (val == 1 && !sentReady0[v] && !sentReady1[v]) {
                            sentReady1[v] = true;
                            readySigs[v][1][v] = 1;
                        }
                    }

                    // DELIVER phase: 2f+1 READYs
                    const int updatedReady = std::accumulate(readySigs[v][val].begin(), readySigs[v][val].end(), 0);
                    if (deliveredVal[v] == -1 && updatedReady >= deliverQuorum) {
                        deliveredVal[v] = val;
                    }
                }

                if (deliveredVal[v] != -1) {
                    ++honestDeliveredCount;
                }
            }

            if (r <= maxTrackedRounds) {
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        roundCounts[r][i][j] += receivedAnySig[i][j];
                    }
                }
            }

            if (honestDeliveredCount == honestTotal && honestTotal > 0) {
                allHonestDelivered = true;
                trialFinishRound = r;
                break;
            }
        }

        // Check safety (agreement): no two honest nodes delivered different values
        int firstDelivered = -1;
        bool agreementViolated = false;
        for (int v = 0; v < n; ++v) {
            if (!isFaulty[v] && deliveredVal[v] != -1) {
                if (firstDelivered == -1) {
                    firstDelivered = deliveredVal[v];
                } else if (deliveredVal[v] != firstDelivered) {
                    agreementViolated = true;
                }
            }
        }
        if (agreementViolated) {
            ++safetyViolations;
        }
        if (allHonestDelivered) {
            ++successfulDeliveries;
        }
        completionRounds.push_back(trialFinishRound);
    }

    MonteCarloSummary summary;
    summary.empiricalProbs.resize(
        maxTrackedRounds + 1, DoubleMatrix(n, std::vector<double>(n, 0.0)));
    for (int r = 0; r <= maxTrackedRounds; ++r) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                summary.empiricalProbs[r][i][j] =
                    static_cast<double>(roundCounts[r][i][j]) / static_cast<double>(cfg.mcTrials);
            }
        }
    }

    std::sort(completionRounds.begin(), completionRounds.end());
    const double sumRounds = std::accumulate(completionRounds.begin(), completionRounds.end(), 0.0);
    summary.meanRounds = sumRounds / static_cast<double>(cfg.mcTrials);
    const std::size_t p95Idx = static_cast<std::size_t>(0.95 * (cfg.mcTrials - 1));
    summary.p95Rounds = completionRounds[p95Idx];
    summary.deliveryRate = static_cast<double>(successfulDeliveries) / static_cast<double>(cfg.mcTrials);
    summary.safetyViolations = safetyViolations;
    summary.meanMessages = totalMessages / static_cast<double>(cfg.mcTrials);
    summary.meanKilobytes = (totalBytes / 1024.0) / static_cast<double>(cfg.mcTrials);
    summary.meanSigVerifications = totalSigVerifications / static_cast<double>(cfg.mcTrials);
    return summary;
}

// ============================================================================
// Synthetic Topology Generators (Erdos-Renyi, Barabasi-Albert, Grid Mesh)
// ============================================================================

IntMatrix generateErdosRenyi(int n, double edgeProb, uint64_t seed) {
    IntMatrix topo(n, std::vector<int>(n, 0));
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (uni(rng) < edgeProb) {
                topo[i][j] = 1;
                topo[j][i] = 1;
            }
        }
    }
    return topo;
}

IntMatrix generateBarabasiAlbert(int n, int mAttach, uint64_t seed) {
    IntMatrix topo(n, std::vector<int>(n, 0));
    const int m0 = std::min(n, std::max(2, mAttach + 1));
    std::vector<int> degreePool;

    for (int i = 0; i < m0; ++i) {
        for (int j = i + 1; j < m0; ++j) {
            topo[i][j] = 1;
            topo[j][i] = 1;
            degreePool.push_back(i);
            degreePool.push_back(j);
        }
    }

    std::mt19937_64 rng(seed);
    for (int newNode = m0; newNode < n; ++newNode) {
        std::set<int> targets;
        while (static_cast<int>(targets.size()) < std::min(mAttach, newNode)) {
            const std::size_t idx = rng() % degreePool.size();
            targets.insert(degreePool[idx]);
        }
        for (int t : targets) {
            topo[newNode][t] = 1;
            topo[t][newNode] = 1;
            degreePool.push_back(newNode);
            degreePool.push_back(t);
        }
    }
    return topo;
}

IntMatrix generateGridMesh(int n) {
    IntMatrix topo(n, std::vector<int>(n, 0));
    const int cols = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n))));
    for (int i = 0; i < n; ++i) {
        const int r = i / cols;
        const int c = i % cols;
        if (c + 1 < cols && i + 1 < n) {
            topo[i][i + 1] = 1;
            topo[i + 1][i] = 1;
        }
        if (i + cols < n) {
            topo[i][i + cols] = 1;
            topo[i + cols][i] = 1;
        }
        if (r == 0 && c == 0 && n > 2) {
            topo[0][1] = 1;
            topo[1][0] = 1;
        }
    }
    return topo;
}

// ============================================================================
// Publication Benchmark Driver
// ============================================================================

BenchmarkMetrics evaluateConfiguration(const std::string& topoName,
                                       const IntMatrix& baseTopo,
                                       RepairPolicy policy,
                                       const SimulationConfig& cfg,
                                       const std::vector<bool>& isFaulty) {
    IntMatrix topo = baseTopo;
    const int kappaBefore = computeGlobalVertexConnectivity(topo, cfg.numNodes);
    int repairEdges = 0;
    std::string policyName = "Unrepaired";

    if (policy == RepairPolicy::TWO_CONNECTED) {
        policyName = "2-Connected";
        repairEdges = enforceFaultAgnosticConnectivity(topo, cfg.numNodes, 2);
    } else if (policy == RepairPolicy::DOLEV_2F_PLUS_1) {
        policyName = "Dolev-(2f+1)";
        repairEdges = enforceFaultAgnosticConnectivity(topo, cfg.numNodes, 2 * cfg.numFaulty + 1);
    }

    const int kappaAfter = computeGlobalVertexConnectivity(topo, cfg.numNodes);
    const MeanFieldTrace mf = runMeanFieldRecurrence(topo, isFaulty, cfg);
    const int trackedRounds = std::min(40, static_cast<int>(mf.history.size()) - 1);
    const MonteCarloSummary mc = runMonteCarloBrachaMultiHop(topo, isFaulty, cfg, trackedRounds);

    double maxGap = 0.0;
    double gapSum = 0.0;
    int gapCount = 0;
    for (int r = 0; r <= trackedRounds; ++r) {
        for (int dst = 0; dst < cfg.numNodes; ++dst) {
            if (isFaulty[dst]) continue;
            for (int i = 0; i < cfg.numNodes; ++i) {
                if (isFaulty[i]) continue;
                const double diff = mf.history[r][dst][i] - mc.empiricalProbs[r][dst][i];
                maxGap = std::max(maxGap, diff);
                gapSum += diff;
                ++gapCount;
            }
        }
    }

    BenchmarkMetrics m;
    m.topologyName = topoName;
    m.policyName = policyName;
    m.adversaryName = (cfg.adversary == AdversaryMode::EQUIVOCATION) ? "Equivocation" : "Omission";
    m.N = cfg.numNodes;
    m.f = cfg.numFaulty;
    m.pLoss = cfg.pLoss;
    m.kappaBefore = kappaBefore;
    m.kappaAfter = kappaAfter;
    m.repairEdgesAdded = repairEdges;
    m.mfRounds = mf.convergedRound;
    m.mcMeanRounds = mc.meanRounds;
    m.mcP95Rounds = mc.p95Rounds;
    m.maxOptimismGap = maxGap;
    m.meanOptimismGap = (gapCount > 0) ? (gapSum / gapCount) : 0.0;
    m.deliveryRate = mc.deliveryRate;
    m.safetyViolations = mc.safetyViolations;
    m.meanMessages = mc.meanMessages;
    m.meanKilobytes = mc.meanKilobytes;
    m.meanSigVerifications = mc.meanSigVerifications;
    return m;
}

int runBenchmarkSuite(const std::string& csvPath) {
    struct TopoSpec {
        std::string name;
        int N;
        int f;
        IntMatrix mat;
    };

    std::vector<TopoSpec> specs;
    specs.push_back({"GridMesh-16", 16, 3, generateGridMesh(16)});
    specs.push_back({"ErdosRenyi-16", 16, 3, generateErdosRenyi(16, 0.28, 101ULL)});
    specs.push_back({"BarabasiAlbert-25", 25, 4, generateBarabasiAlbert(25, 2, 202ULL)});
    specs.push_back({"ErdosRenyi-31", 31, 4, generateErdosRenyi(31, 0.22, 303ULL)});

    const std::vector<double> lossRates = {0.0, 0.15, 0.30};
    const std::vector<RepairPolicy> policies = {
        RepairPolicy::UNREPAIRED,
        RepairPolicy::TWO_CONNECTED,
        RepairPolicy::DOLEV_2F_PLUS_1
    };
    const std::vector<AdversaryMode> adversaries = {
        AdversaryMode::OMISSION,
        AdversaryMode::EQUIVOCATION
    };

    std::ofstream csvOut(csvPath);
    if (!csvOut.is_open()) {
        std::cerr << "Error: Cannot write CSV output to '" << csvPath << "'.\n";
        return 1;
    }

    csvOut << "topology,N,f,policy,adversary,pLoss,kappa_before,kappa_after,repair_edges,"
           << "mf_rounds,mc_mean_rounds,mc_p95_rounds,max_optimism_gap,mean_optimism_gap,"
           << "delivery_rate,safety_violations,mean_messages,mean_kb,mean_sig_verifications\n";

    std::cout << std::left
              << std::setw(20) << "Topology"
              << std::setw(15) << "Policy"
              << std::setw(13) << "Adversary"
              << std::setw(6) << "pLoss"
              << std::setw(9) << "k_in->k"
              << std::setw(6) << "+Edge"
              << std::setw(7) << "MF_rd"
              << std::setw(8) << "MC_rd"
              << std::setw(9) << "MaxGap"
              << std::setw(8) << "Deliv%"
              << std::setw(7) << "SafeV"
              << std::setw(9) << "KB/run"
              << "\n";
    std::cout << std::string(116, '-') << "\n";

    for (const auto& spec : specs) {
        std::vector<bool> isFaulty(spec.N, false);
        // Place faulty nodes deterministically on highest-degree nodes (worst-case adversary)
        std::vector<std::pair<int, int>> degOrder;
        for (int v = 0; v < spec.N; ++v) {
            int d = 0;
            for (int u = 0; u < spec.N; ++u) {
                d += (spec.mat[v][u] != 0 ? 1 : 0);
            }
            degOrder.push_back({d, v});
        }
        std::sort(degOrder.rbegin(), degOrder.rend());

        int sender = degOrder.back().second;
        int assigned = 0;
        for (const auto& kv : degOrder) {
            if (kv.second != sender && assigned < spec.f) {
                isFaulty[kv.second] = true;
                ++assigned;
            }
        }

        for (double pLoss : lossRates) {
            for (RepairPolicy pol : policies) {
                for (AdversaryMode adv : adversaries) {
                    SimulationConfig cfg;
                    cfg.numNodes = spec.N;
                    cfg.numFaulty = spec.f;
                    cfg.broadcastSender = sender;
                    cfg.pLoss = pLoss;
                    cfg.mcTrials = 250;
                    cfg.adversary = adv;

                    const BenchmarkMetrics m =
                        evaluateConfiguration(spec.name, spec.mat, pol, cfg, isFaulty);

                    std::cout << std::left
                              << std::setw(20) << m.topologyName
                              << std::setw(15) << m.policyName
                              << std::setw(13) << m.adversaryName
                              << std::setw(6) << std::fixed << std::setprecision(2) << m.pLoss
                              << std::setw(9) << (std::to_string(m.kappaBefore) + "->" + std::to_string(m.kappaAfter))
                              << std::setw(6) << m.repairEdgesAdded
                              << std::setw(7) << m.mfRounds
                              << std::setw(8) << std::setprecision(2) << m.mcMeanRounds
                              << std::setw(9) << std::setprecision(3) << m.maxOptimismGap
                              << std::setw(8) << std::setprecision(1) << (m.deliveryRate * 100.0)
                              << std::setw(7) << m.safetyViolations
                              << std::setw(9) << std::setprecision(1) << m.meanKilobytes
                              << "\n";

                    csvOut << m.topologyName << "," << m.N << "," << m.f << ","
                           << m.policyName << "," << m.adversaryName << ","
                           << std::fixed << std::setprecision(4) << m.pLoss << ","
                           << m.kappaBefore << "," << m.kappaAfter << "," << m.repairEdgesAdded << ","
                           << m.mfRounds << "," << m.mcMeanRounds << "," << m.mcP95Rounds << ","
                           << m.maxOptimismGap << "," << m.meanOptimismGap << ","
                           << m.deliveryRate << "," << m.safetyViolations << ","
                           << m.meanMessages << "," << m.meanKilobytes << ","
                           << m.meanSigVerifications << "\n";
                }
            }
        }
    }

    csvOut.close();
    std::cout << "\nBenchmark CSV saved to: " << csvPath << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--benchmark") {
        const std::string csvPath = (argc >= 3) ? argv[2] : "benchmark_results.csv";
        return runBenchmarkSuite(csvPath);
    }

    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <topology_input_file>\n"
                  << "  " << argv[0] << " --benchmark [output_csv_path]\n";
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile.is_open()) {
        std::cerr << "Error: Unable to open input file '" << argv[1] << "'.\n";
        return 1;
    }

    int numNodes = 0;
    if (!(infile >> numNodes) || numNodes <= 0) {
        std::cerr << "Error: Invalid number of nodes.\n";
        return 1;
    }

    IntMatrix topo(numNodes, std::vector<int>(numNodes, 0));
    for (int i = 0; i < numNodes; ++i) {
        for (int j = 0; j < numNodes; ++j) {
            infile >> topo[i][j];
        }
    }

    int numFaulty = 0;
    if (!(infile >> numFaulty) || numFaulty < 0 || numFaulty > numNodes) {
        std::cerr << "Error: Invalid number of faulty nodes.\n";
        return 1;
    }

    std::vector<int> faulty(numFaulty, 0);
    std::vector<bool> isFaulty(numNodes, false);
    for (int i = 0; i < numFaulty; ++i) {
        infile >> faulty[i];
        if (faulty[i] < 0 || faulty[i] >= numNodes) {
            std::cerr << "Error: Faulty node ID out of range: " << faulty[i] << "\n";
            return 1;
        }
        isFaulty[faulty[i]] = true;
    }

    double pLoss = 0.0;
    int bid = 0;
    if (!(infile >> pLoss >> bid) || pLoss < 0.0 || pLoss >= 1.0 || bid < 0 || bid >= numNodes) {
        std::cerr << "Error: Invalid packet loss probability or broadcast sender ID.\n";
        return 1;
    }
    infile.close();

    // Fault-agnostic (2f+1)-connectivity repair (never reads `isFaulty`)
    const int kappaBefore = computeGlobalVertexConnectivity(topo, numNodes);
    const int addedEdges = enforceFaultAgnosticConnectivity(topo, numNodes, 2 * numFaulty + 1);
    const int kappaAfter = computeGlobalVertexConnectivity(topo, numNodes);

    std::cout << "Fault-agnostic Dolev (2f+1) augmentation: kappa "
              << kappaBefore << " -> " << kappaAfter
              << " (edges added: " << addedEdges << ")\n\n";

    SimulationConfig cfg;
    cfg.numNodes = numNodes;
    cfg.numFaulty = numFaulty;
    cfg.broadcastSender = bid;
    cfg.pLoss = pLoss;
    cfg.mcTrials = 500;
    cfg.adversary = AdversaryMode::EQUIVOCATION;

    const MeanFieldTrace mf = runMeanFieldRecurrence(topo, isFaulty, cfg);
    std::cout << "Mean-Field Final Matrix (Round " << mf.convergedRound << "):\n";
    printMatrix(mf.history.back(), numNodes);

    const int tracked = static_cast<int>(mf.history.size()) - 1;
    const MonteCarloSummary mc = runMonteCarloBrachaMultiHop(topo, isFaulty, cfg, tracked);

    std::cout << "Monte Carlo 3-Phase Bracha (M=" << cfg.mcTrials
              << " trials, Equivocation adversary):\n"
              << "  Delivery rate:         " << (mc.deliveryRate * 100.0) << "%\n"
              << "  Mean rounds:           " << mc.meanRounds << " (p95: " << mc.p95Rounds << ")\n"
              << "  Safety violations:     " << mc.safetyViolations << "\n"
              << "  Mean messages / run:   " << mc.meanMessages << "\n"
              << "  Mean bandwidth / run:  " << mc.meanKilobytes << " KB\n"
              << "  Mean sig verifications:" << mc.meanSigVerifications << "\n";

    return 0;
}