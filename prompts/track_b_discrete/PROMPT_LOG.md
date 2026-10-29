# Prompt Log: Track B: Discrete Optimization

## Entry 1
- Date: 29/10/26
- Tool and model version: 
- Original prompt: 
STRICT RULES FOR THIS TASK: Do NOT run any command, script, test, python, pip, git or shell command. Do NOT create scratch or test scripts anywhere. Do NOT install anything. Do NOT use torch. Use only numpy, scipy (scipy.optimize.milp for ILP), networkx, scikit-learn, pandas, matplotlib and pytest. Do not modify common/ or any existing file except those named in sections E and F. Before writing code, read the files in common/ so you use their exact function signatures. All code gets type hints, docstrings and short comments explaining the optimization logic. All randomness uses seeded np.random.default_rng.

Create these files in tracks/track_b_discrete/:

1. knapsack.py (selection under a capacity constraint: 0/1 knapsack, maximise total value subject to total weight <= capacity)
   - make_instance(n, seed): integer weights uniform in [5, 50]; values = weights + integer noise in [0, 10] (strongly correlated, harder for exact methods); capacity = floor(0.5 * sum(weights)).
   - brute_force(w, v, cap): exact, only for n <= 20.
   - greedy_ratio(w, v, cap): sort by value/weight, add while feasible.
   - dp_knapsack(w, v, cap): exact dynamic programming table over capacity, returns best value and chosen items.
   - branch_and_bound(w, v, cap, node_limit=2_000_000, time_limit_s=20.0): depth-first, items sorted by ratio, upper bound = fractional (LP-relaxation) bound, prune when bound <= incumbent; return best value, chosen items, nodes explored, and a flag proven_optimal (False if a limit was hit).
   - ilp_knapsack(w, v, cap): scipy.optimize.milp with binary variables and one capacity constraint; return best value, chosen items.
   Every function returns a small dict or dataclass with keys value, items, runtime_s (use common.timing), plus method-specific fields.

2. qubo.py
   Conventions (document them at the top of the file): QUBO energy E(x) = x^T Q x with Q upper-triangular (diagonal = linear terms), x in {0,1}^n, to be MINIMISED. Ising energy E(s) = sum_{i<j} J[i,j] s_i s_j + sum_i h[i] s_i + offset with s in {-1,+1}^n and the substitution x_i = (1 - s_i)/2 (so s_i = +1 corresponds to x_i = 0, matching the qubit convention where Z has eigenvalue +1 on |0>).
   - qubo_energy(Q, x), ising_energy(h, J, offset, s).
   - qubo_to_ising(Q) -> (h, J, offset), derived exactly from the substitution above (expand x_i = (1 - s_i)/2 in both linear and quadratic terms; do not guess signs).
   - maxcut_qubo(graph): maximise cut = sum over edges (x_i + x_j - 2 x_i x_j); as a minimisation E = -cut, so Q[i,i] = -degree(i) and Q[i,j] = +2 for each edge i<j. Also return cut_value(graph, x) as an independent direct count of cut edges.
   - feature_selection_qubo(relevance, redundancy, k, alpha, lam): minimise E(x) = -sum_i r_i x_i + alpha * sum_{i<j} R_ij x_i x_j + lam * (sum_i x_i - k)^2. Expand the penalty term exactly into Q (the constant lam*k^2 goes into a separate returned offset). Return Q and offset.
   - brute_force_qubo(Q, offset=0.0) vectorised with numpy over all 2^n assignments (chunked so n up to 20 does not exhaust memory), returning the minimum energy, all optimal bitstrings, and runtime.
   - greedy_flip_local_search(Q, seed, restarts=20): start from random x, repeatedly flip the single bit that reduces energy most until no flip improves.
   - simulated_annealing_qubo(Q, seed, n_sweeps=2000, t0=2.0, t1=0.01): single-bit-flip Metropolis with geometric cooling.

3. run_all.py with command line: python tracks/track_b_discrete/run_all.py [--quick] [--out results/track_b]. In quick mode use fewer sizes, seeds and sweeps so it finishes in under 30 seconds; full mode must finish in about 3 minutes. It starts with a sys.path insert of the repository root. Use common.seeding, common.timing, common.stats and common.io_utils. Every table is saved as CSV in the output directory, every plot as PNG (dpi 150, labelled axes with units, legend, title). It runs three experiments, each wrapped in try/except that prints the traceback and continues (exit code 1 at the end if any failed):

   Experiment B1, knapsack comparison: sizes n in [10, 20, 30, 40, 50] (quick: [10, 20, 30]), 5 instance seeds per size (quick: 2). Methods: greedy, DP, branch-and-bound, ILP, and brute force only for n <= 20. Verification built into the run: for every instance assert that DP value == ILP value; assert that brute force equals DP when it was run; assert that branch-and-bound equals DP whenever proven_optimal is True; assert greedy value <= DP value; assert every returned item set is feasible (total weight <= capacity) and that its recomputed value equals the reported value. If an assertion fails, record the failure in a "verification_failures" list that is written to the output and printed, and continue. Table: n, seed, method, value, optimality_gap_pct versus DP (computed from the raw values), runtime_s, nodes (B&B only), proven_optimal (B&B only), feasible. Plots: runtime versus n (log y) per method (mean over seeds), and greedy optimality gap (%) versus n (mean with min/max error bars). A summary table with mean runtime and mean gap per method and size.

   Experiment B2, Max-Cut as QUBO/Ising: graphs = networkx.random_regular_graph(3, n, seed=1) for n in [6, 8, 10, 12, 14, 16] (quick: [6, 8, 10]). For each n: build the QUBO and the Ising model; verify on every one of the 2^n assignments for n <= 12 that qubo_energy(Q, x) equals ising_energy(h, J, offset, s(x)) up to 1e-9 and equals -cut_value(graph, x) (record the maximum absolute discrepancy in the table). Solve with brute force (exact baseline), greedy_flip_local_search and simulated_annealing_qubo (10 seeds each, report mean/min/max of the cut value, the fraction of seeds that reached the exact optimum, and runtime). Table: n, n_edges, method, best_cut, mean_cut, success_rate, runtime_s, max_discrepancy. Plot: brute-force runtime versus n (log y) together with heuristic runtimes, and cut value found by each method versus n. Also write results/track_b/qubo_instances.json for the later quantum track, containing for n in [4, 6, 8, 10] (always, even in quick mode): n, edge list, Q as nested lists, Ising h as a list, Ising J as a dict with keys "i,j", offset, exact optimal energy, optimal cut value, the list of all optimal bitstrings as strings with x_0 first, and a "convention" string repeating the conventions above.

   Experiment B3, feature selection as QUBO with a penalty term: load sklearn.datasets.load_breast_cancer, standardise, pick 12 features with np.random.default_rng(0).choice(30, 12, replace=False). relevance_i = |corr(feature_i, y)|; redundancy_ij = |corr(feature_i, feature_j)|. Fix k = 4, alpha = 0.5. (a) Penalty sweep: for lam in [0, 0.1, 0.5, 2.0, 5.0] build the QUBO, solve exactly with brute force and record: number of features selected by the exact QUBO optimum, whether the cardinality constraint (== k) is satisfied, and the objective -sum r x + alpha*sum R x x without the penalty. Plot the number of selected features versus lam with a horizontal line at k. (b) With lam = 2.0 compare: the exact QUBO optimum, greedy forward selection (add the feature with the best marginal objective gain k times, objective computed without penalty), simulated annealing on the QUBO (10 seeds), and 200 random k-subsets. For each selection compute the objective and the 5-fold cross-validated accuracy of LogisticRegression (max_iter=1000) using only the selected features; random subsets report mean and std. Tables for (a) and (b).

   At the end run_all.py writes SUMMARY.md in the output directory with: environment info (python, numpy, scipy, networkx, sklearn versions, platform, cpu count, seeds), and for each experiment a heading, a one-paragraph setup description, its tables rendered as markdown (write a small renderer, do not depend on tabulate), the figure filenames, and the list of verification failures (or the line "No verification failures"). It must contain only numbers produced by the run and no interpretations.

4. tests/test_discrete.py (write only; do not run):
   - on n=12 instances for 3 seeds: brute_force == dp == ilp == branch_and_bound (proven optimal); greedy <= optimum; every solution is feasible.
   - on a tiny hand-made instance, DP returns a known optimum.
   - for a 6-node graph and a random Q: qubo_energy equals ising_energy for all 64 assignments; maxcut_qubo energy equals -cut_value for all assignments; brute_force_qubo minimum equals the best cut found by enumerating cuts directly.
   - feature_selection_qubo: for random r, R and n=8, the QUBO energy plus offset equals the direct formula -sum r x + alpha*sum R x x + lam*(sum x - k)^2 for all 256 assignments.
   - tests/test_smoke.py: run_all's main function with quick=True into pytest's tmp_path; assert the expected CSV and PNG files, SUMMARY.md and qubo_instances.json exist.
   Add tests/conftest.py that inserts the repository root and tracks/track_b_discrete into sys.path.

E. tracks/track_b_discrete/README.md: fill only Objective, Formulation, Methods and How to run (python tracks/track_b_discrete/run_all.py [--quick]; python -m pytest tracks/track_b_discrete/tests -q). Formulation must state: knapsack (variables x_i in {0,1}, maximise sum v_i x_i subject to sum w_i x_i <= C); Max-Cut as QUBO and Ising with the exact conversions above; feature selection QUBO with the penalty term lam*(sum x - k)^2. Leave Results, Correctness check and Limitations unchanged. In CHECKLIST.md (repository root) update only the rows of the Discrete area and the QUBO/Ising, Max-Cut, feature selection and dynamic programming rows: Experiment = B1, B2 or B3 as appropriate (greedy, ILP, branch-and-bound, dynamic programming, knapsack -> B1; QUBO, Ising, binary-to-spin conversion, Max-Cut -> B2; feature selection, penalty terms -> B3), Code location = file path, Status = "Code ready - results pending". Do not touch other rows.

F. README.md (repository root), "AI Tools Register" table only: add or extend the single row for Antigravity IDE / Claude Sonnet 5.5 so that Tracks includes "B", Dates includes 2026-10-29, and Kind of work includes "code generation, unit-test writing, experiment scripting". Do not create duplicate rows and do not change anything else in README.md.

- AI response / relevant excerpt: 
- Revised prompt (if any): 
- Resulting commit link: 
- Verification note (done by the student): 
- Chat export / screenshot filename: 
