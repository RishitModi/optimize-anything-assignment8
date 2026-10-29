# Prompt Log: Track H: DBMS Query Optimization

## Entry 1
- Date: 2026-10-29
- Tool and model version: Antigravity IDE / Claude Sonnet 5.5
- Original prompt:
```text
STRICT RULES FOR THIS TASK: Do NOT run any command, script, test, python, pip, git or shell command. Do NOT create scratch or test scripts anywhere. Do NOT install anything; use only the Python standard library (sqlite3, hashlib, shutil, time, statistics), numpy, scipy (scipy.optimize.milp), pandas, matplotlib and pytest. Do NOT use multiprocessing. Do not modify common/ or any file of other tracks. To avoid module-name collisions with other tracks, name every module in this track with the prefix db_ exactly as listed. All code gets type hints, docstrings and short comments explaining the logic. All randomness uses seeded np.random.default_rng. IMPORTANT: every loop whose termination depends on data must have an iteration cap or a time guard (previous tracks failed because of non-terminating loops). Heavy files (databases) must never be written into results/ or the repository's tracked folders: use work/track_h/ at the repository root and add the line "work/" to .gitignore.

Create in tracks/track_h_dbms/ (each file starts with a sys.path insert of the repository root and of its own folder):

1. db_data.py
2. db_queries.py
3. db_select.py
4. db_eval.py
5. run_all.py
6. tests (write only, do not run): tests/conftest.py, tests/test_dbms.py
7. tracks/track_h_dbms/README.md
8. .gitignore (repository root): append work/
9. README.md (repository root): update AI Tools Register table
```
- AI response / relevant excerpt:
Implemented Track H (DBMS Index Selection Optimization) architecture:
- `db_data.py`: Schema creation without keys or indexes (customer, orders, lineitem, part, supplier, nation), deterministic scaled generation via numpy, fast chunked insertion with `PRAGMA journal_mode=OFF` and `synchronous=OFF`, metadata caching (`_meta` table), `dataset.csv` persistence, and `copy_database` via SQLite backup API.
- `db_queries.py`: `Query` dataclass, 12 parameterized analytical queries (Q1–Q12) with deterministic parameters, `result_hash`, `run_query` harness, and the 16 candidate secondary index specifications (I1–I16).
- `db_select.py`: Pure numpy index selection routines under interaction-aware and additive models: `select_greedy_static`, `select_greedy_marginal`, `select_dp_knapsack` (page-aligned 4096-byte units), `select_ilp` (`scipy.optimize.milp`), and `select_bruteforce` (vectorized chunked 4096-mask evaluation).
- `db_eval.py`: Page-level physical index size measurement via SQLite page counters, single-index benefit matrix benchmarking with `ANALYZE` and hash verification, and full-configuration evaluation capturing EXPLAIN QUERY PLAN, throughput, and latency.
- `run_all.py`: Orchestrator for experiments H0, H1, H2 with deadline timeout guards, saving CSVs, EXPLAIN QUERY PLAN text dumps, Matplotlib visualizations, and markdown renderer for `SUMMARY.md`.
- `tests/conftest.py` and `tests/test_dbms.py`: Pytest suite verifying scaling formulas, hash invariance, index sizing, ILP optimality vs heuristics across 30 random instances, DP knapsack correctness, budget constraints, and dynamic importlib smoke test.
- Documentation & Registries: Updated Track H README.md, root `.gitignore`, and root `README.md` AI Tools Register.
- Revised prompt (if any): None
- Resulting commit link: Pending
- Verification note (done by the student): Inspection of files, mathematical invariants, type annotations, time guards, and schema definitions.
- Chat export / screenshot filename: N/A
