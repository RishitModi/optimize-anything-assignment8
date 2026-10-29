# Track H: DBMS Query Optimization

## Objective
The objective of Track H is solving the relational database index selection problem under strict storage budget constraints. Given an analytical workload of $Q = 12$ SQL queries and a set of $C = 16$ candidate secondary indexes, the goal is to select an optimal subset of indexes that maximizes workload execution time saved (or query throughput) without exceeding a storage budget $B$ (evaluated at 10%, 25%, and 50% of the total candidate index storage footprint).

### Dataset Description
The benchmark runs on synthetic TPC-H-like relational data generated deterministically by `db_data.py`:
- **Tables and Attributes:**
  - `customer(c_custkey, c_nationkey, c_mktsegment, c_acctbal)`
  - `orders(o_orderkey, o_custkey, o_orderstatus, o_totalprice, o_orderdate, o_orderpriority)`
  - `lineitem(l_orderkey, l_linenumber, l_partkey, l_suppkey, l_quantity, l_extendedprice, l_discount, l_shipdate, l_returnflag)`
  - `part(p_partkey, p_brand, p_type, p_size, p_retailprice)`
  - `supplier(s_suppkey, s_nationkey, s_acctbal)`
  - `nation(n_nationkey, n_name, n_regionkey)`
- **Scale Factor Row Counts:**
  - `customer`: $\max(100, \lfloor 150000 \times \text{sf} \rfloor)$
  - `orders`: $\max(1000, \lfloor 1500000 \times \text{sf} \rfloor)$
  - `lineitem`: uniformly 1..7 items per order ($\approx 4$ on average)
  - `part`: $\max(100, \lfloor 200000 \times \text{sf} \rfloor)$
  - `supplier`: $\max(10, \lfloor 10000 \times \text{sf} \rfloor)$
  - `nation`: 25 fixed rows
- **Schema Design:** Deliberately created **without** declared primary/foreign keys and **without** initial secondary indexes. All dates are integer day offsets ($0..2556$) and prices are integer cents, ensuring exact arithmetic.

## Baseline
The baseline configuration is the default SQLite query planner operating on the raw unindexed tables with automatic temporary indexing enabled (the SQLite engine default). In this state, analytical joins, range filters, and point lookups require full table scans (`SCAN TABLE`) or on-the-fly ephemeral B-tree creation, providing a controlled baseline against which secondary index speedups are measured.

## Formulation

### General Index Selection Problem
Given candidate indexes $\{1, \dots, C\}$, let $x_i \in \{0, 1\}$ denote whether index $i$ is created ($x_i = 1$) or omitted ($x_i = 0$). Each index consumes $s_i > 0$ bytes of storage. The total index storage must not exceed budget $B$:
$$\sum_{i=1}^C s_i x_i \le B, \quad x_i \in \{0, 1\}$$

### Interaction-Aware MILP Formulation
In real query execution, a query $q$ selects at most one secondary index (the best available) for each table it accesses. Under the single-table index assumption, query $q$ achieves benefit $\max_{i \in S} b_{q, i}$, where $b_{q, i} = \max(0, t_{\text{baseline}}(q) - t_i(q))$ is the time saved by query $q$ when only index $i$ exists.

Introducing assignment variables $y_{q, i} \in \{0, 1\}$ indicating whether query $q$ uses index $i$:
$$\max_{x, y} \sum_{q=1}^Q \sum_{i=1}^C b_{q, i} y_{q, i}$$
subject to:
1. **Index Availability:**
   $$y_{q, i} \le x_i \quad \forall q \in \{1, \dots, Q\}, \; \forall i \in \{1, \dots, C\}$$
2. **Single-Index Choice per Query:**
   $$\sum_{i=1}^C y_{q, i} \le 1 \quad \forall q \in \{1, \dots, Q\}$$
3. **Storage Budget:**
   $$\sum_{i=1}^C s_i x_i \le B$$
4. **Integrality:**
   $$x_i \in \{0, 1\}, \quad y_{q, i} \in \{0, 1\}$$

Solved to provable global optimality via `scipy.optimize.milp`.

### Additive Knapsack Formulation (DP & Static Greedy)
Approximates total workload benefit as an additive sum:
$$\max_x \sum_{i=1}^C \left(\sum_{q=1}^Q b_{q, i}\right) x_i \quad \text{s.t.} \quad \sum_{i=1}^C s_i x_i \le B, \quad x_i \in \{0, 1\}$$
For dynamic programming, sizes are discretized to 4096-byte page units $w_i = \lceil s_i / 4096 \rceil$ and capacity $W = \lfloor B / 4096 \rfloor$, guaranteeing exact budget compliance.

## Methods
- **`select_greedy_static`**: Static benefit-to-size heuristic sorting candidates by $\left(\sum_q b_{q, i}\right) / s_i$ and greedily packing while capacity permits.
- **`select_greedy_marginal`**: Iterative greedy algorithm computing the marginal gain in interaction-aware predicted benefit per byte $\Delta \text{Benefit} / s_i$ at each step (at most $C$ iterations).
- **`select_dp_knapsack`**: Exact 0/1 knapsack dynamic program on additive candidate values and page-aligned weights.
- **`select_ilp`**: Exact Mixed-Integer Linear Programming via `scipy.optimize.milp` over the interaction-aware formulation.
- **`select_bruteforce`**: Exhaustive enumeration of all $2^C$ subsets evaluated in vectorized chunks of 4096 boolean masks using NumPy (applicable for $C \le 18$).

## How to run
<!-- Instructions to execute experiments and tests for this track -->
```bash
python tracks/track_h_dbms/run_all.py [--quick] [--sf 0.05]
python -m pytest tracks/track_h_dbms/tests -q
```

## Results
<!-- Experimental outcomes, comparison tables, and plots saved in results/ -->

## Correctness check
<!-- Verification steps, assertions, unit tests, and theoretical invariants checked -->

## Limitations
<!-- Computational bottlenecks, approximations, assumptions, and failure modes -->
