# INSTRUCTION.md

**Project**  
Spectrally Adaptive Mirror Descent with Streaming Sketches  
**Goal**  
Author production‑quality Python code to run rigorous experiments that **directly** support the paper “Spectrally Adaptive Mirror Descent with Streaming Sketches: Instance‑Optimal Regret and Fast Rates”. You are a coding agent. You will implement algorithms, data generators, metrics, sanity checks, unit tests, ablations, and a final notebook that reproduces all figures and tables for the paper.

You must prioritize correctness, reproducibility, and traceability. Every claim drawn in the paper must be backed by a corresponding experiment, metric, or inequality check in this repo.

---

## 1. Scope and Acceptance Criteria

1. Implement the proposed method **SAMD‑SS**: Spectrally Adaptive Mirror Descent using a streaming sketch preconditioner.
2. Implement baselines: SGD, AdaGrad‑Diag, AdaGrad‑Full (for small dimensions), and ONS‑Diag.  
   AdaGrad‑Full and exact log‑det computations are only required when the dimension is small enough to run reliably on CPU.
3. Provide synthetic and real‑data experiments that validate four core theoretical statements:

   3.1 Elliptical potential control  
   \[
   S_T \triangleq \sum_{t=1}^{T} g_t^\top H_t^{-1} g_t 
   \;\le\; \frac{2}{1-\epsilon}\,\log\frac{\det(\lambda I + G_T)}{\det(\lambda I)} \quad \text{up to numerical tolerance.}
   \]

   3.2 Instance‑dependent regret bound with effective dimension scaling  
   \[
   \mathrm{Reg}_T(x^\star) \le D \sqrt{\frac{2\lambda}{1-\epsilon}\,\log\frac{\det(\lambda I + G_T)}{\det(\lambda I)}}.
   \]

   3.3 Fast rates under strong convexity using \(\eta_t=\tfrac{1}{\alpha t}\) with clear \(\log T\) growth.

   3.4 Sketch inflation behaves as predicted when the sketch size \(r\) varies, and the empirical inflation approaches \(1/(1-\widehat{\epsilon}_{\mathrm{ridge}})\).

4. Produce a single executable Jupyter notebook that:
   4.1 Runs all experiments deterministically with fixed seeds.  
   4.2 Generates all plots and tables saved under `artifacts/figures` and `artifacts/tables`.  
   4.3 Verifies each inequality or bound with explicit pass or fail checks.  
   4.4 Exports LaTeX‑ready tables and PDFs of figures.

5. Provide unit tests and numerical sanity checks that run via `pytest` and must pass before the notebook runs.

6. Reproducibility:
   6.1 Single `environment.yml` with pinned versions.  
   6.2 A top‑level `run_all.sh` that sets seeds, runs unit tests, launches sweeps, and builds the notebook outputs.

Failure to meet any acceptance item is a fail.

---

## 2. Repository Layout

Create the following structure exactly.

├── environment.yml
├── README.md
├── INSTRUCTION.md
├── run_all.sh
├── src/
│ ├── algorithms/
│ │ ├── samd_ss.py
│ │ ├── sgd.py
│ │ ├── adagrad_diag.py
│ │ ├── adagrad_full.py
│ │ └── ons_diag.py
│ ├── sketches/
│ │ ├── frequent_directions.py
│ │ ├── oja_sketch.py
│ │ └── randomized_svd_stream.py
│ ├── data/
│ │ ├── synthetic.py
│ │ └── real.py
│ ├── losses/
│ │ ├── squared.py
│ │ └── logistic.py
│ ├── metrics/
│ │ ├── regret.py
│ │ ├── logdet.py
│ │ ├── stability.py
│ │ └── complexity.py
│ ├── utils/
│ │ ├── config.py
│ │ ├── logging.py
│ │ ├── linalg.py
│ │ ├── projections.py
│ │ ├── reproducibility.py
│ │ └── timers.py
│ └── experiments/
│ ├── run_experiment.py
│ ├── sweep.py
│ └── configs/
│ ├── 01_effective_dimension.yaml
│ ├── 02_sketch_inflation.yaml
│ ├── 03_fast_rates.yaml
│ ├── 04_stability_generalization.yaml
│ └── 05_large_scale.yaml
├── tests/
│ ├── test_algorithms.py
│ ├── test_sketches.py
│ ├── test_metrics.py
│ └── test_sanity.py
├── artifacts/
│ ├── logs/
│ ├── figures/
│ └── tables/
└── notebooks/
└── main_experiments.ipynb


---

## 3. Environment and Dependencies

1. Provide `environment.yml` with pinned versions that are widely available on Linux x86_64. Example:


name: samd-ss
channels: [conda-forge, defaults]
dependencies:
- python=3.10
- numpy=1.26.*
- scipy=1.11.*
- scikit-learn=1.4.*
- matplotlib=3.8.*
- pandas=2.2.*
- numba=0.58.*
- tqdm=4.66.*
- jupyterlab=4.2.*
- ipywidgets=8.1.*
- pytest=7.4.*
- psutil=5.9.*


2. All scripts must run on CPU by default. If GPU is available, do not change numerical results.

3. Enforce `PYTHONHASHSEED`, NumPy RNG seeds, and `random.seed` through `src/utils/reproducibility.py`.

---

## 4. Mathematical Objects and Notation Mapping

You will implement the following objects.

1. Cumulative covariance  
\( G_t = \sum_{s=1}^{t} g_s g_s^\top \)

2. Metric  
\( H_t = \lambda I + \widetilde{G}_{t-1} \)

3. Elliptical term  
\( q_t = g_t^\top H_t^{-1} g_t \)

4. Potential  
\( \Phi_T(\lambda) = \log \frac{\det(\lambda I + G_T)}{\det(\lambda I)} \)

5. Effective dimension  
\( d_{\mathrm{eff}}(\lambda;G_T) = \mathrm{tr}\big(G_T(G_T+\lambda I)^{-1}\big) \)

6. Sketch quality proxy with ridge  
\( \widehat{\epsilon}_{\mathrm{ridge}} = \frac{\| \widetilde{G}_T - G_T \|_2}{\lambda + \lambda_{\min}(G_T)} \) clipped to \([0,1)\).

---

## 5. Core Algorithms and Interfaces

### 5.1 SAMD-SS

File `src/algorithms/samd_ss.py`

Implement a class `SAMDSS` with the following interface:

```python
class SAMDSS:
 def __init__(self, dim, lambda_ridge, step_schedule, sketch_backend, sketch_kwargs,
              constraint=None, alpha_strong_convexity=None):
     """
     dim: int
     lambda_ridge: float, must satisfy lambda_ridge >= max_grad_norm**2 after warmup
     step_schedule: callable t -> eta_t
     sketch_backend: object exposing .update(g) and .matrix() -> B_t B_t^T or an implicit operator
     sketch_kwargs: dict forwarded to the sketch constructor
     constraint: dict or None, e.g. {"type": "l2_ball", "radius": R}
     alpha_strong_convexity: float or None, if using 1/(alpha t) schedule in fast-rate experiments
     """
 def reset(self, x0=None, rng=None): ...
 def step(self, grad, t): ...
 def get_state(self): ...


Update rule for unconstrained case:

𝑥
𝑡
+
1
=
𝑥
𝑡
−
𝜂
𝑡
𝐻
𝑡
−
1
𝑔
𝑡
x
t+1
	​

=x
t
	​

−η
t
	​

H
t
−1
	​

g
t
	​


with

𝐻
𝑡
−
1
=
𝜆
−
1
𝐼
−
𝜆
−
1
𝐵
(
𝐵
⊤
𝐵
+
𝜆
𝐼
𝑟
)
−
1
𝐵
⊤
𝜆
−
1
H
t
−1
	​

=λ
−1
I−λ
−1
B(B
⊤
B+λI
r
	​

)
−1
B
⊤
λ
−1

computed through the Woodbury identity using the sketch matrix 
𝐵
∈
𝑅
𝑑
×
𝑟
B∈R
d×r
.

If constraint is provided as an 
ℓ
2
ℓ
2
	​

 ball, perform the projection

𝑥
𝑡
+
1
=
arg
⁡
min
⁡
∥
𝑥
∥
2
≤
𝑅
1
2
∥
𝑥
−
(
𝑥
𝑡
−
𝜂
𝑡
𝐻
𝑡
−
1
𝑔
𝑡
)
∥
𝐻
𝑡
2
x
t+1
	​

=arg
∥x∥
2
	​

≤R
min
	​

2
1
	​

∥x−(x
t
	​

−η
t
	​

H
t
−1
	​

g
t
	​

)∥
H
t
	​

2
	​


using the scalar dual search detailed in src/utils/projections.py via a monotone root-find with closed-form matvecs through Woodbury.

The class must:

Track 
𝑞
𝑡
q
t
	​

, cumulative 
𝑆
𝑇
=
∑
𝑞
𝑡
S
T
	​

=∑q
t
	​

, and a rolling estimate of 
max
⁡
∥
𝑔
𝑡
∥
2
max∥g
t
	​

∥
2
	​

.

Expose instrumentation hooks to record per-step wall time, memory, and sketch diagnostics.

5.2 Baselines

SGD with fixed and cosine schedules.

AdaGradDiag with diagonal accumulator and standard update.

AdaGradFull using exact 
𝐻
𝑡
=
𝜆
𝐼
+
𝐺
𝑡
−
1
H
t
	​

=λI+G
t−1
	​

 with Cholesky solves for small 
𝑑
d.

ONS-Diag using a diagonal approximation.

All baselines must share a common interface with fit_one_pass(dataset, loss) returning logs.

5.3 Sketch Backends

Implement three choices behind a unified interface:

class Sketch:
    def update(self, g: np.ndarray): ...
    def matrix(self): ...
    def inv_h_matvec(self, v, lambda_ridge): ...
    def diagnostics(self) -> dict: ...
FrequentDirections with rank r, deterministic and streaming. Maintain a thin SVD and shrink the smallest singular values.
Provide an analytic upper bound proxy for 
∥
𝐺
~
−
𝐺
∥
2
∥
G
−G∥
2
	​

 using the tail energy stored during shrink steps.

OjaSketch incremental subspace estimation with learning rate schedule and periodic re-orthogonalization. Return a PSD approx 
𝐺
~
≈
𝐵
𝐵
⊤
G
≈BB
⊤
.

RandomizedSVDStream with mini-batch sketching and power iterations.

All sketches must provide the matrix 
𝐵
B or an operator sufficient for Woodbury calculations and must record:

Current rank, spectral tail estimate, and approximate 
𝜖
^
r
i
d
g
e
ϵ
ridge
	​

.

5.4 Losses and Gradients

Squared loss: 
𝑓
𝑡
(
𝑥
)
=
1
2
(
𝑦
𝑡
−
𝑎
𝑡
⊤
𝑥
)
2
+
𝜆
reg
2
∥
𝑥
∥
2
2
f
t
	​

(x)=
2
1
	​

(y
t
	​

−a
t
⊤
	​

x)
2
+
2
λ
reg
	​

	​

∥x∥
2
2
	​


Gradient: 
𝑔
𝑡
=
−
(
𝑦
𝑡
−
𝑎
𝑡
⊤
𝑥
𝑡
)
𝑎
𝑡
+
𝜆
reg
𝑥
𝑡
g
t
	​

=−(y
t
	​

−a
t
⊤
	​

x
t
	​

)a
t
	​

+λ
reg
	​

x
t
	​

.

Logistic loss with L2: 
𝑓
𝑡
(
𝑥
)
=
log
⁡
(
1
+
exp
⁡
(
−
𝑦
𝑡
𝑎
𝑡
⊤
𝑥
)
)
+
𝜆
reg
2
∥
𝑥
∥
2
2
f
t
	​

(x)=log(1+exp(−y
t
	​

a
t
⊤
	​

x))+
2
λ
reg
	​

	​

∥x∥
2
2
	​

.

Provide numerically stable implementations with clipping for logits. For strong-convexity experiments use 
𝛼
=
𝜆
reg
α=λ
reg
	​

 as a Euclidean lower bound and report the schedule 
𝜂
𝑡
=
1
/
(
𝛼
𝑡
)
η
t
	​

=1/(αt).

6. Data Generators
6.1 Synthetic with Controlled Spectrum

src/data/synthetic.py must support:

Low-rank subspace with ambient dimension 
𝑑
d, intrinsic rank 
𝑘
k, eigenvalues 
{
𝜎
𝑖
}
{σ
i
	​

} decaying as power-law or exponential.
Generate features 
𝑎
𝑡
=
𝑈
Σ
1
/
2
𝑧
𝑡
+
𝜎
i
s
o
𝜉
𝑡
a
t
	​

=UΣ
1/2
z
t
	​

+σ
iso
	​

ξ
t
	​

 with 
𝑧
𝑡
,
𝜉
𝑡
∼
𝑁
(
0
,
𝐼
)
z
t
	​

,ξ
t
	​

∼N(0,I).

Labels: linear model 
𝑦
𝑡
=
s
i
g
n
(
𝑤
⋆
⊤
𝑎
𝑡
+
𝜀
)
y
t
	​

=sign(w
⋆
⊤
	​

a
t
	​

+ε) or real-valued for squared loss.

𝑤
⋆
w
⋆
	​

 is drawn in the same low-rank subspace.

Expose knobs: 
𝑑
,
𝑘
,
𝑇
,
decay_rate
,
𝜎
i
s
o
,
snr
,
seed
d,k,T,decay_rate,σ
iso
	​

,snr,seed.

6.2 Real Data

src/data/real.py may use scikit-learn fetchers with offline caching to ~/.cache/samd_ss/. Provide at least one classification dataset and one regression dataset of moderate dimension 
𝑑
∈
[
100
,
2000
]
d∈[100,2000]. If download fails, experiments must gracefully skip with a warning and continue with synthetic data.

7. Metrics, Bounds, and Diagnostics

Implement in src/metrics/:

Regret to the best in hindsight
Solve 
𝑥
𝑇
⋆
=
arg
⁡
min
⁡
𝑥
∈
𝐾
∑
𝑡
=
1
𝑇
𝑓
𝑡
(
𝑥
)
x
T
⋆
	​

=argmin
x∈K
	​

∑
t=1
T
	​

f
t
	​

(x) using L-BFGS-B with optional projection to 
ℓ
2
ℓ
2
	​

 ball. Then compute 
R
e
g
𝑇
(
𝑥
𝑇
⋆
)
Reg
T
	​

(x
T
⋆
	​

). For logistic, stop when gradient norm is less than 1e-8 or relative decrease less than 1e-10.

Elliptical potential
Track 
𝑆
𝑇
=
∑
𝑡
=
1
𝑇
𝑞
𝑡
S
T
	​

=∑
t=1
T
	​

q
t
	​

.
For small 
𝑑
d, compute 
Φ
𝑇
(
𝜆
)
Φ
T
	​

(λ) exactly using np.linalg.slogdet.
For large 
𝑑
d, implement Hutch++ log-det approximation with Rademacher probes and Lanczos; provide absolute and relative error estimates.

Effective dimension
Compute 
𝑑
e
f
f
(
𝜆
;
𝐺
𝑇
)
d
eff
	​

(λ;G
T
	​

) exactly when feasible or via trace estimators.

Stability
Replace-one stability: run paired experiments on datasets that differ in exactly one example but share the same RNG. Report the maximum parameter deviation over time and the test generalization gap difference.

Complexity
Measure per-step wall time, cumulative time, memory via psutil and tracemalloc. Validate the observed cost trends with 
𝑂
(
𝑑
𝑟
+
𝑟
3
)
O(dr+r
3
).

Sketch inflation
Estimate 
𝜖
^
r
i
d
g
e
ϵ
ridge
	​

. Report the empirical ratio 
𝜌
𝑇
=
𝑆
𝑇
sketch
𝑆
𝑇
full
ρ
T
	​

=
S
T
full
	​

S
T
sketch
	​

	​

 where the denominator uses exact 
𝐻
𝑡
H
t
	​

 in small-d runs. Compare 
𝜌
𝑇
ρ
T
	​

 to 
1
/
(
1
−
𝜖
^
r
i
d
g
e
)
1/(1−
ϵ
ridge
	​

).

All metrics must be persisted as CSV and JSON lines in artifacts/logs/.

8. Experiments to Implement

Each experiment is configured by a YAML under src/experiments/configs/. For every config below, run 10 seeds unless stated otherwise and report mean ± standard error with 95 percent confidence intervals where applicable.

8.1 E1: Effective Dimension Scaling

Goal: validate instance-dependent behavior on synthetic data with power-law spectra.

Vary decay_rate ∈ {0.5, 1.0, 1.5, 2.0} which alters 
𝑑
e
f
f
d
eff
	​

.

Fixed 
𝑑
=
500
d=500, 
𝑇
=
10000
T=10000, lambda_ridge selected after warmup as 
≥
max
⁡
𝑡
∥
𝑔
𝑡
∥
2
≥max
t
	​

∥g
t
	​

∥
2
.

Methods: SAMD-SS, AdaGrad-Diag, SGD.

Outputs:

R
e
g
𝑇
Reg
T
	​

 vs 
𝑇
T

Φ
𝑇
Φ
T
	​

 vs 
𝑇
T

R
e
g
𝑇
/
Φ
𝑇
Reg
T
	​

/
Φ
T
	​

	​

 vs 
𝑇
T

Table comparing final regret across methods and mapping to 
𝑑
e
f
f
d
eff
	​

.

8.2 E2: Sketch Inflation

Goal: quantify regret and potential inflation when sketch size 
𝑟
r changes.

Vary 
𝑟
∈
{
8
,
16
,
32
,
64
,
128
}
r∈{8,16,32,64,128} at 
𝑑
=
2000
d=2000, 
𝑇
=
20000
T=20000.

Methods: SAMD-SS with FrequentDirections, OjaSketch, RandomizedSVDStream.

Outputs:

𝑆
𝑇
S
T
	​

 vs 
𝑇
T for each 
𝑟
r.

Plot of 
𝜌
𝑇
ρ
T
	​

 and 
1
/
(
1
−
𝜖
^
r
i
d
g
e
)
1/(1−
ϵ
ridge
	​

) as functions of 
𝑟
r.

Table: time and memory by 
𝑟
r.

8.3 E3: Fast Rates under Strong Convexity

Goal: verify 
log
⁡
𝑇
logT regret behavior with 
𝜂
𝑡
=
1
/
(
𝛼
𝑡
)
η
t
	​

=1/(αt).

Use logistic regression with L2 regularization on synthetic data and one real dataset.

Methods: SAMD-SS, AdaGrad-Diag, SGD with tuned schedules.

Outputs:

R
e
g
𝑇
Reg
T
	​

 vs 
log
⁡
𝑇
logT showing linear trend.

Test loss vs passes.

Bound check: cumulative bound computed online compared to observed regret.