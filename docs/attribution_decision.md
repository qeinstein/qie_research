# Phase 5f: Attribution Decision

## Verdict: Negative Result with Mechanistic Attribution

Not "Geometric Advantage." Not "Numerical Equivalence." The honest claim is a
**structured negative result** — which is more valuable than a weak positive, because
it comes with a complete mechanistic explanation for *why* each encoding fails and
*under what conditions* it would not.

---

## Verdict per encoding

### Amplitude → Spectral Collapse

L2-normalization projects all inputs onto the unit hypersphere, destroying magnitude
information. On any dataset where features carry amplitude variation, the encoded
feature matrix collapses to near-rank-1:

- erank: wine=1.37, breast_cancer=1.64, dry_bean=1.04, covertype=3.24
- κ: dry_bean=5.7×10⁹, breast_cancer=766k, covertype=532k
- Consequence: logistic regression receives a near-rank-1 signal with a catastrophically
  ill-conditioned Gram matrix → Cohen's d = −111.9 (dry_bean), −44.0 (covertype),
  −44.3 (higgs)
- **The single exception proves the rule:** on `high_rank_noise` (d=200,
  noise-dominated), amplitude achieves erank=197, κ=2.22 — and is the only near-tie
  with the best classical (d=−0.03, p=0.96). When the data's intrinsic structure fills
  the unit sphere uniformly, the collapse does not happen.

### Angle → Geometric Equivalence to Linear

The cos/sin expansion is a smooth, differentiable map — but geometrically it reduces
to a scaled linear transform under logistic regression. CKA(angle, raw_linear) ≥ 0.95
on 7/10 datasets. The "nonlinearity" in the encoding does not add a representational
dimension that a linear classifier can exploit.

- Achieves statistical near-parity on wine (d=−0.18, p=0.70) and breast_cancer
  (d=+0.10, p=0.83)
- Still significantly worse in 8/10 datasets where the best classical method has
  genuine nonlinear capacity (torch_mlp, rbf_svm)
- The angle encoding is a quantum-circuit-inspired structure that collapses, under
  logistic probing, to the same geometry as standardized raw features

### Basis → Binary Quantization Distortion

Most geometrically distinct from classical (CKA 0.42–0.43 vs raw_linear,
anti-correlated with poly2 on high_rank_noise). High effective rank (erank/d ≈
0.70–0.88), well-conditioned on image datasets (κ=107 on CIFAR-10). But binary
quantization loses the fine-grained amplitude information that linear classifiers need.
The Hamming geometry imposed by uniform bit-slicing is misaligned with the smooth
discriminative manifolds in all 10 datasets.

- 0/10 datasets: significantly competitive
- Best result: high_dim_parity (d=−0.77, p=0.16) — underpowered, not significant

---

## Central claim (paper-ready)

> Quantum-inspired encodings, deployed as fixed feature maps with linear classification,
> provide no statistically significant accuracy advantage over classical feature
> engineering across 10 diverse benchmark datasets. Each encoding exhibits a distinct,
> spectrally identifiable failure mode: amplitude encoding undergoes rank collapse under
> L2-normalization (κ up to 10⁹, erank down to 1.04), rendering the encoded space
> uninformative for all but intrinsically high-rank noise-dominated datasets; angle
> encoding is geometrically equivalent to a scaled linear map (CKA ≥ 0.95 with raw
> features on 7/10 datasets), providing no nonlinear lift under logistic probing; and
> basis encoding, while spectrally well-conditioned, imposes a binary Hamming geometry
> misaligned with smooth discriminative boundaries. The one structural condition under
> which QIE approaches competitive performance — high effective rank and low condition
> number — is satisfied only when the dataset's intrinsic geometry already fills the
> encoding's output space uniformly, a property that does not generalise. These findings
> are consistent across 5 random seeds (24/30 pairwise comparisons significantly worse
> at α=0.05, 3 negligible near-ties, 0 statistically significant wins), with effect
> sizes ranging from negligible (|d|<0.2) on favourable datasets to catastrophic
> (|d|>100) under spectral collapse.

---

## Qualified positive findings

1. **Encoding overhead is not the bottleneck** — amplitude encodes in 35 µs (same as
   StandardScaler), 313× cheaper than poly2. If accuracy could be traded for speed,
   QIE would win on this axis.
2. **Amplitude on high_rank_noise** — the one dataset where the geometry is right,
   amplitude is statistically indistinguishable from the best classical (d=−0.03,
   p=0.96). This is not an accident; it is a falsifiable prediction of the spectral
   attribution framework.
3. **Basis encoding's distinct geometry** could be valuable as a *complement* to linear
   methods (low CKA = low redundancy), not as a replacement — an ensemble direction
   worth noting as future work.

---

## Evidence pointers

| Claim | Source |
|---|---|
| 24/30 significantly worse, 0 wins | `results/summary/statistical_tests.csv` |
| Amplitude erank collapse (1.04–3.24) | `results/summary/spectral_attribution.csv` |
| Amplitude κ up to 5.7×10⁹ | `results/summary/spectral_attribution.csv` |
| CKA(angle, raw_linear) ≥ 0.95 on 7/10 | `results/summary/cka_scores.csv` |
| Amplitude/high_rank_noise: d=−0.03, p=0.96 | `results/summary/statistical_tests.csv` |
| Amplitude encoding: 35 µs, poly2: 10.98 s | `results/summary/overhead_summary.csv` |
| Forest plot (effect sizes) | `results/figures/forest_plot.png` |
| κ vs accuracy gap scatter | `results/figures/spectral_kappa_vs_gap.png` |

---

## Implication for paper framing

The paper should not be framed as "QIE fails" — that is uninteresting. It should be
framed as: **"We provide the first systematic spectral attribution of QIE failure modes,
identifying the precise geometric conditions under which each encoding degrades, and the
one structural condition under which near-parity is achieved."** The negative result is
the finding. The mechanistic explanation is the contribution.
