# SCIC-v2 algorithm audit

This audit distinguishes the mathematical resource bound from an implementation
ranking and from deployment performance. It does not tune on the nine held-out
geometries or relax an observed numerical failure.

1. **Roofline does not rank candidate latency.** Candidate launch overhead,
   occupancy, tile padding, reduction layout, cache reuse, and extra split
   buffers are not determined by FLOPs/ideal cold bytes. The revised policy
   admits candidates in every intensity regime and treats these quantities as
   features. This removes the original memory-band exclusion but does not make
   the resulting regression a reliable cost model.
2. **Validated training support is sparse.** The six HF operator workers each
   admit only three distinct training geometries for split-K=4 and 8. A model
   fitted separately for each candidate cannot learn shape effects from its
   constant split/warp identity columns. Correlated analytical and shape
   features and large LOOG residuals limit identification. See `prior_audit.json`.
   Improving this requires more independent, numerically admitted training
   geometries and schedule/resource measurements; using held-out outcomes to
   select features would invalidate the present test.
3. **Safety expansion costs time.** Wide uncertainty margins frequently expand
   to full probing. This is consistent with the rule, but undermines its search
   savings. Empirical LOOG residuals are not calibrated coverage guarantees;
   the 2% expansion threshold does not establish a regret bound.
4. **Operator and deployment preparation differ.** The operator audit already
   compiled/validated the whole pool and its helper amortizes probe costs.
   The deployment preparer additionally charges incremental compilation and
   validation. Exhaustive-scan accounting cannot establish cold lazy-startup
   savings. Model preparation records describe actual cost under the observed
   compiler cache state, and remain descriptive when complete logits fail.
5. **Semantic correctness has multiple levels.** Matching component outputs
   within tolerance does not guarantee complete-model relative-L2 equivalence
   after many layers. All 65 teacher-forced outputs retain strict verdicts.
   Actual generation checks all request-token hashes. Those checks still do not
   measure language-task accuracy or establish production robustness.
6. **Deployment is bounded.** Serving freezes one candidate per layer using a
   declared 64:1 decode/prefill mixture; it does not implement arbitrary dynamic
   routing. Failed component/joint admission restores the framework layer
   before tracing and records native deployment. FP16 TP1 is the tested scope.
7. **Weight updates require explicit invalidation.** Inference tensors lack an
   automatic mutation counter. Frozen projection or norm weights changed by a
   caller require a version bump and a new preparation/compilation. The tested
   adapters assume loaded weights remain immutable.

Measured results and their final coverage belong to `summary.json`, not this
explanatory audit. A faster revised adapter than the legacy adapter does not
alone prove superiority over the optimized original or an analytical-feature
benefit over the matched no-AI control.
