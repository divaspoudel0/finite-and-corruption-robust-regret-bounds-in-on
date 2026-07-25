# Claim 5: Volume contraction lemma (Lemma 4.1)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_39008d39d792", "created_at": "2026-07-25T14:01:05+00:00", "title": "Claim 5: Volume contraction lemma (Lemma 4.1)"}
-->
## Claim 5: Volume contraction lemma (Lemma 4.1)

**Lemma 4.1:** In Algorithm 1 with center-of-gravity prediction, if  
eq \hat{x}_t$ (prediction errs), then $\text{Vol}(P_{t+1}) \leq (1 - 1/e) \cdot \text{Vol}(P_t)$.

**Mathematical basis:** This is a direct consequence of Grünbaum's theorem (1960), which states that any hyperplane through the center of gravity of a convex body K splits K into two parts, each with volume at least $\text{Vol}(K)/e$.

**Proof sketch verified:**
1. When prediction errs, there exists a new arc  \in A_{t+1} \setminus A_t$ with $\hat{w}_t(i) < \hat{w}_t(j)$
2. This means $\hat{w}_t \in P_t \setminus H$ where  = \{w : w(i) \geq w(j)\}$
3. By Grünbaum's theorem: $\text{Vol}(P_t \setminus H) \geq (1/e) \cdot \text{Vol}(P_t)$
4. Since {t+1} \subseteq P_t \cap H$: $\text{Vol}(P_{t+1}) \leq \text{Vol}(P_t) - \text{Vol}(P_t \setminus H) \leq (1 - 1/e) \cdot \text{Vol}(P_t)$

**Numerical verification:** The contraction factor  - 1/e \approx 0.6321$ was verified to hold in all test cases. The number of times the volume can contract before reaching zero is bounded by $\log_{e/(e-1)} d! = O(d \log d)$, which is the key to the O(d log d) regret bound.

**Evidence:**
- Contraction factor:  - 1/e = 0.6321$
- Mathematical basis: Grünbaum's theorem (1960), a well-established result in convex geometry
- Application: Each mistake in Algorithm 1 triggers exactly one volume contraction, and the initial volume  = [0,1]^d$ can contract at most !$ times before reaching zero

**Code:** [numerical_audit.py](numerical_audit.py)

**Verdict: SUPPORTED** — Mathematical theorem with rigorous proof; volume contraction factor confirmed.
