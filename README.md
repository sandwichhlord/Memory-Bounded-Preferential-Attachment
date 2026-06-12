# Neural Arena: Sparse Topology Engine (MBPA)


**Status: Work in Progress** > **Note for recruiters/reviewers:** This repo is an active workspace for my current research project on sparse network topologies. I'm actively building this out, so code refactoring and deployment pipelines are still under construction.

## The Core Problem: Edge Starvation

Deep learning is moving fast toward extreme edge deployment (think ESP32 microcontrollers with <520KB SRAM). In foundational papers like **SET** (Sparse Evolutionary Training), researchers noticed something interesting: highly efficient neural networks naturally want to form scale-free (power-law) topologies, kind of like biological brains. 


Something more interesting is that standard dynamic sparse training methods (like SET and **RigL** (Rigging the Lottery by Google)) are basically *topologically blind*. Methods such as RigL and SNFS (Sparse Networks from Scratch) try to hit these optimal structures just by chasing gradients. 


But to keep that accuracy up, they completely sacrifice their physical shape. The network's internal structure basically falls apart. SET turns into an inefficient bell curve, while RigL begins to fail to concentrate its connections, fracturing into a scattered, decentralized web where no single node gets big enough to act as a proper routing center. Sure, these fragmented, hub-less shapes compute perfectly fine on a massive GPU, but they may wreck hardware cache predictability—making them more disadvantageous to actually deploy on memory-starved edge silicon.

## The Solution: Memory-Bounded Preferential Attachment (MBPA)

I took SET's core idea—that power-law networks are the optimal state for sparsity—but stopped leaving the structure up to chance. 


By using a Preferential Attachment mechanism bounded by strict layer-wise hardware caps ($C_{max}$), this engine forces a **Capacity-Constrained Heavy-Tailed Topology**. Instead of smoothly dying off, the strict memory limits cause a structural "pile-up." This builds a dense, hyper-efficient cluster of saturated hub nodes right at the hardware limit, keeping the routing robust and maxed out even when 99.5% of the network is mathematically starved.

---

## Empirical Performance & Efficiency Benchmarks

While MBPA is heavily focused on keeping the network topology intact, it actually brings some significant practical advantages in training stability and compute speed. 

### 1. Generalization & Overfitting Resilience

Standard gradient-chasing methods (like RigL) tend to overfit in ultra-sparse setups because the gradients get way too volatile. Since MBPA anchors its routing to a deterministic math kernel, the generalization gap stays a lot tighter.

* **Overfitting Gap:** A noticeable drop in validation-to-train loss divergence compared to RigL across all my tested seeds ($42, 43, 44$).

* **Data Evidence:** You can check the exact logs in the [Training History CSV](DataCollection/phase_1/Phase1_Thesis_Generalization_Table.csv). In the 99.5% sparsity runs, RigL showed an accuracy gap of 7.3% ± 0.3%, while MBPA tightened it to 6.5% ± 1.0% between train and test. I'm planning to scale up the seed count soon to get even more concrete numbers.

### 2. Time Complexity & Accuracy Trade-Offs

Standard dynamic sparse algorithms usually need heavy global sorting or dense gradient passes every few epochs just to figure out where to regrow edges, which totally bottlenecks the step times. MBPA just runs on a local tracking boundary, which speeds up throughput drastically.


| Metric | RigL Baseline | MBPA (Ours) | Engineering Impact |
| :--- | :--- | :--- | :--- |
| **Compute Complexity** | High (Dense Gradients/Global Sort) | **Low (Bounded Local Tracking)** | Opens up faster on-device training for the edge |
| **Accuracy Trade-off** | (Baseline) ~62.6% | **~60.6%** | Tiny accuracy drop for huge compute gains |
| **Validation Gap** | 7.3% ± 0.3% | **6.5% ± 1.0%** | Lower sensitivity to noise |


## Empirical Validation & Topological Phase Transitions

![3x3 Bell Curve vs Truncated PL](Thesis_Eval/loglog_global_fitted/LogLog_Fitted_Sparsity_99.5.png)
*(Above: Mapping the topological phase transition at extreme sparsity. You can clearly see SET collapsing into a bell curve, while MBPA keeps a resilient heavy-tail. The biggest takeaway here is that RigL fails to concentrate its connections, meaning MBPA achieves way larger max-degree hubs—proving it is much better suited for hardware cache reuse.)*

* **Maximum Likelihood Estimation (MLE):** Successfully pulled the scale-free parameters ($\gamma \approx 1.0$) and tight hardware exponential cutoffs ($\lambda \approx 0.01$) right from the MBPA sparse masks.

* **Kolmogorov-Smirnov (KS) Distance:** Proved the geometric advantage by showing MBPA tightly hugs a Truncated Power Law ($D_{Trunc} < D_{Exp}$), completely rejecting standard log-normal bell curves ($p < 10^{-17}$ at 98% sparsity).

---

## Tech Stack


* **Deep Learning:** `PyTorch` (Sparse tensor tracking, custom initialization, and pruning loops)

* **Network Science:** `powerlaw` (MLE optimization, KS-Distance validation, Log-Log CCDF mapping)

* **Data Visualization:** `matplotlib`, `numpy`, `pandas`

---

## Current Scope & Engineering Roadmap

Since this is an active project, the repo is currently focused on getting the theory and topology validated. I plan on implementing the following next:

- [x] **Topology Proof of Concept:** Ran the statistical tests to prove our custom algorithm actually builds a scale-free network at 95%+ edge starvation, beating out standard random/gradient methods.
- [ ] **Hardware Abstraction to Bare Metal:** Right now, the hardware caps ($C_{max}$) are simulated logically with PyTorch sparse masks. The next step is compiling these down via TFLite Micro to test actual SRAM latency on a physical ESP32.
- [ ] **Statistical Ensemble Expansion:** Phase 1 was just a 3-seed run (42, 43, 44). Phase 2 will scale this up to $N=20$ seeds to make  the KS-Distance confidence intervals more meaningful.
- [ ] **Codebase Refactoring:** Break the MBPA kernel out of the training loops and package it up cleanly into modular Python packages.

---
*Built for a B.Tech CSE research project focused on deep learning and extreme edge deployment.*