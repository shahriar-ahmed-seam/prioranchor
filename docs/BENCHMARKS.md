# Quantitative Benchmarks & Evaluation Methodology

This document details the mathematical metric formulations, benchmark datasets, and statistical confidence criteria used to evaluate PriorAnchor.

## Metric Formulations

### 1. Subject Fidelity (DINO / CLIP-I Score)

Quantifies visual identity retention between adapted generation $G$ and ground truth subject references $R$:

$$\text{Sim}_{subj}(G, R) = \frac{1}{|R|} \sum_{r \in R} \frac{\phi_{img}(G) \cdot \phi_{img}(r)}{\|\phi_{img}(G)\|_2 \|\phi_{img}(r)\|_2}$$

Where $\phi_{img}$ represents vision encoder embeddings (DINO ViT-S/16 or CLIP ViT-L/14).

### 2. Prompt Alignment (CLIP-T Score)

Measures semantic adherence of generated images $G$ to test conditioning text prompt $T$:

$$\text{Sim}_{text}(G, T) = \frac{\phi_{img}(G) \cdot \phi_{text}(T)}{\|\phi_{img}(G)\|_2 \|\phi_{text}(T)\|_2}$$

### 3. Class Drift Divergence (CDD)

Quantifies degradation of general class representations when prompted with generic descriptions $T_{class}$ without the identifier token:

$$\text{CDD} = 1.0 - \frac{1}{|K|} \sum_{k=1}^K \frac{\phi_{img}(G_{adapted}(T_{class})) \cdot \phi_{img}(G_{base}(T_{class}))}{\|\phi_{img}(G_{adapted}(T_{class}))\|_2 \|\phi_{img}(G_{base}(T_{class}))\|_2}$$

Where $G_{base}$ is output from the un-adapted frozen foundation model. Lower values indicate superior prior preservation.

### 4. Language Bleed Index (LBI)

Measures excess subject similarity when conditioned on generic class prompts:

$$\text{LBI} = \max\left(0, \text{Sim}_{subj}(G_{adapted}(T_{class}), R) - \text{Sim}_{subj}(G_{base}(T_{class}), R)\right)$$

An ideal system exhibits $\text{LBI} = 0.0$, indicating that generic prompts generate un-personalized class members without leaking subject features.

## Empirical Benchmark Results

Evaluated across 30 subjects over 25 diverse categories (1,200 evaluation generations per configuration):

| Method | Subject Fidelity | Prompt Alignment | Class Drift Divergence | Language Bleed Index | Retention Rate |
|---|---|---|---|---|---|
| **Base Model (Zero-Shot)** | 0.2140 +/- 0.02 | 0.8120 +/- 0.01 | 0.0000 | 0.0000 | 100.0% |
| **Naive Fine-Tuning** | 0.8842 +/- 0.03 | 0.6120 +/- 0.04 | 0.5418 +/- 0.05 | 0.4280 +/- 0.04 | 45.8% |
| **DreamBooth (Full Weights)** | 0.8710 +/- 0.02 | 0.7645 +/- 0.02 | 0.0824 +/- 0.02 | 0.0410 +/- 0.01 | 91.8% |
| **PriorAnchor (LoRA r=16)** | **0.8825 +/- 0.02** | **0.7812 +/- 0.02** | **0.0614 +/- 0.01** | **0.0216 +/- 0.01** | **93.9%** |

## Production Acceptance Thresholds

For enterprise deployment, model checkpoints must satisfy:

- Subject Fidelity: $\ge 0.7500$
- Prompt Alignment: $\ge 0.7000$
- Class Drift Divergence: $\le 0.2500$
- Language Bleed Index: $\le 0.1500$
- Prior Retention Rate: $\ge 85.00\%$
