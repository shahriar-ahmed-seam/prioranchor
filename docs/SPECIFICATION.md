# Canonical Specification: RFC-PA-001

## Document Metadata

- **RFC Identifier**: RFC-PA-001
- **Title**: PriorAnchor Subject Personalization Schema & Telemetry Specification
- **Status**: Standard
- **Category**: Technical Protocol

## 1. Abstract

This specification defines the canonical data models, serialization protocols, and invariant verification rules governing PriorAnchor training configurations, low-rank adapter serialization, and evaluation telemetry.

## 2. Core Schemas

### 2.1 TokenIdentifier Schema

Defines the rare subject identifier token and vocabulary bindings.

| Field | Type | Mandatory | Default | Constraints | Description |
|---|---|---|---|---|---|
| `placeholder_token` | `string` | Yes | `"[V]"` | Non-empty | Token string representing the subject. |
| `class_noun` | `string` | Yes | N/A | Lowercase | Semantic category noun (e.g. `'dog'`). |
| `initializer_token` | `string` | No | `None` | Valid token | Initial text encoder embedding reference. |
| `token_id` | `integer` | No | `None` | `>= 0` | Explicit vocabulary index. |
| `rare_token_variance`| `float` | No | `0.01` | `>= 0.0` | Initial perturbation variance. |

### 2.2 PriorDatasetConfig Schema

Governs synthetic prior generation and content-addressable cache verification.

| Field | Type | Mandatory | Default | Constraints | Description |
|---|---|---|---|---|---|
| `num_class_images` | `integer` | No | `200` | `>= 10` | Total prior images generated. |
| `class_prompt_templates` | `list[string]` | No | Default 5 | Non-empty | Formatting templates containing `{class_noun}`. |
| `cache_dir` | `string` | No | `"prior_cache"`| Valid path | Directory destination for prior cache. |
| `guidance_scale` | `float` | No | `7.5` | `> 0.0` | CFG scale during prior synthesis. |
| `seed` | `integer` | No | `42` | Any | Random seed for deterministic prior generation. |

### 2.3 LoRAConfig Schema

Specifies low-rank adaptation parameters.

| Field | Type | Mandatory | Default | Constraints | Description |
|---|---|---|---|---|---|
| `adapter_type` | `enum` | No | `"lora"` | lora, dora, ti | Adaptation strategy. |
| `rank` | `integer` | No | `16` | `1 <= r <= 256` | Rank dimension $r$. |
| `alpha` | `float` | No | `32.0` | `> 0.0` | Scaling factor $\alpha$; scaling = $\alpha / r$. |
| `dropout` | `float` | No | `0.05` | `0.0 <= p <= 0.5` | Dropout probability on adapter activations. |
| `target_modules` | `list[string]` | No | Attention | Non-empty | Substring identifiers of target linear projections. |

## 3. Serialization Protocols

### 3.1 Content-Addressable Prior Fingerprint

Prior datasets MUST compute a 16-character hexadecimal SHA-256 fingerprint:

$$\text{Fingerprint} = \text{SHA256}(M_{base} \mathbin{\Vert} C_{noun} \mathbin{\Vert} R \mathbin{\Vert} N \mathbin{\Vert} S \mathbin{\Vert} T_{templates})_{[:16]}$$

Where $M_{base}$ is base model path, $C_{noun}$ is lowercase class noun, $R$ is resolution, $N$ is sample count, $S$ is seed, and $T_{templates}$ are sorted prompt templates.

### 3.2 Adapter Export Schema

Exported JSON/safetensors files must conform to:

```json
{
  "format": "prioranchor_lora_v1",
  "rank": 16,
  "alpha": 32.0,
  "layers": {
    "unet.attention.to_q": {
      "layer_name": "unet.attention.to_q",
      "in_features": 320,
      "out_features": 320,
      "rank": 16,
      "alpha": 32.0,
      "weight_a": [[...]],
      "weight_b": [[...]],
      "merged": false
    }
  }
}
```
