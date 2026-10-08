# exp — code for *SIA: Surface-Isomorphic Abstention*

Experiments behind the paper's criterion, method, and controlled construction.

Run everything from this directory; scripts write to `out/`.

---

## 1. Requirements

```bash
pip install -r requirements.txt
```

- **Python 3.11+**, **PyTorch**, and a GPU.
- The reported runs used Python 3.11, PyTorch 2.13.0+cu130, on one DGX Spark
  GB10 with 121 GiB of unified memory. A 9B model in bf16 needs **18.3 GiB**,
  so a 16 GB card will not hold it whole; see *Running on a small GPU* below.
- **MATLAB R2023a** with the Statistics and Machine Learning Toolbox, for
  `surf_cv.m` (the criterion check) and `make_figs.m` (the paper's figures)
  only. Nothing else needs MATLAB.

### Model

Qwen3.5-9B (~18 GB). Point the code at your copy:

```bash
export SIA_MODEL_DIR=/path/to/Qwen3.5-9B/master
```

`common.py` falls back to the original author's absolute path if this is
unset, so external users should always set it.

### Running on a small GPU

`common.load_model` takes device placement from the environment, so the model
can be split between the card and system RAM without touching the source:

```bash
export SIA_DEVICE_MAP=auto       # let accelerate place the layers
export SIA_MAX_MEMORY=11GiB      # budget for device 0
export SIA_MAX_MEMORY_CPU=40GiB  # budget for the host
```

Leaving these unset reproduces the original single-device placement. Layers
that land on the CPU make the run several times slower, but the arithmetic is
still bf16, so the numbers remain comparable. **Do not use `precision="4bit"`
to fit a smaller card** if you intend to compare against the paper's figures:
quantisation changes the base model's behaviour and makes it a different
experiment.

---

## 2. Reproducing the paper

```bash
# Main experiment (paper section 5): the crossed, three-unknown-entity
# construction, plus the one-unknown and all-answer-control arms.
G3_12_PROFILE=multiunk python g3_12_abstain_method.py

# The earlier rounds that motivate the three conditions (paper section 4).
G3_12_PROFILE=main     python g3_12_abstain_method.py   # uncrossed templates
G3_12_PROFILE=cross    python g3_12_abstain_method.py   # crossed, one entity

# Origin of the whole line: the gate that will not abstain (paper section 1).
python g3_9_missing_cue.py
```

```matlab
% The criterion check (paper section 5.2). Writes out/surf_cv_result.json.
run('surf_cv.m')

% The paper's figures. Writes ../figures/fig1..4.pdf.
run('make_figs.m')
```

Multi-seed reruns of the main experiment, which is how the seed spread is
measured:

```bash
for s in 1 2 3; do
  G3_12_PROFILE=multiunk G3_12_SEED=$s python g3_12_abstain_method.py
done
# writes out/g3_12_abstain_method_multiunk_seed{1,2,3}.json
```

Without `G3_12_SEED` the scripts behave exactly as they did for the paper: the
seed is the fixed constant `7`.

---

## 3. File map

### Used by the paper

| File | Role |
|---|---|
| `common.py` | Model loading, device placement, path resolution, text backbone. |
| `capability.py` | Verifiable capability battery (math / format) and held-out LM loss. |
| `abstain_data.py` | The synthetic slice: 9 isomorphic series, shared fields, crossed templates. |
| `g3_12_abstain_method.py` | The main experiment. Profiles `main` / `cross` / `multiunk`. |
| `g3_9_missing_cue.py` | The gated-adapter origin experiment (Figure 1). |
| `topic_domains.py` | Topic sets used by `g3_9`. |
| `s5_edit.py` | The SFT step used by `g3_12`. |
| `mk_surf_inputs_syn.py` | Exports the synthetic slice's item table for the criterion check. |
| `surf_cv.m` | MATLAB: the counter-example completeness check. |
| `make_figs.m` | MATLAB: the four figures. |
| `probe_host.py` | Not part of the paper. Checks whether a host can hold the model. |

### Entangled helpers

`g3_12_abstain_method.py` imports `g3_5_isolation.py` (for `LAYERS`, `TARGETS`,
`R16`, `A16`, `SEQ`, `SEED`) and `g3_1_premise.py`; `g3_9_missing_cue.py`
imports `g3_7_router.py`. Those modules belong to a different line of work, so
they are included here to keep every import resolvable rather than to support
the paper's claims.

### Kept for context, not used by the paper

`g3_0`–`g3_8`, `g3_10`, `g3_11`, `g3_13`–`g3_16`, `ge_peft.py`, `domains.py`,
`overlap_domains.py`, `grpo.py`, `kb_grade.py`, `real_kb_data.py`,
`mk_surf_inputs.py`, `s0`–`s9`. These belong to the companion paper and to a
separate singularity / gated-adapter study. The `s*.py` scripts in particular
are unrelated to abstention.

---

## 4. Two things to know before trying to reproduce

1. **The synthetic slice is label-pure by construction.** Every supported
   series is answer-only and every unknown series is abstention-only, which is
   why leave-one-series-out is undefined on it (`surf_cv.m` will tell you so)
   and why the third condition — at least three unknown entities — is needed.
   It also means the labels do not have to be re-measured, so this experiment
   reproduces on a different host, unlike the companion paper's real-boundary
   experiment.
2. **`memory`-heavy arms are sensitive to adapter creation order.** Creating
   an adapter changes the RNG stream for the next one. A control arm moved from
   1.000 to 0.480 under this effect. The fix, applied throughout, is to create
   one `zero` adapter under a fixed seed, snapshot it, and `copy_` it back for
   every arm with a bit-level equality assertion.

---

## 5. Grader integrity

There must be exactly one grader. In the companion line, a second inline copy
drifted from `kb_grade.py`: it stripped the entity string throughout the
response, including inside the gold answer, and scored `Brazil :: capital` as
a failure because its gold answer contains the entity string. Two
implementations of one grader are two graders. Import it; do not re-implement
it.

---

## 6. License

Apache License 2.0. See [../LICENSE](../LICENSE).
