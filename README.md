# Pattern Validation Toolkit

A lightweight Python library for testing whether an observed pattern in
data is statistically real -- or just random noise. Built around four
core hypothesis tests (chi-square, z-test, Kolmogorov-Smirnov, Markov
transition), each returning a structured, plain-language result.

Validated against two independent real-world datasets:
- 1,824 historical lottery draws (confirmed: no exploitable pattern, as expected for a regulated system)
- 574 US bank failures, 2000-2026 (correctly detected and dated the 2008-2012 financial crisis, and confirmed the system's recovery by 2013)

See [`case_study.md`](./case_study.md) for the full write-up.

## Install

```bash
pip install numpy scipy
```

No other dependencies. Drop `pattern_validator.py` into your project.

## Quick start

```python
from pattern_validator import test_uniformity, test_proportion_vs_baseline

# Are these categories really unequal, or is that just noise?
result = test_uniformity([140, 145, 138, 142, 139, 210])
print(result)

# Did this rate really change vs. a known baseline?
result = test_proportion_vs_baseline(successes=620, n_trials=5000, baseline_p=0.10)
print(result)
```

## API

| Function | Use it when... |
|---|---|
| `test_uniformity(counts)` | You have counts across categories and want to know if they're really unequal |
| `test_proportion_vs_baseline(successes, n, baseline_p)` | You want to compare an observed rate to a known/expected rate |
| `test_distribution_match(sample, simulate_fn)` | You want to compare a sample to a theoretical model that's easier to simulate than derive |
| `test_markov_transition(binary_sequence)` | You want to know if a state genuinely depends on the previous state (momentum/streaks) |

Every function returns a `TestResult` with `.statistic`, `.p_value`,
`.significant`, and a plain-language `.interpretation` -- ready to hand
to a non-technical stakeholder.

## Run the demo

```bash
python pattern_validator.py
```

## License

MIT -- use freely, including commercially.
