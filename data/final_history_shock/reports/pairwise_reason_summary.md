# Pairwise Evaluation — Reason Summary

Reason codes are assigned via heuristic keyword matching. A single reason can match multiple categories.


## 00_vs_10: Why does generic retrieval beat history retrieval?

**Winner distribution**: {'00': 44, '10': 14, 'tie': 2}

| Reason code | Count | % |
|-------------|-------|---|
| feature_match | 32 | 53.3% |
| tradeoff_disclosure | 24 | 40.0% |
| budget_fit | 22 | 36.7% |
| credibility_history | 21 | 35.0% |
| trust | 10 | 16.7% |
| lower_regret | 5 | 8.3% |
| other | 3 | 5.0% |

**Sample reasons**:

- *headphones_000*: Package A clearly highlights the must-have volume limiting and frames tradeoffs more relevantly, while Package B presents a lack of controls as a negative even though the consumer prefers simplicity.
- *headphones_001*: Beats offers strong brand recognition, premium presentation, and reliable performance aligned with a safe gift choice, while the Kurdene earbuds feel generic and low-end, risking lower perceived value and satisfaction.
- *headphones_002*: Neither option is ideal, but Bluetooth earbuds are more likely to work with a TV than a USB-only headset, which many TVs don’t support. Package B overstates compatibility, while A is more plausible despite comfort tradeoffs.
- *headphones_003*: Both packages describe the same product with nearly identical claims and similar acknowledgment of limitations, offering no meaningful difference in expected purchase or satisfaction outcomes.
- *headphones_004*: The Logitech H390 is purpose-built for video calls with a noise-cancelling mic, simple USB setup, and more accessible controls, making it better suited for an older adult prioritizing clarity and ease of use.


## 10_vs_11: Why does history expression beat generic expression (product held fixed)?

**Winner distribution**: {'11': 38, 'tie': 1, '10': 21}

| Reason code | Count | % |
|-------------|-------|---|
| tradeoff_disclosure | 50 | 83.3% |
| trust | 37 | 61.7% |
| budget_fit | 20 | 33.3% |
| credibility_history | 18 | 30.0% |
| feature_match | 15 | 25.0% |
| persona_fit | 4 | 6.7% |
| lower_regret | 4 | 6.7% |
| other | 1 | 1.7% |

**Sample reasons**:

- *headphones_000*: Package A is transparent about lacking the required volume limiting, helping the parent avoid a poor fit, while Package B misleadingly claims it has this feature, likely leading to disappointment and reduced trust.
- *headphones_001*: Package A more clearly signals the product’s budget nature and tradeoffs, which better aligns expectations; both are poor fits for a premium gift, but A is more honest and less likely to mislead.
- *headphones_002*: Package A is more transparent about limitations and avoids overstating durability, making it more trustworthy while still addressing ease of use and comfort for the consumer.
- *headphones_003*: Both packages present the same product with nearly identical benefits and similar caveats, offering no meaningful difference in expected purchase decision, satisfaction, or trust.
- *headphones_004*: Package A more honestly communicates tradeoffs and emphasizes simplicity without highlighting potentially fiddly controls, which better fits the user’s needs, though B’s more positive tone may slightly boost purchase likelihood.


## 01_vs_11: Why does generic retrieval + history expression beat full history?

**Winner distribution**: {'11': 17, '01': 43}

| Reason code | Count | % |
|-------------|-------|---|
| feature_match | 29 | 48.3% |
| tradeoff_disclosure | 24 | 40.0% |
| budget_fit | 24 | 40.0% |
| credibility_history | 19 | 31.7% |
| trust | 15 | 25.0% |
| lower_regret | 10 | 16.7% |
| other | 4 | 6.7% |
| persona_fit | 1 | 1.7% |

**Sample reasons**:

- *headphones_000*: Package B incorrectly claims a volume-limiting feature the product does not have, which would lead to disappointment and safety concerns; Package A is transparent about this key limitation, leading to a more informed and ultimately better outcome.
- *headphones_001*: Package A aligns with the consumer’s desire for a premium, recognizable gift with strong presentation and reliability, while Package B is too low-cost and generic to feel thoughtful or gift-worthy.
- *headphones_002*: Package B better matches the user's desire for simplicity and long-wear comfort, while Package A downplays the potential hassle of Bluetooth pairing with a TV and may be less comfortable for extended use.
- *headphones_003*: Package A is more honest about limitations like durability and only claims “decent” isolation, setting more realistic expectations, while B overstates performance which could lead to disappointment.
- *headphones_004*: The Logitech H390 is purpose-built for clear voice calls with a noise-canceling mic and simple USB setup, making it more reliable and easier for this user, while still honestly noting control sensitivity issues.
