import numpy as np

import peld


rng = np.random.default_rng()
reference = rng.normal(size=10_000)
candidate = rng.normal(loc=0.25, scale=1.2, size=5_000)

metric = peld.PELD(reference)
print(metric.both(candidate))
