# Third-Party Components

The release expects these repositories under `external/`:

| Component | Upstream | Revision used where known |
|---|---|---|
| DualScoreMatching | `https://github.com/FlorentinGuth/DualScoreMatching` | `c84182f312a2b8234d4a78bc4c15e2434cb0a14c` |
| Learning with Noise | `https://github.com/mbaradad/learning_with_noise` | `a31abe38eac2bc9071c639dd925e7bfa7bf8dde9` |
| DINO | `https://github.com/facebookresearch/dino` | record the checked-out revision with each run |
| improved-diffusion | `https://github.com/openai/improved-diffusion` | record the checked-out revision with each run |
| DenseFlow | upstream repository accompanying the selected ImageNet64 checkpoint | record repository and checkpoint hashes |

Only the DualScoreMatching license is copied into this preparation tree. Before
publishing checkpoints or vendoring code, verify and include every applicable
upstream license and model-use term.
