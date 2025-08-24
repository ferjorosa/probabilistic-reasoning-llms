# probabilistic-reasoning-llms

## GPU Requirements

**Note:** This project uses PyTorch nightly builds to support RTX 5090 GPU compatibility. The RTX 5090 requires CUDA 12.8 support which is only available in PyTorch nightly builds, not the stable releases. If you're using a different GPU, you may be able to use stable PyTorch versions instead.

## Setup
This project uses UV as the package manager. To set up the project:

1. Install UV if you haven't already:

```shell
curl -sSf https://astral.sh/uv/install.sh | sh
```

2. Create a virtual environment and install dependencies:
 
```shell
uv sync