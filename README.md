# Catan GAN

Catan GAN is an experimental machine learning project that explores generating Catan-style hex terrain tiles using Generative Adversarial Networks (GANs). The project focuses on dataset preparation, model training, and qualitative evaluation of generated terrain images rather than production-ready asset generation.

The goal is to better understand image-based generative modeling, training stability, and dataset constraints when working with structured game assets.

---

## Project Goals

- Explore GAN architectures for terrain-style image generation
- Train models on hex-shaped satellite imagery tiles
- Evaluate generated outputs for visual coherence and class diversity
- Experiment with dataset size, preprocessing, and resolution tradeoffs

This is primarily a learning and research-oriented project.

---

## Dataset

- Source: Satellite imagery cropped into hex-shaped tiles
- Dataset size: ~39k total tiles, ~24k usable after filtering
- Classes: Terrain categories inspired by Catan-style tiles (e.g. forest, field, mountain, water)
- Image resolution: Experimented with 128×128 and higher resolutions

The full-resolution dataset was evaluated but ultimately rejected for final training due to hardware and training-time constraints.

---

## Model Approach

- Generative Adversarial Networks (GANs)
- CNN-based generator and discriminator architectures
- Experiments with:
  - Different latent vector sizes
  - Batch sizes and learning rates
  - Class-conditional generation
  - Training stability and mode collapse mitigation

Checkpoint saving and progress monitoring are used to support long training runs.

---

## Evaluation

Evaluation is primarily qualitative and exploratory:

- Visual inspection of generated tiles
- Grid-based sampling to compare diversity across terrain classes
- Monitoring loss curves and training behavior over time

This project does not aim to produce production-ready assets or fully objective evaluation metrics.

---

## Project Structure

---

## Getting Started

1. Clone the repository
2. Install dependencies (Python, TensorFlow or PyTorch depending on configuration)
3. Prepare the dataset using provided preprocessing scripts
4. Run training notebooks or scripts
5. Inspect generated outputs in the `outputs/` directory

Exact training parameters and experiments are documented in notebooks.

---

## Notes

- This project prioritizes clarity and experimentation over optimization
- Training time and results are hardware-dependent
- Outputs are intended for research and learning purposes only

---

## License

See LICENSE file for details.
