# Thermal Image Enhancement & Colorization for Satellite Imagery

> 🚧 **Work in Progress**  
> This repository is currently under active development as part of the **Bharatiya Antariksh Hackathon (BAH) 2026**. Features, models, and documentation are continuously being improved.

## Overview

This project aims to develop an end-to-end deep learning framework for enhancing the spatial resolution of thermal infrared (TIR) satellite imagery and generating realistic RGB visualizations while preserving geospatial information.

The proposed framework consists of two stages:

1. **Thermal Super-Resolution** – Enhancing 200 m TIR imagery to 100 m resolution.
2. **Thermal Colorization** – Translating enhanced thermal imagery into realistic RGB imagery.

The long-term objective is to build a **physics-aware, geospatially consistent** framework for large-scale Earth observation applications.

---

## Current Status

- ✅ Dataset preprocessing pipeline
- ✅ Thermal Super-Resolution training pipeline
- ✅ Thermal Colorization training pipeline
- ✅ End-to-end inference pipeline
- ✅ GeoTIFF input/output support
- 🚧 Physics-aware learning *(In Progress)*
- 🚧 Overlap-tile inference with seamless blending *(In Progress)*
- 🚧 Attention-enhanced colorization network *(In Progress)*
- 🚧 Documentation and benchmarking *(In Progress)*

---

## Repository Structure

```text
.
├── Enhancement/          # Thermal Super-Resolution
├── Colorization/         # Thermal-to-RGB Colorization
├── inference_utils/      # Inference utilities
├── weights/              # Exported model weights
├── outputs/              # Generated outputs
├── inference.py          # End-to-end inference pipeline
└── export_model.py       # Export Lightning checkpoints
```

---

## Future Work

- Physics-aware thermal reconstruction
- Attention-based colorization
- Multi-scale feature fusion
- Overlap-tile inference with weighted blending
- Improved model generalization on diverse satellite scenes
- Comprehensive benchmarking and evaluation

---

## Disclaimer

This repository represents an actively evolving research prototype. Model architectures, training strategies, inference methods, and documentation may change significantly as development progresses.