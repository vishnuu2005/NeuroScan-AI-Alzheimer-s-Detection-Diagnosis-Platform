# NeuroScan AI — Recovery & Component Mapping Specification

## Overview
This document records the forensic recovery and architectural reconciliation performed on the NeuroScan AI platform from an unversioned ZIP extraction back to the canonical GitHub repository.

## Discovered Assets & Reconciled Structure
1. **Model Checkpoints**:
   - `resnet2d_alzheimers.pth`: Single-channel Grayscale ResNet-18 model.
   - `resnet2d_alzheimers_best.pth`: Three-channel RGB ResNet-18 model with Dropout regularization.
2. **Clinical Database**:
   - SQLite `neuroscan.db` containing 3 clinicians, 83 validated scans, and 11 consultation threads.
3. **Clinical Evaluation Artifacts**:
   - Training curves and confusion matrix evaluations retained in `plots/`.
4. **Platform Staging**:
   - `NonDemented` (GDS 1-2)
   - `VeryMildDemented` (GDS 3)
   - `MildDemented` (GDS 4)
   - `ModerateDemented` (GDS 5-6)

## Reconciliation Matrix
- Extracted code was aligned with the GitHub master branch.
- Hardcoded drive references (`D:/...`) were replaced with dynamically resolved absolute paths.
- Relative imports inside the `chatbot` module package were standardized.
- Automated testing harnesses and runtime startup scripts were introduced.
