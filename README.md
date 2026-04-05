# TimeNet & MultiModalNet for ASD Classification using MEG

This repository provides the implementation of deep learning models for classifying Autism Spectrum Disorder (ASD) using resting-state magnetoencephalography (MEG) data.

## Overview

Two models are implemented:

- TimeNet: A convolutional neural network (CNN) using time-domain MEG images
- MultiModalNet: A multimodal CNN integrating both time-domain and frequency-domain (PSD) representations

---

## Features

- 10-fold StratifiedGroupKFold cross-validation
- Subject-level prediction using probability aggregation
- Evaluation metrics:
  - Accuracy
  - ROC AUC
  - F1-score
- Visualization of training curves and performance

---

## Requirements

```bash
pip install -r requirements.txt

```
---
## Data Format

The input data should be preprocessed into image format:

Time-domain signal images:

CL_subjectID_channel_001.png

PSD images (for MultiModalNet):

CL_subjectID_channel_001_psd.png
Labels:
CL = ASD
TD = Typically Developing

---
## Usage
Run TimeNet
python TimeNet_upload.py
Run MultiModalNet
python Multimodalnet_upload.py

---
## Notes
Please modify the data paths in the script before running.
Data and trained models are not included in this repository.
The code is intended for research purposes.

---
## Contact

For questions or collaborations, please contact:xsydoc@126.com
