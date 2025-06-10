# Graph-based EEG Analysis

This repository contains implementations of various neural network models applied to EEG data, including:

- **CNN (Convolutional Neural Network)**
- **MPNN (Message Passing Neural Network)**
- **Graph Transformer Network**
- **Simple Baseline LSTM**


## Repository Structure

```
├── project_report.pdf # Project Report
├── CNN.ipynb # Convolutional Neural Network notebook
├── MPNN.ipynb # Message Passing Neural Network notebook
├── example.ipynb # Given example notebook containing Baseline LSTM implementation
├── helper.py # Utility functions shared across notebooks
├── graph_transformer/
│ ├── distances_3d.csv # Given distance matrix for EEG nodes
│ ├── eeggraphdataset.py # Dataset class for EEG graph input
│ ├── helpers.py # Preprocessing and FFT utilities
│ ├── main.py # Main script to train/test the model
│ ├── placeholder.py # Placeholder script
│ ├── run.run # Script to execute full graph transformer pipeline
│ ├── transformer_model.py # Graph transformer model definition
│ └── utils.py # Utilit functions
```
---

## Dependencies

Make sure you have Python ≥ 3.10 installed.

### Install dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Authors

Eylül İpçi, Filip Mikovíny, Tuğba Tümer, Yağız Gençer
