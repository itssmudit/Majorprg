# Federated Learning with Differential Privacy for Medical Data Analytics

IILM University, Greater Noida.
B.Tech CSE Major Project

Team: Mudit Kumar Pal, Jitender Kumar, Amitesh Kumar, Akash Kumar. 

Guide: Mr Amul Tiwary.

Five simulated hospitals train a shared model with Federated Averaging (FedAvg), without sharing
patient records. Differential privacy (update clipping + Gaussian noise) is added, and the
privacy-utility trade-off is measured on three public medical datasets.

## Datasets used
- Breast Cancer Wisconsin (built into scikit-learn)
- Pima Indians Diabetes (`data_diabetes.csv`)
- Heart Disease, Cleveland (`data_heart.csv`)

## Run
    pip install -r requirements.txt
    python fl_dp.py

Outputs: result tables in the terminal and plots saved as `privacy_utility_*.png`.

## Limitations as of now
Simulated clients, small test sets, no clinical validation.
