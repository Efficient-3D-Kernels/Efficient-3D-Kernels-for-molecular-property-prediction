# GPU-Accelerated 2D Molecular Graph Matching Kernel

This repository contains a high-performance, GPU-accelerated Python implementation of the **WCS-GNCCP** (Weighted Common Subgraph via Graduated Non-Convexity and Concavity Procedure) algorithm. 

It is designed to perform incredibly fast, highly accurate 2D topological graph matching on massive molecular datasets, specifically optimized for execution on **Google Colab**.

---

## 📁 Automated Data Extraction
The pipeline is designed for "plug-and-play" execution on molecular datasets stored in Google Drive. 

**How it works:**
1. The script points to your dataset directory (e.g., `/content/drive/MyDrive/datasets`).
2. It automatically uses Python's `glob` library to scan the folder and extract matching pairs of `.sdf` files (e.g., `Aromatase_actives_new.sdf` and `Aromatase_inactives_new.sdf`).
3. It **ignores** all irrelevant files (such as PyTorch `.pt` tensors) so your execution does not crash.
4. It extracts the structural topology (bonds/rings) and atomic features directly from the SDF files using RDKit.

---

## ⚙️ Core Components

### 1. The Mathematical Engine (`wcs_gnccp.py`)
This file is the core mathematical solver. It takes two molecules and calculates their exact mathematical similarity.
*   **Continuous Relaxation:** Instead of forcing atoms to match 1-to-1 immediately (which causes algorithms to freeze or crash), it initializes a flat matrix of "fractional" matching probabilities.
*   **Heuristic Cost Matrix:** It extracts atom features (Atomic Number, Hybridization, Charge, etc.) and uses Expert Chemical Heuristics to build an initial penalty matrix, heavily penalizing mismatched elements.
*   **Frank-Wolfe & Rank-One Factorization:** It optimizes the structural alignment. We successfully mathematically reduced the innermost gradient calculation from a slow $O(N^3)$ matrix multiplication to a lightning-fast $O(N^2)$ operation using Rank-One Factorization ($U = r \cdot r^T$).
*   **Sinkhorn-Knopp Projection:** It uses alternating projections to guarantee the matching probabilities never break the mathematical laws of the assignment boundary.
*   **GNCCP Sweep:** It shifts the objective function from Convex to Concave ($\zeta = 1 \rightarrow -1$) to prevent the algorithm from getting trapped in Local Minima. Finally, it uses the **Hungarian Algorithm** to snap the final probabilities to a rigid 1-to-1 Bijective Map.

### 2. The Colab Benchmark Script (`evaluate_colab.py`)
This script acts as the orchestrator. It feeds the extracted datasets into the math engine and measures the accuracy.
*   **Feature Precomputation:** To save massive amounts of time during hyperparameter sweeps, it precomputes the RDKit atomic features of the entire dataset into memory *once*, rather than recalculating them for every pair.
*   **Classification:** It converts the final WCS penalty score into a **Discriminative Indefinite Kernel** similarity score (between 0 and 1). It then evaluates the accuracy of the algorithm using two machine learning classifiers: $k$-Nearest Neighbors ($1$-NN) and a Support Vector Machine (SVM) utilizing an Eigenvalue Ridge Correction.

---

## ⚡ Hardware Integration (CPU & GPU)

Calculating an $N \times N$ similarity matrix for thousands of molecules requires millions of Frank-Wolfe iterations. We engineered a dual-hardware strategy to solve this instantly:

### A. Multi-Core CPU Parallelization
Because comparing Molecule A to Molecule B is completely independent of comparing Molecule A to Molecule C, this is an "embarrassingly parallel" problem. 
*   `evaluate_colab.py` utilizes Python's **`joblib`** library. When running in CPU mode, it automatically blasts the pairwise calculations across all available CPU cores simultaneously, achieving near-perfect parallel speedup.

### B. NVIDIA CUDA GPU Acceleration
For extremely large protein structures where matrix multiplications become the primary bottleneck, we engineered a dynamic GPU backend.
*   **Dynamic Backend Switching:** The `wcs_gnccp.py` engine automatically detects if **CuPy** (a direct CUDA drop-in replacement for NumPy) is installed on the machine. 
*   If the `--use_gpu` flag is thrown, all heavy $O(N^2)$ matrix multiplications, gradients, and Sinkhorn-Knopp projections instantly execute on the NVIDIA GPU hardware.
*   **The CPU Bridge:** Because the Hungarian algorithm (`linear_sum_assignment`) cannot natively run on a GPU, our engine safely pulls the small gradient matrix back to the CPU memory *just* for this step, and then pushes the result back to the GPU to seamlessly continue the Frank-Wolfe loop.

---

## 🚀 How to run on Google Colab

**1. Mount your drive and install dependencies:**
Open a new cell in Colab and run:
```python
from google.colab import drive
drive.mount('/content/drive')

# Install RDKit and ML libraries
!pip install rdkit-pypi scipy numpy scikit-learn
```

**2. For GPU Acceleration (Optional but Recommended):**
Change your Colab runtime to `T4 GPU` (Runtime > Change runtime type), and install CuPy:
```python
!pip install cupy-cuda12x
```

**3. Run the Benchmark (CPU Multi-Core Mode):**
Perfect for smaller molecules or if you do not want to use a GPU.
```bash
%cd /content/drive/MyDrive/FYP_Project/2D_Kernel_Final_Pipeline
!python evaluate_colab.py --data_dir /content/drive/MyDrive/datasets --max_mols 10000 --n_jobs -1
```

**4. Run the Benchmark (GPU Accelerated Mode):**
Perfect for massive protein structures.
```bash
%cd /content/drive/MyDrive/FYP_Project/2D_Kernel_Final_Pipeline
!python evaluate_colab.py --data_dir /content/drive/MyDrive/datasets --max_mols 10000 --use_gpu
```
