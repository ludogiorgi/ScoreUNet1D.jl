# AGENTS.md

## Goal

Create a **professional Julia repository** that implements a **1D U-Net architecture** to learn the **score function** of a dataset using **denoising score matching at zero diffusion time**.

The code must be **dataset-agnostic** and usable for **any 1D data** (e.g. time series, spatial fields). I will then test it on a **Kuramoto–Sivashinsky (KS)** dataset stored in an HDF5 file in the `data/` folder.

---

## Repository structure and general requirements

1. Create a **professional Julia package repository**, including:
   - `Project.toml`
   - `Manifest.toml`
   - `.gitignore`

2. The repository must contain a folder:
   - `data/`
     - This folder will contain an **HDF5 file with a dataset** (e.g. the KS dataset that I will use for testing).

3. The code must:
   - Be written in **Julia**.
   - Implement a **1D U-Net** architecture with an **option for periodic convolutions**.
   - Be **professionally written**, **well optimized**, and **parallelized on CPU**.
   - Have a **modular structure**, with clear separation of concerns (model definition, data handling, training, evaluation, etc.).
   - Be **well commented**, easy to read, and easy to modify.

4. After implementing the code:
   - Ensure that it **compiles**.
   - Ensure that it **passes all tests**.

---

## Model requirements: 1D U-Net with optional periodic convolutions

1. Implement a **1D U-Net** in Julia with:
   - Configurable **input/output dimensions** suitable for general 1D data.
   - An **option to use periodic convolutions** (for data with periodic boundary conditions).
   - A design that makes it simple to switch between **periodic** and **non-periodic** convolutions via configuration or constructor arguments.

2. For periodic convolutions:
   - Implement them so that the convolution wraps around the boundaries (circular padding).
   - Make this implementation reusable across the network blocks.

3. Ensure that:
   - The implementation is **modular** (e.g. separate files/modules for layers, blocks, U-Net architecture).
   - The code is **well optimized** and **parallelized on CPU**.
   - The code remains **clear and readable**, with **good comments** explaining key design choices.

---

## Training requirements: denoising score matching at zero diffusion time

1. The repository must implement **denoising score matching** to learn the score function at **zero diffusion time**.

2. Given data samples \( x_i \) from the dataset:
   - Perturb the data as:
     \[
     y_i = x_i + \sigma z_i,\quad z_i \sim \mathcal{N}(0, I)
     \]
   - Use a **small fixed noise level**:
     - Set **\(\sigma = 0.05\)**.

3. The training objective should be based on **learning the score function through**:
   - The conditional expectation **\(\mathbb{E}[z \mid y]\)**.
   - The network should be trained so that, given the noisy input \( y \), it learns to approximate the quantity needed to reconstruct the score via this conditional expectation.

4. The code must:
   - **Normalize the data before training**.
   - Provide a **clean training pipeline** that:
     - Loads data from the HDF5 file.
     - Normalizes it.
     - Constructs noisy samples \( y_i \) from clean samples \( x_i \) using \( \sigma = 0.05 \).
     - Trains the 1D U-Net to learn the score function via denoising score matching.

5. The design must remain **general** so that:
   - Any suitable 1D dataset can be plugged into the same pipeline.
   - For the KS dataset (used later), **periodic convolutions** can be enabled.

---

## Langevin SDE integration and validation

1. Once the score function \( s(x) \) is learned, use it to integrate the Langevin equation:
   \[
   \dot{x} = s(x) + \sqrt{2}\,\xi,
   \]
   where:
   - \( s(x) \) is the **learned score function**.
   - \( \xi \) is standard Gaussian white noise.

2. For SDE integration:
   - Use the Julia package **FastSDE.jl**:
     - Repository: <https://github.com/ludogiorgi/FastSDE.jl>

3. From the Langevin simulation:
   - Obtain the **steady-state PDF for a single mode** of the system.
   - When working with systems like KS where each mode has the same PDF:
     - Compute the PDFs for all modes and **take their average** to get a robust estimate.

4. From the original dataset (e.g. the KS dataset stored in the HDF5 file):
   - Compute the **observed PDF** for the same mode(s).

5. Compare the PDFs:
   - Compare the **PDF from the Langevin simulation** with the **PDF from the observed data**.
   - Compute the **relative entropy** (e.g. Kullback–Leibler divergence) between the two PDFs.

6. Validation criterion:
   - The **relative entropy should be very small**.
   - If the relative entropy is **not** very small, there is likely **something wrong in the code** that needs to be investigated and fixed.

---

## Final checklist

Before considering the task complete, ensure that:

1. The repository contains:
   - `Project.toml`
   - `Manifest.toml`
   - `.gitignore`
   - `data/` folder prepared to hold an HDF5 dataset
   - A **modular, well-commented Julia implementation** of a **1D U-Net** with **optional periodic convolutions**.

2. The code is:
   - **Professionally structured**
   - **Optimized**
   - **Parallelized on CPU**
   - **Readable and easy to modify**

3. All code:
   - **Compiles without errors**
   - **Passes all tests**

4. The training pipeline:
   - **Normalizes the data**.
   - Implements **denoising score matching at zero diffusion time** using:
     \[
     y_i = x_i + \sigma z_i,\quad z_i \sim \mathcal{N}(0, I),\quad \sigma = 0.05,
     \]
   - Learns the score function through the conditional expectation **\(\mathbb{E}[z \mid y]\)**.

5. The learned score function \( s(x) \) has been:
   - Used in the Langevin SDE \( \dot{x} = s(x) + \sqrt{2}\,\xi \) integrated with **FastSDE.jl**.
   - Validated by:
     - Comparing PDFs (from simulation vs. observed data).
     - Computing the **relative entropy** and confirming that it is **very small**; otherwise, the implementation must be inspected and corrected.
