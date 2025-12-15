# Reproducing Results

This guide explains how to reproduce the main figures and results from the
notebooks in this repository.

## Prerequisites

Before reproducing results, ensure you have:

1. Set up the Python environment as described in the main `README.md`
2. Installed all required dependencies from `requirements.txt`
3. For notebooks that generate figures with LaTeX rendering, ensure you have
   a working LaTeX installation and the `cm-super` package

## Notebooks with Generated Figures

### Coupled Harmonic Oscillators

**Notebook:** `coupled_harmonic_oscillators/N_coupled_harmonic_oscillators_1_D_N_2.ipynb`

**Generated Figures:**
- `figures/hs_n_2.png` - Hamiltonian simulation results (generated in Cell 49)
- `figures/ar_n_2.png` - Analytical solution comparison (generated in Cell 59)

**To Reproduce:**
1. Navigate to the `coupled_harmonic_oscillators/` directory
2. Run the notebook `N_coupled_harmonic_oscillators_1_D_N_2.ipynb`
3. The plots will be displayed using `plt.show()`
4. To save figures, add `plt.savefig('figures/filename.png')` before
   `plt.show()` in the respective cells

**Note:** The figures in the `figures/` directory are precomputed examples.
Running the notebook will generate new plots based on current execution
results, which may differ slightly due to quantum measurement statistics.

**Expected Runtime:** The notebook performs multiple quantum simulations
over different time spans, so execution time depends on the number of shots
and time points. Expect several minutes for full execution.

## Notebooks with Generated Quantum Model Files

Several notebooks generate `.qmod` (quantum model) files:

### Coupled Harmonic Oscillators
- **Files:** `N-2-dt-*.qmod` and `N-2-dt-*.qprog` (where `*` is the time step)
- **Location:** `coupled_harmonic_oscillators/` directory
- **Generated in:** Cell containing `write_qmod(qmod, f"N-2-dt-{t}")`

### Quantum Fourier Transform
- **File:** `qft-expvalue.qmod`
- **Location:** `quantum_fourier_transform_abelian/` directory
- **Generated in:** Cell containing `write_qmod(create_model(main), "qft-expvalue")`

### Non-Unitary Quantum Computing (LCU)
- **File:** `lcu-2x2.qmod`
- **Location:** `nonunitary_quantum_computing/` directory
- **Generated in:** Cell containing `write_qmod(quantum_model, "lcu-2x2")`

**Note:** These `.qmod` files are generated when the corresponding cells are
executed. They are not required for the notebooks to run, but they allow you
to save and reuse quantum circuit models.

## Execution Order

The notebooks are designed to be independent and can be run in any order.
However, if you want to follow a logical learning progression:

1. **Quantum Fourier Transform** (`quantum_fourier_transform_abelian/`) -
   Fundamental concepts
2. **Quantum Walk** (`quantum_walk/`) - Graph-based algorithms
3. **Non-Unitary Quantum Computing** (`nonunitary_quantum_computing/`) -
   Advanced techniques
4. **Minimize Expectation Value** (`minimize_expectation_value/`) -
   Optimization basics
5. **Quantum Variational Algorithms** (`quantum_variational_algorithms/`) -
   VQE and QUBO
6. **Coupled Harmonic Oscillators** (`coupled_harmonic_oscillators/`) -
   Hamiltonian simulation

## Troubleshooting

### Figures Not Rendering Properly

If mathematical notation in plots appears incorrectly:

1. Ensure LaTeX is installed: `which latex` or `which pdflatex`
2. Install `cm-super` package for your LaTeX distribution
3. Check that `plt.rcParams['text.usetex'] = True` is set in the notebook

### Quantum Model Files Not Generated

If `.qmod` files are not created:

1. Ensure you have write permissions in the notebook's directory
2. Check that the `write_qmod` cell executed without errors
3. Verify Classiq authentication is configured (if required)

### Execution Errors

If notebooks fail to execute:

1. Verify all dependencies are installed: `pip install -r requirements.txt`
2. Check that you're using the correct Python version (3.9+ recommended)
3. Ensure you're running from the correct working directory
4. For Classiq notebooks, verify API authentication is set up

## Additional Notes

- Some notebooks may produce slightly different results on each run due to
  quantum measurement randomness. This is expected behavior.
- Execution times vary significantly depending on:
  - Number of quantum shots
  - Complexity of the quantum circuit
  - Whether quantum hardware or simulators are used
- For best reproducibility, consider setting random seeds where available
  (e.g., `np.random.seed()` for classical randomness, `random_seed` parameter
  for quantum execution)
