# Contributing to openqcp-lab

Thank you for considering contributing to openqcp-lab! Here are some
guidelines to help you get started.

## How to Contribute

1. **Fork the Repository:**
   - Click the "Fork" button at the top right of the repository page.

2. **Clone Your Fork Locally:**
   - Clone your fork to your local machine:
     ```bash
     git clone git@codeberg.org:<YOURCODEBERGUSERNAME>/openqcp-lab.git
     ```
   - Navigate to the cloned directory:
     ```bash
     cd openqcp-lab
     ```

3. **Set Up the Development Environment:**
   - Set up the Python environment using one of the methods described
     in the main README:
     ```bash
     # Option 1: Using the bootstrap script (recommended)
     ./bootstrap

     # Option 2: Using Makefile
     make env

     # Option 3: Manual setup (requires Python 3.12)
     python3.12 -m venv venv
     . venv/bin/activate
     pip install -r requirements.txt
     ```
   - Activate the environment:
     ```bash
     . venv/bin/activate
     ```

4. **Set Up the Upstream Repository:**
   - Add the original repository as an upstream remote to keep your
     fork up-to-date:
     ```bash
     git remote add upstream git@codeberg.org:mkhellat/openqcp-lab.git
     ```

5. **Create a Branch:**
   - Create a new branch for your changes:
     ```bash
     git checkout -b feature/your-feature-name
     ```

6. **Make Your Changes:**
   - Make your changes to the codebase.
   - If adding or modifying a **tutorial notebook**, follow the
     notebook structure guidelines in `NOTEBOOK_TEMPLATE.md`.
   - If adding or modifying a **tool** under `tools/` (e.g.
     `paulikit`), follow that package's own conventions: a
     `pyproject.toml` with full PEP 621 metadata, a `src/<package>/`
     layout, type hints on public functions, and a `tests/` directory
     with real pytest coverage - not ad hoc scripts. See `tools/paulikit`
     for a worked example.
   - Ensure code follows Python style conventions (PEP 8).

7. **Test Your Changes:**
   - Run basic tests to ensure dependencies are working:
     ```bash
     make test
     ```
   - If modifying notebooks, verify they have valid JSON structure
     and can be opened in Jupyter.
   - If modifying a tool under `tools/`, install it in editable mode
     with its test extras and run its test suite, e.g.:
     ```bash
     cd tools/paulikit
     pip install -e ".[test]"
     pytest
     ```

8. **Commit Your Changes:**
   - Commit your changes with a clear and descriptive message:
     ```bash
     git add .
     git commit
     ```
   - It would be quite helpful if your commit messages are
     comprehensive and explanatory of what has been done and adhere
     to the following format with line breaks at 70 characters:
     ```
     [COMMIT MSG TITLE]

     [COMMIT BODY]
     ```

9. **Push Your Changes:**
   - Push your changes to your fork:
     ```bash
     git push origin feature/your-feature-name
     ```

10. **Create a Pull Request:**
    - Go to the fork repository (your repository) on Codeberg.
    - Create a "New Pull Request".
    - Select your fork and the branch you created as the `HEAD`, and
      `openqcp-lab/main` as the `BASE` and then submit the pull
      request.

11. **Update your Fork and Local repos:**
    - Upon successful merge of your changes to `openqcp-lab`, sync your
      fork repo and then update the cloned local repo:
      ```bash
      git checkout main
      git pull (--rebase) origin main
      git branch -D feature/your-feature-name
      ```

## Code Style and Guidelines

- **Python Code:** Follow PEP 8 style guidelines where applicable
- **Notebooks:** Follow the structure outlined in `NOTEBOOK_TEMPLATE.md`
- **Tools (`tools/`):** Each tool is independent, installable software
  in its own right - not tutorial support code, even when it
  originated from a tutorial's need. Each has its own `pyproject.toml`,
  dependencies, and test suite - don't add its dependencies to the
  top-level `requirements.txt`, which is scoped to the tutorial
  notebooks
- **Documentation:** Update README files when adding new features,
  notebooks, or tools
- **Dependencies:** Add new tutorial-notebook dependencies to
  `requirements.txt` with appropriate version constraints; add new
  tool dependencies to that tool's own `pyproject.toml`

## Code of Conduct

In a productive, open, and transparent collaboration, we should be
wary of all sorts of discrimination especially the ones rooted in
nationality, race, and gender.

## Reporting Issues

If you find a bug or have a feature request, please open an issue here.

## Contact

If you have any questions, feel free to reach out to the maintainers.
