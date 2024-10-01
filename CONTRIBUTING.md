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

3. **Set Up the Upstream Repository:**
   - Add the original repository as an upstream remote to keep your
     fork up-to-date:
     ```bash
     git remote add upstream git@codeberg.org:mkhellat/openqcp-lab.git
     ```

4. **Create a Branch:**
   - Create a new branch for your changes:
     ```bash
     git checkout -b feature/your-feature-name
     ```

5. **Make Your Changes:**
   - Make your changes to the codebase.

6. **Commit Your Changes:**
   - Commit your changes with a clear and descriptive message:
     ```bash
     git add .
     git commit
     ```
   - It would be quite helpful if your commit message are
     comprehensive and explanatory of what has been done and adheres
     to the following format with line breaks at 70 characters.
     ```
     [COMMIT MSG TITLE]

     [COMMIT BODY]
     ```

7. **Push Your Changes:**
   - Push your changes to your fork :
     ```bash
     git push origin feature/your-feature-name
     ```

8. **Create a Pull Request:**
   - Go to the fork repository (your repository) on Codeberg.
   - Create a "New Pull Request".
   - Select your fork and the branch you created as the `HEAD`, and
     `openqcp-lab/main` as the `BASE` and then submit the pull
     request.

9. **Update your Fork and Local repos**
   - Upon successful merge of your changes to `openqcp-lab`, sync your
     fork repo and then update the cloned local repo :
     ```bash
     git checkout main
     git pull (--rebase) origin main
     git branch -D feature/your-feature-name
     ```

## Code of Conduct

In a productive, open, and transparent collaboration, we should be
wary of all sorts of discrimination especially the ones rooted in
nationality, race, and gender.

## Reporting Issues

If you find a bug or have a feature request, please open an issue here.

## Contact

If you have any questions, feel free to reach out to the maintainers.
