# Synchronize to the Public Repo

This guide explains how to sync changes from this private repository to the public [blueprints](https://github.com/flexaihq/blueprints) repository.

## Prerequisites

- Ensure the public repo is cloned at `../blueprints` on the `main` branch:

  ```sh
  git clone https://github.com/flexaihq/blueprints.git ../blueprints
  cd ../blueprints
  git checkout main
  git pull origin main
  ```

## Steps

1. **Run the synchronization script:**

   ```sh
   ./ci/dev/public-sync.sh
   ```

   This script will copy the relevant changes to your local public repo avoiding the copy of private files.

2. **Review the changes:**

   Navigate to your local public repo to see the diffs:

   ```sh
   cd ../blueprints
   git status
   git diff
   ```

3. **Create a new branch:**

   ```sh
   git checkout -b <your-branch-name>
   ```

4. **Selectively stage changes:**

   Manually pick the diffs that correspond to your changes. It's recommended to commit changes for one blueprint at a time (create multiple PRs if needed).

   ```sh
   git add <specific-files>
   ...
   ```

5. **Push and create a PR:**

   ```sh
   git push origin <your-branch-name>
   ```

   Then open a new Pull Request on the [public repository](https://github.com/flexaihq/blueprints).

## Tips

- Review changes carefully before committing to avoid including private/sensitive information
- Keep commits focused on a single blueprint for easier review
- Write clear commit messages describing the blueprint changes
