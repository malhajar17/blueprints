# Running End-to-End (E2E) Tests

This guide explains how to run the End-to-End (E2E) tests for the project using the provided script and `make` command. The tests are executed on the staging environment and target the main branch by default.

## Running the E2E Tests

Before running the E2E tests, ensure the following:

1. **FCS Authentication**:
   - Log in using the `flexai` CLI to generate a fresh access token.

   ```bash
   flexai auth login
   ```

1. Use the following `make` command to trigger the E2E tests:

   ```bash
   make e2e
   ```

   This command executes the script located at `ci/dev/trigger_gh_workflow.sh` to trigger the GitHub workflow for the E2E tests.

## Details

1. The script sets up the following variables:
   - **REVISION**: The branch to test, default is `main`.
   - **ACCESS_TOKEN**: Extracted from `~/.flexai/config.yaml`.
   - **REFRESH_TOKEN**: Extracted from `~/.flexai/config.yaml`.
   - **ENV**: The target environment, default is `staging`.

1. Finally, the script uses these variables to run the GitHub workflow:

## Editing the Script

You can modify the script `ci/dev/trigger_gh_workflow.sh` to customize its behavior:

1. **Branch Selection**:
   - Change the `REVISION` variable to specify a different branch.

2. **Environment**:
   - Update the `ENV` variable to target a different environment (e.g., `prod`).

3. **Access Tokens**:
   - Ensure that valid tokens are present in `~/.flexai/config.yaml`.

## Notes

- If you encounter issues with tokens or authentication, re-run `flexai auth login` to refresh your credentials.
