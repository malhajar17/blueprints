# Running End-to-End (E2E) Tests

This guide explains how to run the End-to-End (E2E) tests for the project using the provided script and `make` command. The tests are executed on the staging environment and target the main branch by default.

## Running the E2E Tests

Before running the E2E tests, ensure the following:

1. **FCS Authentication**:
   - Log in using the `flexai` CLI to generate a fresh access token.

   ```bash
   flexai auth login
   ```

   - Ensure that valid tokens are present in `~/.flexai/config.yaml`.

1. Ensure that the `infra` repository is cloned at the same directory level as this project, so it is accessible at `../infra`.

1. Use the following `make` command to trigger the E2E tests:

   ```bash
   make e2e FCS_EXPERIMENTS_REV=my-fcs-rev-containing-e2e ENV=staging
   ```

   This command trigger the ML e2e GitHub workflow (infra repo) with the tests from this repo revision `my-fcs-rev-containing-e2e` on staging env.

## Running a single test directly

When developing or debugging the tests, it may be useful to run a single test directly without using the CI workflows.
Using this method, the test will run on `staging` or `production` depending on which flexai CLI is configured locally. The test won't appear in any CI jobs.

To execute a test directly a simple trampoline is provided, for example:

```bash
./ci/e2e_tests/gpt2-simple.py
```
