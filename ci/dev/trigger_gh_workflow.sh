# Description: This script is used to trigger the GitHub workflow for the E2E tests.
# Before you run this script, make sure you have a fresh FCS token:
# flexai auth login
# Here the tests are executed on staging environment and on the main branch.
REVISION=main
ACCESS_TOKEN=$(cat ~/.flexai/config.yaml | grep access-token: | cut -d " " -f2)
REFRESH_TOKEN=$(cat ~/.flexai/config.yaml | grep refresh-token: | cut -d " " -f2)
ENV=staging

gh workflow run flexai_e2e_tests.yml --ref $REVISION -f revision=$REVISION -f access-token=$ACCESS_TOKEN -f refresh-token=$REFRESH_TOKEN -f env=$ENV
echo "\nView workflow run at:https://github.com/flexaihq/fcs-experiments-private/actions/workflows/flexai_e2e_tests.yml"
