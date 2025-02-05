# Description: This script is used to run the e2e tests locally using act.
# Before you run this script, make sure you have a fresh FCS token:
# flexai auth login
# Here the tests are executed on staging environment and on the main branch.
REVISION=main
ACCESS_TOKEN=$(cat ~/.flexai/config.yaml | grep access-token: | cut -d " " -f2)
REFRESH_TOKEN=$(cat ~/.flexai/config.yaml | grep refresh-token: | cut -d " " -f2)
ENV=staging

act --container-architecture linux/amd64 -W .github/workflows/flexai_e2e_tests.yml --input revision=$REVISION --input access-token=$ACCESS_TOKEN --input refresh-token=$REFRESH_TOKEN --input env=$ENV
