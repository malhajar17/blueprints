#!/bin/bash
# Description: This script is used to trigger the ML E2E test GitHub workflow of the infra repo.
# Before you run this script, make sure you have a fresh FCS token:
# flexai auth login

FCS_EXPERIMENTS_REV=${FCS_EXPERIMENTS_REV:-main}
ENV=${ENV:-staging}
INFRA_WORKFLOW_REV=${INFRA_WORKFLOW_REV:-main}

INFRA_PATH=$(realpath "$(dirname "$0")/../../../infra")
if [ ! -d "$INFRA_PATH" ]; then
    echo "The infra repository is not found at the expected path: $INFRA_PATH"
    echo "Please ensure that the infra repository is cloned at the same directory level as this project."
    exit 1
fi

(cd "$INFRA_PATH" && ./scripts/run_e2e_ml.py --rev $INFRA_WORKFLOW_REV --fcs-private-experiments-rev $FCS_EXPERIMENTS_REV --env $ENV)
