#! /usr/bin/bash

DATASET_NAME=$1

echo "Waiting for dataset $DATASET_NAME to be available"
flexai dataset inspect $DATASET_NAME

while true; do
  sleep 10
  INSPECT=$(flexai dataset inspect $DATASET_NAME --json | sed -n '/^{/,$p') # ignore the first line that is not a json
  runtime_status=$(echo "$INSPECT" | jq -r .status.status)
  echo "Runtime status: $runtime_status"

  if [ "$runtime_status" == "failed" ]; then
    flexai dataset inspect $DATASET_NAME
    exit 1
  fi

  if [ "$runtime_status" == "available" ]; then
    flexai dataset inspect $DATASET_NAME
    exit 0
  fi
done
