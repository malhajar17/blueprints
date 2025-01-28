#! /usr/bin/bash

TRAINING_NAME=$1

echo "Waiting for training $TRAINING_NAME to finish"
flexai training inspect $TRAINING_NAME

while true; do
  sleep 10
  INSPECT=$(flexai training inspect $TRAINING_NAME -j | sed -n '/^{/,$p') # ignore the first line that is not a json
  runtime_status=$(echo "$INSPECT" | jq -r .runtime.status)
  echo "Runtime status: $runtime_status"

  if [ "$runtime_status" == "rejected" ]; then
    flexai training inspect $TRAINING_NAME

    # use timeout to avoid infinite loop on logs command
    timeout 30 flexai training logs $TRAINING_NAME
    exit 1
  fi

  if [ "$runtime_status" == "failed" ]; then
    flexai training inspect $TRAINING_NAME

    # use timeout to avoid infinite loop on logs command
    timeout 30 flexai training logs $TRAINING_NAME
    exit 1
  fi

  if [ "$runtime_status" == "interrupted" ]; then
    flexai training inspect $TRAINING_NAME

    # use timeout to avoid infinite loop on logs command
    timeout 30 flexai training logs $TRAINING_NAME
    exit 1
  fi

  if [ "$runtime_status" == "succeeded" ]; then
    flexai training inspect $TRAINING_NAME

    timeout 30 flexai training logs $TRAINING_NAME || echo "training succeeded"
    exit 0
  fi
done
