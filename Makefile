# Lints the codebase.
.PHONY: lint
lint:
	pre-commit run --all-files

# run e2e tests locally using act
.PHONY: local-e2e
local-e2e:
	./ci/dev/run_local_act.sh

# run e2e tests triggering GH actions
.PHONY: e2e
e2e:
	./ci/dev/trigger_gh_workflow.sh

# push ci datasets
.PHONY: push-ci-datasets
push-ci-datasets:
	./ci/dev/push-ci-datasets.sh

# diff with specified branch without test related files
.PHONY: diff-branch
diff-branch:
	@if [ -z "$(b)" ]; then \
		echo "Usage: make diff-branch b=<branch_name>"; \
		exit 1; \
	fi
	git diff $(b) -- ":(exclude).github/*" ":(exclude)Makefile" ":(exclude)REAME-e2e.md" ":(exclude).pre-commit*" ":(exclude)ci/*"
