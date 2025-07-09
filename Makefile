# Lints the codebase.
.PHONY: lint
lint:
	pre-commit run --all-files

# run e2e tests triggering GH actions
.PHONY: e2e
e2e:
	FCS_EXPERIMENTS_REV=$(FCS_EXPERIMENTS_REV) ENV=$(ENV) INFRA_WORKFLOW_REV=$(INFRA_WORKFLOW_REV) ./ci/dev/trigger_gh_workflow.sh

# diff with specified branch without test related files
.PHONY: diff-branch
diff-branch:
	@if [ -z "$(b)" ]; then \
		echo "Usage: make diff-branch b=<branch_name>"; \
		exit 1; \
	fi
	git diff $(b) -- ":(exclude).github/*" ":(exclude)Makefile" ":(exclude)REAME-e2e.md" ":(exclude).pre-commit*" ":(exclude)ci/*"

# Default values for e2e
FCS_EXPERIMENTS_REV ?= main
ENV ?= staging
INFRA_WORKFLOW_REV ?= main
