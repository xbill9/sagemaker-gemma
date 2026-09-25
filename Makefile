export AWS_PAGER :=
# Makefile for sagemaker-gemma: Gemma on a SageMaker endpoint via the aws CLI.

-include .env
export

PY := python3
PROMPT ?= Explain what a SageMaker endpoint is in two sentences.

.PHONY: all install run test lint format image quota role deploy status wait logs invoke list destroy clean

all: install test lint

install: ## Install dependencies into the system python3
	python3 -m pip install -r requirements.txt

run: ## Run the MCP server on stdio
	@python3 server.py

test: ## Offline tests (the aws CLI is faked)
	@python3 -m unittest discover -s tests -v

lint:
	@ruff check .
	@ruff format --check .

format:
	@ruff format .
	@ruff check --fix .

# --- AWS (each target is one or more aws CLI calls; see sm.py) ---------------

image: ## Newest SageMaker vLLM container image in AWS_REGION
	@$(PY) sm.py image

quota: ## Endpoint quota for INSTANCE_TYPE (0 means request an increase)
	@$(PY) sm.py quota

role: ## Create the SageMaker execution role if missing
	@$(PY) sm.py role

deploy: ## Create model, endpoint config and endpoint (starts billing)
	@$(PY) sm.py deploy

status:
	@$(PY) sm.py status

wait: ## Block until the endpoint is InService (or fails)
	@aws sagemaker wait endpoint-in-service --endpoint-name $(ENDPOINT_NAME) --region $(AWS_REGION)
	@$(PY) sm.py status

logs:
	@$(PY) sm.py logs

invoke: ## Send PROMPT to the endpoint
	@$(PY) sm.py invoke "$(PROMPT)"

list:
	@$(PY) sm.py list

destroy: ## Delete endpoint, endpoint config and model (stops billing)
	@$(PY) sm.py destroy

clean:
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@rm -rf .ruff_cache
