SRCS = typical_cnn tests
POETRY_PREFIX = poetry run
LINTER_IGNORE = scripts

format:
	@for SRC in $(SRCS); do $(POETRY_PREFIX) black $$SRC --config pyproject.toml; done
	@for SRC in $(SRCS); do $(POETRY_PREFIX) isort $$SRC --profile black; done

lint:
	@for SRC in $(SRCS); do $(POETRY_PREFIX) pylint --fail-under 6.0 $$SRC --exit-zero ; done