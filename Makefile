SRCS = sigma_chan_network tests
TEST_TARGET = sigma_chan_network
POETRY_PREFIX = poetry run
LINTER_IGNORE = scripts
PATH_TEST_COV_BADGE = pics/cov.svg

format:
	@for SRC in $(SRCS); do $(POETRY_PREFIX) black $$SRC --config pyproject.toml; done
	@for SRC in $(SRCS); do $(POETRY_PREFIX) isort $$SRC --profile black; done

lint:
	@for SRC in $(SRCS); do $(POETRY_PREFIX) pylint --fail-under 6.0 $$SRC --exit-zero ; done
test:
	$(POETRY_PREFIX) pytest --cov=$(TEST_TARGET) --cov-fail-under 60
	$(POETRY_PREFIX) coverage-badge -fo $(PATH_TEST_COV_BADGE)