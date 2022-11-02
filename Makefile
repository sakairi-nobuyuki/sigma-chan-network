SRCS = db_oparators sigma_chan_db tweet_getter tests
POETRY_PREFIX = poetry run


format:
	@for SRC in $(SRCS); do $(POETRY_PREFIX) black $$SRC --config pyproject.toml; done
	@for SRC in $(SRCS); do $(POETRY_PREFIX) isort $$SRC --profile black; done
