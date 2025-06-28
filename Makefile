.PHONY: tests

-include .env
export

package-build:
	rm -rf dist/*
	uv build --no-sources

package-publish:
	uv publish --token ${UV_PUBLISH_TOKEN}

package: package-build package-publish

####
# project commands
#
# `uv add` to add new package
# `uv add --dev` to add new package as a dev dependency
####
# commands to run inside docker container
linting:
	uv run ruff check src/flex_evals/ --fix --unsafe-fixes
	uv run ruff check tests/ --fix --unsafe-fixes

unittests:
	# pytest tests
	uv run coverage run -m pytest --durations=0 tests
	uv run coverage html

tests: linting unittests
