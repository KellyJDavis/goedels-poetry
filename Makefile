.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry .

.PHONY: test
test: ## Test the code with pytest (excludes integration tests)
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --ignore=tests/test_kimina_agents.py --cov --cov-config=pyproject.toml --cov-report=xml

.PHONY: test-integration
test-integration: ## Run integration tests (requires Lean installation)
	@echo "🚀 Running integration tests (requires Lean)"
	@if ! command -v lake > /dev/null; then \
		echo "❌ Error: Lean (lake) is not installed. Run 'cd kimina-lean-server && bash setup.sh' first."; \
		exit 1; \
	fi
	@echo "📦 Installing server dependencies..."
	@cd kimina-lean-server && uv pip install -q prisma fastapi uvicorn psutil google-cloud-logging
	@echo "🔧 Generating Prisma types..."
	@cd kimina-lean-server && uv run prisma generate
	@echo "🧪 Running integration tests..."
	@uv run python -m pytest tests/test_kimina_agents.py -v --cov --cov-config=pyproject.toml --cov-append --cov-report=xml

.PHONY: test-all
test-all: ## Run all tests including integration tests
	@echo "🚀 Running all tests"
	@$(MAKE) test
	@$(MAKE) test-integration

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: publish
publish: ## Publish a release to PyPI.
	@echo "🚀 Publishing."
	@uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
