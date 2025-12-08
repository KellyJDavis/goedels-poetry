# Publishing Guide

This document describes how to publish new versions of `goedels-poetry` to PyPI.

## Prerequisites

1. **PyPI Account**: You must have an account on [pypi.org](https://pypi.org)
2. **PyPI API Token**: Generate a token at https://pypi.org/manage/account/token/
3. **GitHub Secret**: Add the token to your GitHub repository

## Setting Up the GitHub Secret

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `PYPI_API_TOKEN`
5. Value: Paste your PyPI API token (starts with `pypi-`)
6. Click **Add secret**

## Publishing a New Release

### Automatic Publishing (Recommended)

The repository is configured to automatically publish to PyPI when you push a version tag:

```bash
# 1. Update the version in pyproject.toml
# Edit pyproject.toml and change: version = "0.0.10"

# 2. Update the version in goedels_poetry/__init__.py
# Edit goedels_poetry/__init__.py and change: __version__ = "0.0.10"

# 3. Synchronize version in uv.lock with version in pyproject.toml
uv lock

# 4. Update CHANGELOG.md with the new version
# Add a new section for the version

# 5. Commit your changes
git add pyproject.toml goedels_poetry/__init__.py uv.lock CHANGELOG.md
git commit -m "Bump version to 0.0.10"

# 6. Create and push the tag
git tag v0.0.10
git push origin main
git push origin v0.0.10
```

The GitHub Action will automatically:
- ✅ Verify the tag matches the version in `pyproject.toml`
- ✅ Build the package
- ✅ Check the package with twine
- ✅ Publish to PyPI
- ✅ Create a GitHub Release with the distribution files

### Manual Publishing

If you prefer to publish manually:

```bash
# Build the package
make clean-build
make build

# Check the package
uvx twine check dist/*

# Upload to PyPI
uvx twine upload dist/*
# You'll be prompted for your PyPI username and password/token
```

### Testing on TestPyPI First

To test the publishing process without affecting the real PyPI:

```bash
# Build
make clean-build
make build

# Upload to TestPyPI
uvx twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ goedels-poetry
```

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version (1.0.0): Incompatible API changes
- **MINOR** version (0.1.0): New functionality, backwards compatible
- **PATCH** version (0.0.1): Bug fixes, backwards compatible

## Checklist Before Publishing

- [ ] Version updated in `pyproject.toml`
- [ ] Version updated in `goedels_poetry/__init__.py`
- [ ] CHANGELOG.md updated with changes
- [ ] All tests passing (`make test`)
- [ ] Package builds successfully (`make build`)
- [ ] Changes committed to main branch
- [ ] Tag created and pushed

## Troubleshooting

### "Version mismatch" error
The tag version must match the version in `pyproject.toml`. If you tagged `v0.0.10` but `pyproject.toml` has `0.0.9`, the workflow will fail.

### "File already exists" error
You cannot re-upload the same version to PyPI. You must increment the version number.

### Authentication error
Ensure your `PYPI_API_TOKEN` secret is correctly set in GitHub and that the token has the necessary permissions.
