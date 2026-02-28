# Fixing GitHub Pages Documentation

## Problem

The `gh-pages` branch contains source code instead of built documentation, causing versioned documentation URLs (like `/v2.0.5/goedels-poetry/`) to return 404 errors.

## Solution Options

### Option 1: Run the Automated Workflow (Recommended)

This workflow will clean the gh-pages branch and rebuild all documentation for all tags:

1. Go to: https://github.com/KellyJDavis/goedels-poetry/actions/workflows/rebuild-all-docs.yml
2. Click "Run workflow" button
3. Click the green "Run workflow" button in the dropdown
4. Wait for the workflow to complete (~5-10 minutes depending on number of tags)
5. Verify the fix by visiting:
   - https://kellyjdavis.github.io/goedels-poetry/ (should redirect to latest)
   - https://kellyjdavis.github.io/goedels-poetry/v2.0.5/goedels-poetry/
   - https://kellyjdavis.github.io/goedels-poetry/v2.0.6/goedels-poetry/
   - https://kellyjdavis.github.io/goedels-poetry/main/goedels-poetry/

### Option 2: Run the Script Locally

If you prefer to fix it locally:

```bash
# Run the fix script
./scripts/fix-gh-pages.sh

# When prompted, confirm you want to push
# Then manually trigger a documentation rebuild by either:
# - Pushing a commit to main
# - Creating a new tag
# - Running the "Rebuild All Documentation" workflow
```

## What Was Wrong?

1. The `gh-pages` branch contained source code (`.github/`, `goedels_poetry/`, etc.)
2. When the deployment workflows ran, they:
   - Cloned gh-pages (getting source code)
   - Added the new version directory (e.g., `v2.0.5/goedels-poetry/`)
   - Created an artifact with the mixed content
   - Deployed that artifact to GitHub Pages
3. GitHub Pages serves the artifact, but the presence of source files interferes with proper routing

## Verification

After running the fix, verify these URLs work:

- ✅ https://kellyjdavis.github.io/goedels-poetry/ (root redirect)
- ✅ https://kellyjdavis.github.io/goedels-poetry/main/goedels-poetry/
- ✅ https://kellyjdavis.github.io/goedels-poetry/v2.0.6/goedels-poetry/
- ✅ https://kellyjdavis.github.io/goedels-poetry/v2.0.5/goedels-poetry/

## Future Deployments

After this fix, your existing workflows will work correctly:
- `deploy-docs.yml` - Deploys main branch documentation
- `deploy-tag-docs.yml` - Deploys tag documentation
- `rebuild-all-docs.yml` - Rebuilds everything (manual trigger only)

The gh-pages branch will now contain ONLY:
- `index.html` (root redirect)
- `main/goedels-poetry/` (main branch docs)
- `v*/goedels-poetry/` (versioned docs for each tag)
