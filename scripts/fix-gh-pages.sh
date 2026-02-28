#!/bin/bash
set -e

echo "🔧 Fixing gh-pages branch to contain only built documentation..."

# Store current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $CURRENT_BRANCH"

# Ensure we're in the repository root
cd "$(git rev-parse --show-toplevel)"

# Fetch latest gh-pages
echo "📥 Fetching gh-pages branch..."
git fetch origin gh-pages

# Create a temporary directory for the clean gh-pages
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Clone gh-pages to temp directory
echo "📦 Cloning gh-pages branch..."
git clone --branch gh-pages --single-branch . "$TEMP_DIR/gh-pages-repo"
cd "$TEMP_DIR/gh-pages-repo"

# Check if there are any version directories (v*.*/goedels-poetry or main/goedels-poetry)
echo "🔍 Looking for existing documentation directories..."
FOUND_DOCS=false
if compgen -G "v*/goedels-poetry" > /dev/null 2>&1; then
    FOUND_DOCS=true
    echo "  ✓ Found version directories"
fi
if [ -d "main/goedels-poetry" ]; then
    FOUND_DOCS=true
    echo "  ✓ Found main directory"
fi

# Save any existing documentation directories
DOC_BACKUP="$TEMP_DIR/docs-backup"
mkdir -p "$DOC_BACKUP"

if [ "$FOUND_DOCS" = true ]; then
    echo "💾 Backing up existing documentation..."
    # Copy all version directories
    if compgen -G "v*" > /dev/null 2>&1; then
        cp -r v* "$DOC_BACKUP/" 2>/dev/null || true
    fi
    # Copy main directory
    if [ -d "main" ]; then
        cp -r main "$DOC_BACKUP/" 2>/dev/null || true
    fi
    # Copy index.html if it exists
    if [ -f "index.html" ]; then
        cp index.html "$DOC_BACKUP/" 2>/dev/null || true
    fi
fi

# Remove everything except .git
echo "🧹 Cleaning gh-pages branch..."
find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +

# Restore documentation directories if they existed
if [ "$FOUND_DOCS" = true ]; then
    echo "♻️  Restoring documentation..."
    cp -r "$DOC_BACKUP"/* . 2>/dev/null || true
else
    echo "⚠️  No existing documentation found - will be created on next deployment"
    # Create a placeholder index.html
    cat > index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Gödel's Poetry Documentation</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      max-width: 800px;
      margin: 100px auto;
      padding: 20px;
      text-align: center;
    }
    h1 { color: #333; }
    p { color: #666; line-height: 1.6; }
    a { color: #0366d6; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <h1>Gödel's Poetry Documentation</h1>
  <p>Documentation is being generated. Please check back shortly.</p>
  <p>Visit the <a href="https://github.com/KellyJDavis/goedels-poetry">GitHub repository</a> for more information.</p>
</body>
</html>
EOF
fi

# Commit the cleaned branch
echo "💾 Committing cleaned gh-pages branch..."
git add -A
if git commit -m "Clean gh-pages branch - remove source code, keep only built docs"; then
    echo "✅ Changes committed"
else
    echo "ℹ️  No changes to commit"
fi

# Push to remote
echo "🚀 Pushing to origin/gh-pages..."
read -p "Push cleaned gh-pages branch to GitHub? (yes/no): " -r
if [[ $REPLY =~ ^[Yy]([Ee][Ss])?$ ]]; then
    git push origin gh-pages --force
    echo "✅ gh-pages branch has been cleaned and pushed!"
else
    echo "⏸️  Skipped push. You can manually push later with:"
    echo "   cd $TEMP_DIR/gh-pages-repo && git push origin gh-pages --force"
fi

# Return to original directory and branch
cd "$(git rev-parse --show-toplevel)"
git checkout "$CURRENT_BRANCH" 2>/dev/null || true

echo ""
echo "✨ Done! Next steps:"
echo "1. Trigger a workflow run to rebuild documentation:"
echo "   - Push a commit to main, OR"
echo "   - Manually trigger the 'Deploy Documentation' workflow, OR"
echo "   - Create a new tag to trigger tag documentation deployment"
echo ""
echo "2. Once the workflow completes, your documentation will be available at:"
echo "   https://kellyjdavis.github.io/goedels-poetry/"
echo ""
echo "Temporary files are in: $TEMP_DIR"
echo "You can safely delete this after verifying the fix works."
