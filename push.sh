#!/bin/bash
# push.sh - Simple script to push changes to GitHub

# Repo URL
REPO_URL="https://github.com/Mohithan-05/churn-prediction-model.git"
BRANCH="main"

# Initialize repo if not already
if [ ! -d ".git" ]; then
  echo "Initializing Git repo..."
  git init
  git remote add origin $REPO_URL
fi

# Stage all changes
git add .

# Commit with message (use current date/time if none provided)
if [ -z "$1" ]; then
  COMMIT_MSG="Update on $(date)"
else
  COMMIT_MSG="$1"
fi

git commit -m "$COMMIT_MSG"

# Push changes
git push origin $BRANCH
