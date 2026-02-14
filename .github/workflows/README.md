# This directory contains GitHub Actions workflows for the omnisvg-server project

## Workflows

### docker-build.yml
Automatically builds and pushes Docker images to Docker Hub when:
- Code is pushed to main/master branch
- Pull requests are opened (build only, no push)
- Manually triggered via workflow_dispatch

## Required Secrets

Set these in your GitHub repository settings (Settings > Secrets and variables > Actions):

- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: Your Docker Hub access token (create at https://hub.docker.com/settings/security)

## Image Tags

The workflow creates multiple tags:
- `latest` - For the default branch
- `main` or `master` - Branch name
- `sha-abc123` - Short commit SHA
- `pr-123` - For pull requests

## Local Testing

Build locally:
```bash
cd omnisvg-server
docker build -t omnisvg-server:latest .
```
