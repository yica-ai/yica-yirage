# GitHub Repository Setup Guide

This guide will help you set up the YICA-Mirage repository on GitHub and prepare it for public release.

## 🚀 Quick Setup

### Prerequisites

1. **GitHub CLI** installed:
   ```bash
   # macOS
   brew install gh
   
   # Linux (Ubuntu/Debian)
   sudo apt install gh
   
   # Linux (RHEL/CentOS/Fedora)
   sudo yum install gh
   ```

2. **GitHub authentication**:
   ```bash
   gh auth login
   ```

3. **Repository access** to the [yica-ai organization](https://github.com/yica-ai)

### Automated Setup

Run the automated setup script:

```bash
./scripts/release/create-github-repo.sh
```

This script will:
- ✅ Create the repository `yica-ai/yica-mirage`
- ✅ Configure repository settings and topics
- ✅ Set up branch protection rules
- ✅ Create essential files (LICENSE, CONTRIBUTING.md, CHANGELOG.md)
- ✅ Push initial code to GitHub
- ✅ Create the first release (v1.0.0)
- ✅ Configure repository features (Issues, Projects, Wiki, Discussions)

## 📋 Manual Setup Steps

If you prefer manual setup or need to troubleshoot:

### 1. Create Repository

```bash
gh repo create yica-ai/yica-mirage \
    --description "YICA-Mirage: High-Performance AI Computing Optimization Framework for In-Memory Computing Architecture" \
    --homepage "https://yica.ai" \
    --public
```

### 2. Configure Repository Settings

```bash
# Enable features
gh repo edit yica-ai/yica-mirage \
    --enable-issues \
    --enable-projects \
    --enable-wiki \
    --enable-discussions

# Add topics
gh repo edit yica-ai/yica-mirage \
    --add-topic "ai" \
    --add-topic "optimization" \
    --add-topic "compiler" \
    --add-topic "triton" \
    --add-topic "yica" \
    --add-topic "mirage" \
    --add-topic "deep-learning" \
    --add-topic "in-memory-computing"
```

### 3. Set Up Git Repository

```bash
# Initialize git (if not already done)
git init

# Add remote
git remote add origin https://github.com/yica-ai/yica-mirage.git

# Configure git
git config --local user.name "YICA Team"
git config --local user.email "contact@yica.ai"

# Stage and commit files
git add .
git commit -m "Initial release: YICA-Mirage v1.0.0"

# Push to GitHub
git branch -M main
git push -u origin main
```

### 4. Create Release

```bash
# Create and push tag
git tag -a v1.0.0 -m "YICA-Mirage v1.0.0 - Initial Release"
git push origin v1.0.0

# Create GitHub release
gh release create v1.0.0 \
    --repo yica-ai/yica-mirage \
    --title "YICA-Mirage v1.0.0" \
    --notes-file release-notes.md \
    --latest
```

## 🔒 Repository Secrets

After repository creation, add these secrets in GitHub repository settings:

Navigate to: `https://github.com/yica-ai/yica-mirage/settings/secrets/actions`

### Required Secrets

1. **PYPI_API_TOKEN**
   - Description: PyPI API token for package publishing
   - How to get: Visit https://pypi.org/manage/account/token/

2. **DOCKER_USERNAME**
   - Description: Docker Hub username
   - Value: Your Docker Hub username

3. **DOCKER_PASSWORD**
   - Description: Docker Hub password or access token
   - How to get: Visit https://hub.docker.com/settings/security

### Optional Secrets

4. **CODECOV_TOKEN** (for code coverage)
5. **READTHEDOCS_TOKEN** (for documentation)

## 🛠️ Post-Setup Configuration

### 1. Enable GitHub Actions

The repository includes a comprehensive CI/CD pipeline in `.github/workflows/release.yml`. This will automatically:
- Build and test on multiple platforms
- Create wheels for all supported platforms
- Publish to PyPI on releases
- Build and push Docker images
- Create GitHub releases with artifacts

### 2. Set Up Package Repositories

#### PyPI
- The package will be automatically published on releases
- Ensure `PYPI_API_TOKEN` secret is configured

#### Homebrew
- Submit the formula in `scripts/packaging/yica-mirage.rb` to Homebrew
- Or create a custom tap: `yica-ai/homebrew-tap`

#### APT Repository
- Set up a Debian package repository
- Configure GPG signing keys
- Update the installation script URLs

#### Docker Hub
- Create repositories: `yicaai/yica-mirage`
- Configure automated builds

### 3. Documentation Hosting

#### Read the Docs
1. Visit https://readthedocs.org/
2. Import the GitHub repository
3. Configure build settings
4. Set up custom domain: `yica-mirage.readthedocs.io`

#### GitHub Pages (Alternative)
```bash
# Enable GitHub Pages for documentation
gh repo edit yica-ai/yica-mirage --enable-pages --pages-source gh-pages
```

## 🚦 Verification

After setup, verify everything works:

1. **Repository Access**: Visit https://github.com/yica-ai/yica-mirage
2. **Release**: Check https://github.com/yica-ai/yica-mirage/releases
3. **CI/CD**: Verify GitHub Actions are running
4. **Topics**: Ensure repository topics are visible
5. **Features**: Check Issues, Projects, Wiki, Discussions are enabled

## 🔄 Next Steps

1. **Configure branch protection rules** for the main branch
2. **Set up code review requirements**
3. **Create project boards** for issue tracking
4. **Write documentation** and examples
5. **Set up monitoring** and analytics
6. **Plan release cycles** and versioning strategy

## 📞 Support

If you encounter issues during setup:

1. **Check GitHub CLI authentication**: `gh auth status`
2. **Verify repository permissions**: Ensure you have admin access to yica-ai organization
3. **Review error messages**: Most issues are related to permissions or authentication
4. **Contact support**: Create an issue or contact the team

---

**Repository URL**: https://github.com/yica-ai/yica-mirage
**Setup Script**: `./scripts/release/create-github-repo.sh`
**Status**: Ready for public release 🚀 