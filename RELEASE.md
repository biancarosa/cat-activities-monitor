# Releasing this project

This project uses semantic versioning and conventional commits.
The frontend and the api are versioned together, making it easier to deploy the project. Didn't make sense to bite the complexity of having to version them separately.

## Versioning

We follow [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

## How to generate a new version (changelog and git tag)

1. **Generate new version**:
   ```bash
   # For a regular release:
   npm run release

   # For pre-releases:
   npm run release:alpha  # e.g., 1.0.0-alpha.0
   npm run release:beta   # e.g., 1.0.0-beta.0
   npm run release:rc     # e.g., 1.0.0-rc.0
   ```

   This will:
   - Update version in api/pyproject.toml
   - Update version in frontend/package.json
   - Generate/update CHANGELOG.md
   - Create git tag
   - Create commit with all changes

2. **Review changes**:
   - Check the generated CHANGELOG.md
   - Review the git tag that was created
   - Make sure all changes are properly documented

3. **Push changes**:
   ```bash
   git push --follow-tags origin main
   ```

4. **Create GitHub Release**:
   - Go to GitHub repository
   - Create a new release from the tag
   - Copy the changelog content for the release notes

## Commit Types and Versioning

The following commit types will trigger version bumps:

- `feat`: MINOR version bump
- `fix`: PATCH version bump
- `BREAKING CHANGE`: MAJOR version bump

## Release Workflow

1. **Development**:
   - Make changes following conventional commits
   - Push changes to main branch

2. **Release Preparation**:
   - Ensure all changes are committed
   - Run tests and verify everything works

3. **Release**:
   - Run appropriate release command
   - Push changes and tag
   - Create GitHub release

4. **Post-Release**:
   - Verify the release on GitHub
   - Check that Docker images are built and pushed
   - Update deployment if needed
