module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'scope-enum': [
      2,
      'always',
      [
        // Frontend scopes
        'frontend',
        'ui',
        'components',
        'styles',
        // Backend scopes
        'api',
        'ml',
        // Shared scopes
        'deps',
        'deps-dev',
        'docker',
        'ci',
        'docs',
        'config',
        'release'
      ]
    ],
    'scope-empty': [0, 'never'],
    'type-enum': [
      2,
      'always',
      [
        'feat',     // New feature
        'fix',      // Bug fix
        'docs',     // Documentation changes
        'style',    // Code style changes (formatting, etc)
        'refactor', // Code changes that neither fix bugs nor add features
        'perf',     // Performance improvements
        'test',     // Adding or fixing tests
        'build',    // Changes to build system or dependencies
        'ci',       // Changes to CI configuration
        'chore',    // Other changes that don't modify src or test files
        'revert'    // Reverts a previous commit
      ]
    ]
  }
}; 