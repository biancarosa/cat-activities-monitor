# Cat Activities Monitor

## Requirements

- Camera that provides unauthenticated real-time snapshots available at an acessible URL.
- A machine (computer, raspberry pi, any iot device) that can run the api and frontend.

## Setup

### Docker-Compose

1 - The docker-compose.example.yml file contains a base configuration for the docker compose. You can use it as a reference to create your own, or just copy and paste it.
2 - Change the <your-server-ip-address> to the ip address / domain name of the machine that will run the api and frontend.
3 - Create a config.yaml file next to your docker-compose.yml file.
4 - Use the config.yaml.example file as a reference to create your own.
5 - Run the docker compose up command.
6 - Go to <your-server-ip-address>:3000 and you should see the frontend <3

## Development

### Project Structure

This project is composed of two main parts:
- `api/`: FastAPI backend service
- `frontend/`: Next.js frontend application

Each folder has a README.md file with instructions on how to run the project.
You can also run the project using the docker compose file in the root of the project.

### Development Setup

1. **Install dependencies**:
   ```bash
   # Install root dependencies
   npm install
   ```

### Commit Convention

This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification. Each commit message should be structured as follows:

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Types
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation
- `style`: Code style
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Tests
- `build`: Build system
- `ci`: CI configuration
- `chore`: Other changes
- `revert`: Revert commits

#### Scopes
- Frontend: `frontend`, `ui`, `components`, `styles`
- Backend: `api`, `ml`
- Shared: `deps`, `deps-dev`, `docker`, `ci`, `docs`, `config`

#### Examples
```
feat(api): add new detection endpoint
fix(frontend): resolve image loading issue
docs(docs): update API documentation
```

### Versioning

This project uses semantic versioning. See [RELEASE.md](./RELEASE.md) for details on the release process.

