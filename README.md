# AWS-Trader

AI-powered trading system built with Python and Docker.

## Project Structure

```
mes_ai_trader/
├── docs/         # Documentation files
├── infra/        # Infrastructure as code (Terraform, etc.)
├── models/       # ML model definitions and training code
├── pipelines/    # Data processing pipelines
├── services/     # Microservices
└── tests/        # Test suite
```

## Development

### Prerequisites

- [Python 3.11+](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/#installation)
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd AWS-Trader

# Setup development environment
make dev-setup
```

### Running Tests

```bash
make tests
```

### Deployment

```bash
# Deploy to development environment
make deploy-dev
```

## License

[MIT](LICENSE) 