# DigiPal Deployment Guide

## Overview

This guide covers deploying DigiPal in various environments, from local development to production cloud deployments. DigiPal is designed to be flexible and can run on different platforms with appropriate configuration.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Deployment](#docker-deployment)
3. [HuggingFace Spaces](#huggingface-spaces)
4. [Cloud Deployment](#cloud-deployment)
5. [Configuration](#configuration)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/your-org/digipal.git
cd digipal

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run application
python launch_digipal.py
```

Access at: http://localhost:7860

### Docker Quick Start

```bash
# Build and run
docker build -t digipal .
docker run -p 7860:7860 digipal

# Or use docker-compose
docker-compose up
```

## Performance Testing

DigiPal includes comprehensive performance testing to validate scalability and real-world usage:

### Scalability Benchmarks
- **Large Scale Pet Creation**: Tests creating 100+ pets with performance validation
- **Database Load Testing**: Concurrent database operations with 5 worker threads
- **Memory Efficiency**: Multi-pet memory usage tracking with cleanup validation
- **Real-World Scenarios**: Typical user sessions and long-running stability tests

### Performance Validation
```bash
# Run performance benchmarks
python -m pytest tests/test_performance_benchmarks.py::TestScalabilityBenchmarks -v -s

# Run real-world scenario tests
python -m pytest tests/test_performance_benchmarks.py::TestRealWorldScenarios -v -s

# Full performance suite
python -m pytest tests/test_performance_benchmarks.py -v -s
```

### Key Performance Metrics
- **Pet Creation**: 100 pets in <10 seconds (0.1s average per pet)
- **Database Operations**: <100ms average, <500ms maximum under load
- **Memory Usage**: <10MB per active pet with efficient cleanup
- **Concurrent Users**: 95%+ success rate with multiple simultaneous users
- **Long Sessions**: Stable performance over 100+ interactions with <50MB memory growth

### Load Testing Scenarios
- **High Volume Interactions**: 100+ interactions with 95%+ success rate
- **Sustained Load**: Multiple users over extended periods with performance monitoring
- **Memory Stability**: Extended operation validation with automatic cleanup
- **Concurrent Database Access**: Multi-threaded database operations with performance tracking

## Docker Deployment

### Basic Docker Setup

#### 1. Build Image

```bash
docker build -t digipal:latest .
```

#### 2. Run Container

```bash
docker run -d \
  --name digipal \
  -p 7860:7860 \
  -v $(pwd)/assets:/app/assets \
  -v $(pwd)/logs:/app/logs \
  -e DIGIPAL_ENV=production \
  -e HF_TOKEN=your_token_here \
  digipal:latest
```

#### 3. Environment Variables

```bash
# Required
-e HF_TOKEN=your_huggingface_token

# Optional
-e DIGIPAL_ENV=production
-e DIGIPAL_LOG_LEVEL=INFO
-e GRADIO_SERVER_PORT=7860
-e DIGIPAL_SECRET_KEY=your_secret_key
```

### Docker Compose

#### Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  digipal:
    build: .
    ports:
      - "7860:7860"
    environment:
      - DIGIPAL_ENV=production
      - HF_TOKEN=${HF_TOKEN}
      - DIGIPAL_SECRET_KEY=${SECRET_KEY}
    volumes:
      - digipal_data:/app/data
      - digipal_logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7860/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - digipal
    restart: unless-stopped

volumes:
  digipal_data:
  digipal_logs:
```

#### Run Production

```bash
# Set environment variables
export HF_TOKEN=your_token
export SECRET_KEY=your_secret

# Deploy
docker-compose -f docker-compose.prod.yml up -d
```

### Multi-Stage Build (Optimized)

```dockerfile
# Dockerfile.optimized
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim

# Copy installed packages
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash digipal
RUN chown -R digipal:digipal /app
USER digipal

EXPOSE 7860
CMD ["python", "launch_digipal.py"]
```

## HuggingFace Spaces

DigiPal is optimized for deployment on HuggingFace Spaces.

### 1. Create Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose "Docker" as SDK
4. Set visibility (Public/Private)

### 2. Configure Space

#### README.md Header

```yaml
---
title: DigiPal
emoji: 🎮
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---
```

#### Dockerfile for Spaces

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p assets/images assets/backups logs

# Set environment for Spaces
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV DIGIPAL_ENV=production

EXPOSE 7860

CMD ["python", "launch_digipal.py"]
```

### 3. Environment Secrets

In your Space settings, add secrets:
- `HF_TOKEN`: Your HuggingFace token
- `DIGIPAL_SECRET_KEY`: Random secret key

### 4. Deploy

```bash
# Clone your space
git clone https://huggingface.co/spaces/your-username/digipal
cd digipal

# Add files
git add .
git commit -m "Initial DigiPal deployment"
git push
```

## Cloud Deployment

### AWS Deployment

#### Using ECS (Elastic Container Service)

1. **Build and Push Image**

```bash
# Build for AWS
docker build -t digipal:aws .

# Tag for ECR
docker tag digipal:aws 123456789012.dkr.ecr.us-west-2.amazonaws.com/digipal:latest

# Push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/digipal:latest
```

2. **ECS Task Definition**

```json
{
  "family": "digipal",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "digipal",
      "image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/digipal:latest",
      "portMappings": [
        {
          "containerPort": 7860,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DIGIPAL_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "HF_TOKEN",
          "valueFrom": "arn:aws:secretsmanager:us-west-2:123456789012:secret:digipal/hf-token"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/digipal",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Using Lambda (Serverless)

```python
# lambda_handler.py
import json
from mangum import Mangum
from launch_digipal import create_app

app = create_app()
handler = Mangum(app)

def lambda_handler(event, context):
    return handler(event, context)
```

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT_ID/digipal
gcloud run deploy digipal \
  --image gcr.io/PROJECT_ID/digipal \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 7860 \
  --memory 2Gi \
  --cpu 1 \
  --set-env-vars DIGIPAL_ENV=production
```

### Azure Container Instances

```bash
# Deploy to ACI
az container create \
  --resource-group myResourceGroup \
  --name digipal \
  --image digipal:latest \
  --cpu 1 \
  --memory 2 \
  --ports 7860 \
  --environment-variables DIGIPAL_ENV=production \
  --secure-environment-variables HF_TOKEN=your_token
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DIGIPAL_ENV` | Environment (development/testing/production) | development | No |
| `HF_TOKEN` | HuggingFace API token | None | Yes |
| `GRADIO_SERVER_NAME` | Server bind address | 0.0.0.0 | No |
| `GRADIO_SERVER_PORT` | Server port | 7860 | No |
| `DIGIPAL_LOG_LEVEL` | Logging level | INFO | No |
| `DIGIPAL_SECRET_KEY` | Secret key for sessions | Generated | No |
| `DIGIPAL_DB_PATH` | Database file path | digipal.db | No |

### Configuration Files

#### Production Config

```python
# config/production.py
import os

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/digipal.db')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
SECRET_KEY = os.getenv('DIGIPAL_SECRET_KEY')
DEBUG = False
TESTING = False

# AI Model settings
QWEN_MODEL = "Qwen/Qwen3-0.6B"
KYUTAI_MODEL = "kyutai/stt-2.6b-en_fr-trfs"
FLUX_MODEL = "black-forest-labs/FLUX.1-dev"

# Performance settings
CACHE_SIZE_MB = 1024
MAX_CONCURRENT_USERS = 500
BACKGROUND_UPDATE_INTERVAL = 60
```

### SSL/TLS Configuration

#### Nginx Reverse Proxy

```nginx
# nginx.conf
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://digipal:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Monitoring

### Health Checks

DigiPal includes built-in health checks:

```bash
# Check application health
curl http://localhost:7860/health

# Check detailed status
curl http://localhost:7860/health/detailed
```

### Prometheus Metrics

Enable metrics collection:

```bash
# Start with monitoring
docker-compose --profile monitoring up -d
```

Access metrics:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Logging

#### Structured Logging

```python
# Enable structured logging
DIGIPAL_LOG_LEVEL=DEBUG
ENABLE_STRUCTURED_LOGGING=true
```

#### Log Aggregation

```yaml
# docker-compose with logging
version: '3.8'
services:
  digipal:
    # ... other config
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Alerting

#### Basic Alerts

```bash
# Monitor container health
docker run -d \
  --name watchtower \
  -v /var/run/docker.sock:/var/run/docker.sock \
  containrrr/watchtower \
  --monitor-only \
  --notifications slack \
  --notification-slack-hook-url YOUR_SLACK_WEBHOOK
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues

```bash
# Check memory usage
docker stats digipal

# Increase memory limit
docker run --memory=4g digipal
```

#### 2. Model Loading Failures

```bash
# Check model cache
docker exec digipal ls -la ~/.cache/huggingface

# Clear cache and restart
docker exec digipal rm -rf ~/.cache/huggingface
docker restart digipal
```

#### 3. Database Corruption

```bash
# Backup database
docker exec digipal cp /app/digipal.db /app/backup.db

# Restore from backup
docker exec digipal cp /app/assets/backups/latest.db /app/digipal.db
```

#### 4. Network Issues

```bash
# Check container networking
docker network ls
docker inspect digipal

# Test connectivity
docker exec digipal curl -I https://huggingface.co
```

### Performance Optimization

#### 1. Model Caching

```python
# Pre-load models
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache
```

#### 2. Database Optimization

```sql
-- Add indexes for better performance
CREATE INDEX idx_user_id ON digipals(user_id);
CREATE INDEX idx_timestamp ON interactions(timestamp);
```

#### 3. Resource Limits

```yaml
# docker-compose resource limits
services:
  digipal:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Debugging

#### 1. Enable Debug Mode

```bash
export DIGIPAL_ENV=development
export DIGIPAL_LOG_LEVEL=DEBUG
```

#### 2. Container Debugging

```bash
# Access container shell
docker exec -it digipal /bin/bash

# View logs
docker logs -f digipal

# Check processes
docker exec digipal ps aux
```

#### 3. Application Debugging

```python
# Add debug endpoints
@app.route('/debug/status')
def debug_status():
    return {
        'memory_usage': get_memory_usage(),
        'active_pets': get_active_pet_count(),
        'model_status': check_model_status()
    }
```

## Security Considerations

### 1. Token Security

- Store HF tokens in secure secret management
- Rotate tokens regularly
- Use least-privilege access

### 2. Network Security

- Use HTTPS in production
- Implement rate limiting
- Configure CORS properly

### 3. Container Security

```dockerfile
# Run as non-root user
RUN useradd --create-home --shell /bin/bash digipal
USER digipal

# Remove unnecessary packages
RUN apt-get remove --purge -y build-essential && \
    apt-get autoremove -y && \
    apt-get clean
```

## Backup and Recovery

### Automated Backups

```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker exec digipal tar -czf /app/backup_$DATE.tar.gz /app/data
docker cp digipal:/app/backup_$DATE.tar.gz ./backups/
```

### Disaster Recovery

1. **Data Backup**: Regular database and asset backups
2. **Configuration Backup**: Version control for configs
3. **Image Registry**: Store Docker images in registry
4. **Recovery Testing**: Regular recovery drills

This deployment guide should help you get DigiPal running in any environment. For specific issues, check the logs and health endpoints for detailed information.