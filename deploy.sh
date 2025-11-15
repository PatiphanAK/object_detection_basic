#!/bin/bash

# YOLO API Deployment Script
# Works for both Local and EC2 deployment

set -e

echo "=========================================="
echo "🚀 YOLO API Deployment Script (UV)"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    echo "Please install Docker Compose first"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose are installed${NC}"

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ pyproject.toml not found${NC}"
    echo "Please run setup-uv.sh first or create pyproject.toml"
    exit 1
fi

echo -e "${GREEN}✅ pyproject.toml found${NC}"

# Check if uv.lock exists
if [ ! -f "uv.lock" ]; then
    echo -e "${YELLOW}⚠️  uv.lock not found, generating...${NC}"
    if command -v uv &> /dev/null; then
        uv lock
        echo -e "${GREEN}✅ uv.lock generated${NC}"
    else
        echo -e "${RED}❌ UV is not installed. Install it first or create uv.lock${NC}"
        exit 1
    fi
fi

# Check if model file exists
if [ ! -f "best.pt" ]; then
    echo -e "${YELLOW}⚠️  Warning: best.pt not found${NC}"
    echo "Please place your YOLO model file (best.pt) in the current directory"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p prometheus grafana/provisioning/datasources grafana/provisioning/dashboards grafana/dashboards
mkdir -p uploads results gradcam_results

# Build and start containers
echo "🔨 Building Docker images with UV..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo ""
echo "🔍 Checking service health..."

if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✅ YOLO API is running${NC}"
else
    echo -e "${RED}❌ YOLO API is not responding${NC}"
fi

if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo -e "${GREEN}✅ Prometheus is running${NC}"
else
    echo -e "${RED}❌ Prometheus is not responding${NC}"
fi

if curl -s http://localhost:3000/api/health > /dev/null; then
    echo -e "${GREEN}✅ Grafana is running${NC}"
else
    echo -e "${RED}❌ Grafana is not responding${NC}"
fi

echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
echo ""
echo "📊 Access your services:"
echo "  • YOLO API:    http://localhost:8000"
echo "  • API Docs:    http://localhost:8000/docs"
echo "  • Prometheus:  http://localhost:9090"
echo "  • Grafana:     http://localhost:3000"
echo ""
echo "🔐 Grafana credentials:"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "📈 Useful commands:"
echo "  • View logs:        docker-compose logs -f"
echo "  • Stop services:    docker-compose down"
echo "  • Restart services: docker-compose restart"
echo "  • View metrics:     curl http://localhost:8000/metrics"
echo ""
echo "💡 UV commands (local development):"
echo "  • Run locally:      uv run uvicorn main:app --reload"
echo "  • Add package:      uv add <package>"
echo "  • Sync deps:        uv sync"
echo ""
echo "=========================================="
