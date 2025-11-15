#!/bin/bash

# =========================================
# YOLO Inference Deployment Script
# Supports: local, ec2
# =========================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="yolo-inference"
IMAGE_NAME="yolo-inference:latest"
ARCHIVE_NAME="yolo-inference.tar.gz"

# Default values
ENVIRONMENT="local"
EC2_HOST=""
EC2_USER="ubuntu"
EC2_PATH="/home/ubuntu/yolo-app"

# =========================================
# Helper Functions
# =========================================

print_header() {
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║   🚀 YOLO Inference Deployment Tool   ║${NC}"
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -e, --env ENVIRONMENT       Environment to deploy (local, ec2) [default: local]
    -h, --host EC2_HOST        EC2 host address (required for ec2)
    -u, --user EC2_USER        EC2 SSH user [default: ubuntu]
    -p, --path EC2_PATH        EC2 deployment path [default: /home/ubuntu/yolo-app]
    --help                     Show this help message

EXAMPLES:
    # Deploy locally
    ./deploy.sh --env local

    # Deploy to EC2
    ./deploy.sh --env ec2 --host 54.123.45.67

    # Deploy to EC2 with custom user and path
    ./deploy.sh --env ec2 --host ec2.example.com --user ec2-user --path /opt/yolo

EOF
    exit 0
}

# =========================================
# Parse Arguments
# =========================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -h|--host)
            EC2_HOST="$2"
            shift 2
            ;;
        -u|--user)
            EC2_USER="$2"
            shift 2
            ;;
        -p|--path)
            EC2_PATH="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# =========================================
# Validation
# =========================================

validate_environment() {
    if [[ "$ENVIRONMENT" != "local" && "$ENVIRONMENT" != "ec2" ]]; then
        print_error "Invalid environment: $ENVIRONMENT"
        print_info "Valid options: local, ec2"
        exit 1
    fi

    if [[ "$ENVIRONMENT" == "ec2" && -z "$EC2_HOST" ]]; then
        print_error "EC2 host is required for EC2 deployment"
        print_info "Usage: $0 --env ec2 --host YOUR_EC2_IP"
        exit 1
    fi
}

check_dependencies() {
    print_info "Checking dependencies..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi

    # Check for Docker Compose (V1 or V2)
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
        print_info "Using Docker Compose V1"
    elif docker compose version &> /dev/null; then
        DOCKER_COMPOSE="docker compose"
        print_info "Using Docker Compose V2"
    else
        print_error "Docker Compose is not installed"
        exit 1
    fi

    if [[ "$ENVIRONMENT" == "ec2" ]]; then
        if ! command -v ssh &> /dev/null; then
            print_error "SSH is not installed"
            exit 1
        fi
        if ! command -v scp &> /dev/null; then
            print_error "SCP is not installed"
            exit 1
        fi
    fi

    print_success "All dependencies are available"
}

check_model_weights() {
    print_info "Checking model weights..."

    if [[ ! -f "weights/best/best.pt" ]]; then
        print_error "Model weights not found at weights/best/best.pt"
        exit 1
    fi

    print_success "Model weights found"
}

# =========================================
# Build Functions
# =========================================

build_image() {
    print_info "Building Docker image..."

    if docker build -t "$IMAGE_NAME" .; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

create_archive() {
    print_info "Creating image archive..."

    if docker save "$IMAGE_NAME" | gzip > "$ARCHIVE_NAME"; then
        print_success "Archive created: $ARCHIVE_NAME"
        local size=$(du -h "$ARCHIVE_NAME" | cut -f1)
        print_info "Archive size: $size"
    else
        print_error "Failed to create archive"
        exit 1
    fi
}

# =========================================
# Local Deployment
# =========================================

deploy_local() {
    print_header
    print_info "🏠 Starting LOCAL deployment..."
    echo ""

    # Build
    build_image

    # Stop old containers
    print_info "Stopping old containers..."
    $DOCKER_COMPOSE down || true

    # Start new containers
    print_info "Starting new containers..."
    $DOCKER_COMPOSE up -d

    # Wait for services
    print_info "Waiting for services to start..."
    sleep 10

    # Check status
    print_info "Container status:"
    $DOCKER_COMPOSE ps

    # Health check
    print_info "Performing health check..."
    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if curl -s -f http://localhost:8000/ > /dev/null 2>&1; then
            print_success "API is healthy!"
            break
        fi

        if [[ $attempt -eq $max_attempts ]]; then
            print_error "API health check failed after $max_attempts attempts"
            print_info "Check logs with: $DOCKER_COMPOSE logs yolo-api"
            exit 1
        fi

        echo -ne "\rAttempt $attempt/$max_attempts..."
        sleep 2
        ((attempt++))
    done

    echo ""
    print_success "🎉 Local deployment completed successfully!"
    echo ""
    print_info "📊 Services are available at:"
    echo "   • API:        http://localhost:8000"
    echo "   • API Docs:   http://localhost:8000/docs"
    echo "   • Prometheus: http://localhost:9090"
    echo "   • Grafana:    http://localhost:3000 (admin/admin)"
    echo "   • cAdvisor:   http://localhost:8080"
    echo ""
    print_info "📝 Useful commands:"
    echo "   • View logs:     $DOCKER_COMPOSE logs -f yolo-api"
    echo "   • Stop all:      $DOCKER_COMPOSE down"
    echo "   • Restart:       $DOCKER_COMPOSE restart yolo-api"
}

# =========================================
# EC2 Deployment
# =========================================

test_ssh_connection() {
    print_info "Testing SSH connection to $EC2_USER@$EC2_HOST..."

    if ssh -o ConnectTimeout=10 -o BatchMode=yes "$EC2_USER@$EC2_HOST" "echo 'SSH OK'" &> /dev/null; then
        print_success "SSH connection successful"
        return 0
    else
        print_error "Cannot connect to EC2 instance"
        print_info "Please check:"
        echo "   • EC2 host is correct: $EC2_HOST"
        echo "   • SSH key is configured"
        echo "   • Security group allows SSH (port 22)"
        exit 1
    fi
}

setup_ec2_environment() {
    print_info "Setting up EC2 environment..."

    ssh "$EC2_USER@$EC2_HOST" << 'EOF'
        # Install Docker if not exists
        if ! command -v docker &> /dev/null; then
            echo "Installing Docker..."
            sudo apt-get update
            sudo apt-get install -y docker.io docker-compose
            sudo usermod -aG docker $USER
            newgrp docker
        fi

        # Create deployment directory
        mkdir -p ~/yolo-app/{prometheus,grafana/provisioning,grafana/dashboards}

        echo "EC2 environment ready"
EOF

    print_success "EC2 environment configured"
}

transfer_files() {
    print_info "Transferring files to EC2..."

    # Transfer archive
    print_info "Uploading Docker image ($ARCHIVE_NAME)..."
    if scp -C "$ARCHIVE_NAME" "$EC2_USER@$EC2_HOST:$EC2_PATH/"; then
        print_success "Archive uploaded"
    else
        print_error "Failed to upload archive"
        exit 1
    fi

    # Transfer docker-compose and configs
    print_info "Uploading configuration files..."
    scp docker-compose.yml "$EC2_USER@$EC2_HOST:$EC2_PATH/" || true
    scp -r prometheus "$EC2_USER@$EC2_HOST:$EC2_PATH/" 2>/dev/null || true
    scp -r grafana "$EC2_USER@$EC2_HOST:$EC2_PATH/" 2>/dev/null || true

    print_success "Files transferred successfully"
}

deploy_on_ec2() {
    print_info "Deploying on EC2..."

    ssh "$EC2_USER@$EC2_HOST" << EOF
        cd $EC2_PATH

        # Detect Docker Compose version
        if command -v docker-compose &> /dev/null; then
            COMPOSE_CMD="docker-compose"
        else
            COMPOSE_CMD="docker compose"
        fi

        # Load image
        echo "Loading Docker image..."
        docker load < $ARCHIVE_NAME

        # Stop old containers
        echo "Stopping old containers..."
        \$COMPOSE_CMD down || true

        # Start new containers
        echo "Starting new containers..."
        \$COMPOSE_CMD up -d

        # Wait and check
        sleep 15
        \$COMPOSE_CMD ps

        # Cleanup archive
        rm -f $ARCHIVE_NAME

        echo "Deployment complete on EC2"
EOF

    print_success "EC2 deployment successful"
}

verify_ec2_deployment() {
    print_info "Verifying EC2 deployment..."

    local max_attempts=30
    local attempt=1

    while [[ $attempt -le $max_attempts ]]; do
        if curl -s -f "http://$EC2_HOST:8000/" > /dev/null 2>&1; then
            print_success "API is healthy on EC2!"
            return 0
        fi

        if [[ $attempt -eq $max_attempts ]]; then
            print_warn "Could not verify API health (might be security group issue)"
            return 1
        fi

        echo -ne "\rAttempt $attempt/$max_attempts..."
        sleep 2
        ((attempt++))
    done
}

deploy_ec2() {
    print_header
    print_info "☁️  Starting EC2 deployment to $EC2_HOST..."
    echo ""

    # Validate SSH
    test_ssh_connection

    # Build and archive
    build_image
    create_archive

    # Setup EC2
    setup_ec2_environment

    # Transfer files
    transfer_files

    # Deploy
    deploy_on_ec2

    # Cleanup local archive
    print_info "Cleaning up local archive..."
    rm -f "$ARCHIVE_NAME"

    # Verify
    verify_ec2_deployment

    echo ""
    print_success "🎉 EC2 deployment completed successfully!"
    echo ""
    print_info "📊 Services should be available at:"
    echo "   • API:        http://$EC2_HOST:8000"
    echo "   • API Docs:   http://$EC2_HOST:8000/docs"
    echo "   • Prometheus: http://$EC2_HOST:9090"
    echo "   • Grafana:    http://$EC2_HOST:3000"
    echo "   • cAdvisor:   http://$EC2_HOST:8080"
    echo ""
    print_warn "⚠️  Make sure EC2 Security Group allows inbound traffic on ports:"
    echo "   • 8000 (API), 3000 (Grafana), 9090 (Prometheus), 8080 (cAdvisor)"
    echo ""
    print_info "📝 Useful commands:"
    echo "   • SSH to EC2:    ssh $EC2_USER@$EC2_HOST"
    echo "   • View logs:     ssh $EC2_USER@$EC2_HOST 'cd $EC2_PATH && docker compose logs -f yolo-api'"
    echo "   • Stop all:      ssh $EC2_USER@$EC2_HOST 'cd $EC2_PATH && docker compose down'"
}

# =========================================
# Main Execution
# =========================================

main() {
    validate_environment
    check_dependencies
    check_model_weights

    case $ENVIRONMENT in
        local)
            deploy_local
            ;;
        ec2)
            deploy_ec2
            ;;
    esac
}

# Run main function
main
