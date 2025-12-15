#!/bin/bash
# ============================================================================
# AWS EC2 Deployment Script
# Cloud LLM Inference Benchmark Project
# MSML 650 - Cloud Computing
# ============================================================================
# This script helps deploy and manage EC2 GPU instances for benchmarking
#
# Prerequisites:
#   - AWS CLI configured with credentials
#   - jq installed (for JSON parsing)
#
# Usage:
#   ./deploy_aws.sh launch    # Launch new EC2 instance
#   ./deploy_aws.sh status    # Check instance status
#   ./deploy_aws.sh connect   # SSH into instance
#   ./deploy_aws.sh upload    # Upload project files
#   ./deploy_aws.sh download  # Download results
#   ./deploy_aws.sh terminate # Terminate instance
# ============================================================================

set -e

# Configuration - MODIFY THESE VALUES
AWS_REGION="us-east-1"
INSTANCE_TYPE="g5.xlarge"  # NVIDIA A10G GPU (~$1/hr)
AMI_ID="ami-0123456789abcdef0"  # Replace with Deep Learning AMI ID for your region
KEY_NAME="your-key-pair"  # Replace with your SSH key pair name
SECURITY_GROUP="sg-xxxxxxxx"  # Replace with your security group ID
SUBNET_ID="subnet-xxxxxxxx"  # Replace with your subnet ID (optional)
KEY_PATH="~/.ssh/your-key.pem"  # Path to your SSH private key

# Instance tracking file
INSTANCE_FILE=".aws_instance_id"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}=============================================="
    echo "  $1"
    echo -e "==============================================${NC}"
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Get stored instance ID
get_instance_id() {
    if [ -f "$INSTANCE_FILE" ]; then
        cat "$INSTANCE_FILE"
    else
        echo ""
    fi
}

# Launch new EC2 instance
launch_instance() {
    print_header "Launching EC2 GPU Instance"
    
    echo "Configuration:"
    echo "  Region: $AWS_REGION"
    echo "  Instance Type: $INSTANCE_TYPE"
    echo "  AMI: $AMI_ID"
    echo ""
    
    # Launch instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --region "$AWS_REGION" \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SECURITY_GROUP" \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=llm-benchmark-server},{Key=Project,Value=MSML650}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
    
    echo "$INSTANCE_ID" > "$INSTANCE_FILE"
    print_status "Instance launched: $INSTANCE_ID"
    
    # Wait for instance to be running
    echo "Waiting for instance to start..."
    aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
    print_status "Instance is running"
    
    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    echo ""
    print_status "Instance Details:"
    echo "  Instance ID: $INSTANCE_ID"
    echo "  Public IP: $PUBLIC_IP"
    echo ""
    echo "Connect with:"
    echo "  ssh -i $KEY_PATH ubuntu@$PUBLIC_IP"
    echo ""
    print_warning "Wait 2-3 minutes for the instance to fully initialize before connecting"
}

# Check instance status
check_status() {
    print_header "Instance Status"
    
    INSTANCE_ID=$(get_instance_id)
    
    if [ -z "$INSTANCE_ID" ]; then
        print_warning "No instance found. Launch one first with: ./deploy_aws.sh launch"
        return
    fi
    
    STATUS=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text 2>/dev/null || echo "not-found")
    
    PUBLIC_IP=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text 2>/dev/null || echo "N/A")
    
    echo "Instance ID: $INSTANCE_ID"
    echo "Status: $STATUS"
    echo "Public IP: $PUBLIC_IP"
    
    if [ "$STATUS" = "running" ]; then
        echo ""
        echo "Connect with:"
        echo "  ssh -i $KEY_PATH ubuntu@$PUBLIC_IP"
    fi
}

# SSH into instance
connect_instance() {
    INSTANCE_ID=$(get_instance_id)
    
    if [ -z "$INSTANCE_ID" ]; then
        print_error "No instance found"
        exit 1
    fi
    
    PUBLIC_IP=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    if [ "$PUBLIC_IP" = "None" ] || [ -z "$PUBLIC_IP" ]; then
        print_error "Instance has no public IP. Is it running?"
        exit 1
    fi
    
    print_status "Connecting to $PUBLIC_IP..."
    ssh -i "$KEY_PATH" "ubuntu@$PUBLIC_IP"
}

# Upload project files
upload_files() {
    print_header "Uploading Project Files"
    
    INSTANCE_ID=$(get_instance_id)
    
    if [ -z "$INSTANCE_ID" ]; then
        print_error "No instance found"
        exit 1
    fi
    
    PUBLIC_IP=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    echo "Uploading to ubuntu@$PUBLIC_IP..."
    
    # Upload entire project directory
    scp -i "$KEY_PATH" -r \
        ../cloud-llm-inference-benchmark \
        "ubuntu@$PUBLIC_IP:~/"
    
    print_status "Upload complete!"
    echo "Files uploaded to: ~/cloud-llm-inference-benchmark"
}

# Download results
download_results() {
    print_header "Downloading Results"
    
    INSTANCE_ID=$(get_instance_id)
    
    if [ -z "$INSTANCE_ID" ]; then
        print_error "No instance found"
        exit 1
    fi
    
    PUBLIC_IP=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    # Create local results directory
    mkdir -p ../results_from_aws
    
    echo "Downloading from ubuntu@$PUBLIC_IP..."
    
    scp -i "$KEY_PATH" -r \
        "ubuntu@$PUBLIC_IP:~/cloud-llm-inference-benchmark/results/*" \
        ../results_from_aws/
    
    print_status "Download complete!"
    echo "Results saved to: ../results_from_aws/"
}

# Terminate instance
terminate_instance() {
    print_header "Terminating Instance"
    
    INSTANCE_ID=$(get_instance_id)
    
    if [ -z "$INSTANCE_ID" ]; then
        print_warning "No instance found"
        return
    fi
    
    echo "Instance ID: $INSTANCE_ID"
    read -p "Are you sure you want to terminate? (yes/no): " CONFIRM
    
    if [ "$CONFIRM" = "yes" ]; then
        aws ec2 terminate-instances \
            --region "$AWS_REGION" \
            --instance-ids "$INSTANCE_ID"
        
        rm -f "$INSTANCE_FILE"
        print_status "Instance termination initiated"
    else
        echo "Cancelled"
    fi
}

# Show usage
show_usage() {
    echo "AWS EC2 Deployment Script for LLM Benchmarking"
    echo ""
    echo "Usage: ./deploy_aws.sh <command>"
    echo ""
    echo "Commands:"
    echo "  launch     Launch a new EC2 GPU instance"
    echo "  status     Check instance status"
    echo "  connect    SSH into the instance"
    echo "  upload     Upload project files to instance"
    echo "  download   Download results from instance"
    echo "  terminate  Terminate the instance"
    echo ""
    echo "Configuration:"
    echo "  Edit the variables at the top of this script:"
    echo "  - AWS_REGION, INSTANCE_TYPE, AMI_ID"
    echo "  - KEY_NAME, SECURITY_GROUP, KEY_PATH"
}

# Main
case "${1:-}" in
    launch)
        launch_instance
        ;;
    status)
        check_status
        ;;
    connect)
        connect_instance
        ;;
    upload)
        upload_files
        ;;
    download)
        download_results
        ;;
    terminate)
        terminate_instance
        ;;
    *)
        show_usage
        ;;
esac

