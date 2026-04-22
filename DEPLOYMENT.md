"""
Deployment and Production Configuration Guide
"""

def print_section(title):
    """Print formatted section"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def docker_deployment():
    """Docker deployment guide"""
    print_section("DOCKER DEPLOYMENT")
    
    dockerfile = '''
# Dockerfile for Unified MLOps Layer

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p mlflow/artifacts logs model_store

# Expose ports
EXPOSE 5001 5000

# Set environment variables
ENV FLASK_APP=api.py
ENV PYTHONUNBUFFERED=1

# Run the API
CMD ["python", "api.py"]
'''
    
    print("Dockerfile:")
    print(dockerfile)
    
    print("\nBuild and run:")
    print("""
# Build image
docker build -t mlops-unified:1.0 .

# Run container
docker run -p 5001:5001 -p 5000:5000 \\
  -v $(pwd)/model_store:/app/model_store \\
  -v $(pwd)/mlflow:/app/mlflow \\
  -v $(pwd)/logs:/app/logs \\
  mlops-unified:1.0

# Or with docker-compose
docker-compose up
""")


def docker_compose_guide():
    """Docker Compose deployment guide"""
    print_section("DOCKER-COMPOSE DEPLOYMENT")
    
    compose = '''
version: '3.8'

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: mlflow
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow:/mlflow
    command: |
      mlflow server
      --backend-store-uri sqlite:///mlflow/mlflow.db
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5000
    
  mlops-api:
    build: .
    container_name: mlops-api
    ports:
      - "5001:5001"
    volumes:
      - ./model_store:/app/model_store
      - ./mlflow:/app/mlflow
      - ./logs:/app/logs
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - FLASK_ENV=production
    depends_on:
      - mlflow
    command: python api.py

  mlops-inference:
    build:
      context: .
      dockerfile: Dockerfile.inference
    container_name: mlops-inference
    ports:
      - "8000:8000"
    volumes:
      - ./model_store:/app/model_store
      - ./mlflow:/app/mlflow
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
'''
    
    print("docker-compose.yml:")
    print(compose)
    
    print("\nUsage:")
    print("""
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f mlops-api

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
""")


def kubernetes_deployment():
    """Kubernetes deployment guide"""
    print_section("KUBERNETES DEPLOYMENT")
    
    deployment = '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlops-api
  labels:
    app: mlops-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlops-api
  template:
    metadata:
      labels:
        app: mlops-api
    spec:
      containers:
      - name: mlops-api
        image: mlops-unified:1.0
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 5001
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-service:5000"
        - name: FLASK_ENV
          value: "production"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: model-store
          mountPath: /app/model_store
        - name: mlflow-artifacts
          mountPath: /app/mlflow
        livenessProbe:
          httpGet:
            path: /health
            port: 5001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5001
          initialDelaySeconds: 10
          periodSeconds: 5
      volumes:
      - name: model-store
        persistentVolumeClaim:
          claimName: model-store-pvc
      - name: mlflow-artifacts
        persistentVolumeClaim:
          claimName: mlflow-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: mlops-api-service
spec:
  type: LoadBalancer
  selector:
    app: mlops-api
  ports:
  - protocol: TCP
    port: 5001
    targetPort: 5001
'''
    
    print("kubernetes-deployment.yaml:")
    print(deployment)


def aws_deployment():
    """AWS deployment guide"""
    print_section("AWS DEPLOYMENT GUIDE")
    
    print("""
1. USING AWS LAMBDA + API GATEWAY
   • Package MLOps as Lambda layers
   • Use API Gateway for REST endpoints
   • Store models in S3
   • Use RDS for audit database

2. USING ECS
   • Build Docker image
   • Push to ECR: aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
   • Create ECS task definition
   • Deploy to ECS cluster

3. USING SAGEMAKER
   • Use SageMaker Model Registry instead of MLFlow
   • Deploy models as SageMaker endpoints
   • Use SageMaker Feature Store

4. INFRASTRUCTURE
   • VPC with private subnets for models
   • S3 bucket for model artifacts: s3://mlops-models-{env}/
   • RDS for MLFlow backend: mlflow-db.{account}.rds.amazonaws.com
   • CloudWatch for monitoring and logging
   • KMS for encryption at rest

Example Terraform:
""")
    
    terraform = '''
# main.tf
provider "aws" {
  region = "us-east-1"
}

# S3 bucket for models
resource "aws_s3_bucket" "model_store" {
  bucket = "mlops-models-${var.environment}"
}

# RDS for MLFlow
resource "aws_db_instance" "mlflow" {
  allocated_storage    = 20
  engine              = "mysql"
  engine_version      = "8.0"
  instance_class      = "db.t3.micro"
  name                = "mlflow"
  username            = "admin"
  password            = random_password.db_password.result
  skip_final_snapshot = true
}

# ECS Task Definition
resource "aws_ecs_task_definition" "mlops" {
  family                   = "mlops-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"

  container_definitions = jsonencode([
    {
      name      = "mlops-api"
      image     = "mlops-unified:1.0"
      portMappings = [
        {
          containerPort = 5001
          hostPort      = 5001
        }
      ]
      environment = [
        {
          name  = "MLFLOW_TRACKING_URI"
          value = "http://mlflow:5000"
        }
      ]
    }
  ])
}
'''
    
    print(terraform)


def gcp_deployment():
    """GCP deployment guide"""
    print_section("GCP DEPLOYMENT GUIDE")
    
    print("""
1. USING CLOUD RUN
   • Deploy containerized MLOps application
   • Automatic scaling based on traffic
   • Pay only for compute time used

2. USING APP ENGINE
   • Deploy Python application directly
   • Managed runtime environment
   • Automatic SSL/TLS certificates

3. STORAGE
   • Cloud Storage for model artifacts
   • Cloud SQL for MLFlow backend
   • Firestore for audit logs

4. MONITORING
   • Cloud Logging for application logs
   • Cloud Monitoring for metrics
   • Cloud Trace for performance analysis

Example gcloud commands:
""")
    
    commands = '''
# Build and push to Artifact Registry
gcloud builds submit --tag gcr.io/$PROJECT_ID/mlops-unified

# Deploy to Cloud Run
gcloud run deploy mlops-api \\
  --image gcr.io/$PROJECT_ID/mlops-unified \\
  --platform managed \\
  --region us-central1 \\
  --allow-unauthenticated \\
  --set-env-vars MLFLOW_TRACKING_URI=http://mlflow:5000

# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=mlops-api" \\
  --limit 50 \\
  --format json
'''
    
    print(commands)


def monitoring_logging():
    """Monitoring and logging guide"""
    print_section("MONITORING & LOGGING")
    
    print("""
1. APPLICATION LOGGING
   • Structured logging to logs/audit.log
   • Log levels: DEBUG, INFO, WARNING, ERROR
   • Rotation enabled for large files

2. METRICS TO MONITOR
   • API response time
   • Model upload/conversion duration
   • Inference latency
   • Error rates by operation
   • Database query performance
   • Storage usage

3. ALERTING
   • Alert on conversion failures
   • Alert on high inference latency (>100ms)
   • Alert on prediction errors
   • Alert on disk space usage (>90%)
   • Alert on database connection failures

4. TOOLS
   • Prometheus: Metrics collection
   • Grafana: Dashboards and visualization
   • ELK Stack: Log aggregation and analysis
   • New Relic: APM and monitoring
   • DataDog: Infrastructure monitoring

Example Prometheus config:
""")
    
    prometheus = '''
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'mlops-api'
    static_configs:
      - targets: ['localhost:5001']
    metrics_path: '/metrics'

  - job_name: 'mlflow'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
'''
    
    print(prometheus)


def security_hardening():
    """Security hardening guide"""
    print_section("SECURITY HARDENING")
    
    print("""
1. AUTHENTICATION & AUTHORIZATION
   • Implement JWT/OAuth2 for API authentication
   • Use role-based access control (RBAC)
   • Require HTTPS/TLS certificates
   • Implement rate limiting

2. DATA SECURITY
   • Encrypt model files at rest (S3 SSE, AES-256)
   • Encrypt in-transit (TLS 1.3)
   • Scan models for malware before serving
   • Audit trail for access logs

3. NETWORK SECURITY
   • Use VPC with private subnets
   • Implement WAF (Web Application Firewall)
   • Use security groups and NACLs
   • Enable VPC Flow Logs

4. COMPLIANCE
   • GDPR: User data deletion support
   • HIPAA: Encryption and audit logs
   • SOC 2: Access controls and monitoring
   • PCI DSS: If handling payment data

5. IMPLEMENTATION EXAMPLE
""")
    
    security_code = '''
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize JWT
jwt = JWTManager(app)

@app.route('/auth/login', methods=['POST'])
def login():
    """Authenticate user and return JWT token"""
    username = request.json.get('username')
    password = request.json.get('password')
    
    # Verify credentials
    user = verify_user(username, password)
    if not user:
        return {'error': 'Invalid credentials'}, 401
    
    # Create access token
    access_token = create_access_token(identity=user.id)
    return {'access_token': access_token}

@app.route('/api/models/upload', methods=['POST'])
@jwt_required()
def upload_model():
    """Protected endpoint - requires JWT token"""
    # ... existing upload logic
'''
    
    print(security_code)


def performance_optimization():
    """Performance optimization guide"""
    print_section("PERFORMANCE OPTIMIZATION")
    
    print("""
1. CACHING
   • Cache model metadata in Redis
   • Cache predictions for identical inputs
   • Use CDN for model artifacts

2. BATCHING
   • Batch multiple prediction requests
   • Use batch prediction for large datasets
   • Async processing for long operations

3. DATABASE
   • Index audit tables by model_id and timestamp
   • Partition large tables by date
   • Use connection pooling

4. API
   • Enable gzip compression
   • Async endpoints for long operations
   • Request/response filtering

5. INFRASTRUCTURE
   • Horizontal scaling with load balancer
   • Model caching on GPU instances
   • Dedicated inference servers
   • CDN for static content
""")


def disaster_recovery():
    """Disaster recovery guide"""
    print_section("DISASTER RECOVERY & BACKUP")
    
    print("""
1. BACKUP STRATEGY
   • Daily backups of audit database
   • Weekly snapshots of model storage
   • Multi-region replication
   • Test restores monthly

2. HIGH AVAILABILITY
   • Multi-zone deployment
   • Database replication
   • Load balancing across instances
   • Automated failover

3. RECOVERY PROCEDURES
   • RPO (Recovery Point Objective): < 1 hour
   • RTO (Recovery Time Objective): < 4 hours
   • Document runbooks for all scenarios
   • Practice disaster recovery drills

4. BACKUP COMMANDS
""")
    
    backup_commands = '''
# Backup MLFlow database
mysqldump -u admin -p mlflow > mlflow_backup_$(date +%Y%m%d).sql

# Backup S3 models
aws s3 sync s3://mlops-models-prod s3://mlops-models-backup/

# Backup audit logs
tar -czf audit_logs_$(date +%Y%m%d).tar.gz logs/

# Restore from backup
mysql -u admin -p mlflow < mlflow_backup_20240421.sql
aws s3 sync s3://mlops-models-backup/ s3://mlops-models-prod/
'''
    
    print(backup_commands)


def main():
    """Show deployment guide"""
    print("\n" + "="*70)
    print("  DEPLOYMENT & PRODUCTION CONFIGURATION GUIDE")
    print("="*70)
    
    docker_deployment()
    docker_compose_guide()
    kubernetes_deployment()
    aws_deployment()
    gcp_deployment()
    monitoring_logging()
    security_hardening()
    performance_optimization()
    disaster_recovery()
    
    print("\n" + "="*70)
    print("  DEPLOYMENT GUIDE COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
