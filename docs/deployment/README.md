# Deployment Documentation

This directory contains comprehensive deployment and operations documentation for YICA/YiRage.

## 📖 Documentation Overview

### Deployment Guides
- **[Docker Deployment](docker-deployment.md)** - Deploy YICA environment using Docker containers
- **[Deployment Report](deployment-report.md)** - Deployment implementation reports and status

### Planned Documentation
- **Production Deployment** - Large-scale production deployment guide
- **Cloud Deployment** - AWS, GCP, Azure deployment strategies
- **Kubernetes Deployment** - Container orchestration deployment
- **Monitoring and Operations** - System monitoring and operational procedures

## 🚀 Deployment Options

### 1. Docker Deployment (Recommended)

#### Quick Start
```bash
# One-click deployment
./scripts/docker_yica_deployment.sh

# Access via web interface
# URL: http://localhost:6080 (password: yica)
```

#### Custom Configuration
```bash
# Build custom image
docker build -t yica-custom -f docker/Dockerfile .

# Run with custom settings
docker run -d \
  --name yica-container \
  -p 6080:6080 \
  -p 5900:5900 \
  -v $(pwd)/data:/home/yica/data \
  yica-custom
```

### 2. Native Installation

#### System Requirements
- **OS**: Ubuntu 20.04+ / CentOS 8+ / macOS 11+
- **CPU**: 4+ cores, x86_64 or ARM64
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 20GB free space
- **Network**: Internet access for dependencies

#### Installation Steps
```bash
# Install dependencies
sudo apt update
sudo apt install -y cmake build-essential python3-dev git

# Clone repository
git clone https://github.com/yica-ai/yica-yirage.git
cd yica-yirage

# Build and install
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install

# Install Python package
cd ../yirage/python
pip install -e .
```

### 3. Cloud Deployment

#### AWS Deployment
```bash
# Using AWS CLI
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1d0 \
  --instance-type c5.2xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxxxxx \
  --user-data file://deploy-script.sh

# Using Terraform
terraform init
terraform plan -var="instance_type=c5.2xlarge"
terraform apply
```

#### Google Cloud Platform
```bash
# Create VM instance
gcloud compute instances create yica-instance \
  --zone=us-central1-a \
  --machine-type=c2-standard-8 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --metadata-from-file startup-script=startup.sh

# Deploy using Kubernetes
kubectl apply -f k8s/yica-deployment.yaml
```

## 🏗️ Architecture Deployment

### Single Node Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  yica-optimizer:
    image: yica/yirage:latest
    ports:
      - "6080:6080"
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - YICA_BACKEND=yica
      - YICA_LOG_LEVEL=INFO
```

### Multi-Node Cluster
```yaml
# k8s/yica-cluster.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: yica-cluster
spec:
  replicas: 3
  selector:
    matchLabels:
      app: yica-optimizer
  template:
    metadata:
      labels:
        app: yica-optimizer
    spec:
      containers:
      - name: yica-optimizer
        image: yica/yirage:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"
```

### Load Balancer Configuration
```nginx
# nginx.conf
upstream yica_backend {
    least_conn;
    server yica-node1:8080 max_fails=3 fail_timeout=30s;
    server yica-node2:8080 max_fails=3 fail_timeout=30s;
    server yica-node3:8080 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name yica.example.com;
    
    location / {
        proxy_pass http://yica_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## ⚙️ Configuration Management

### Environment Configuration
```bash
# Environment variables
export YICA_BACKEND=yica
export YICA_CONFIG_FILE=/etc/yica/config.json
export YICA_LOG_LEVEL=INFO
export YICA_DATA_DIR=/var/lib/yica
export YICA_CACHE_DIR=/var/cache/yica
```

### Configuration Files
```json
{
  "yica": {
    "hardware": {
      "num_dies": 8,
      "clusters_per_die": 4,
      "cim_arrays_per_cluster": 16
    },
    "optimization": {
      "strategy": "balanced",
      "max_search_time": 3600,
      "parallel_jobs": 4
    }
  },
  "logging": {
    "level": "INFO",
    "file": "/var/log/yica/yica.log",
    "rotation": "daily",
    "max_size": "100MB"
  }
}
```

### Security Configuration
```yaml
# security.yaml
authentication:
  enabled: true
  method: "jwt"
  secret_key: "${JWT_SECRET_KEY}"

authorization:
  enabled: true
  roles:
    - name: "admin"
      permissions: ["*"]
    - name: "user"
      permissions: ["optimize", "profile"]

tls:
  enabled: true
  cert_file: "/etc/ssl/certs/yica.crt"
  key_file: "/etc/ssl/private/yica.key"
```

## 📊 Monitoring and Operations

### Health Checks
```bash
# Basic health check
curl -f http://localhost:8080/health || exit 1

# Detailed status
curl http://localhost:8080/api/status

# Performance metrics
curl http://localhost:8080/metrics
```

### Monitoring Setup
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'yica'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 15s
    metrics_path: '/metrics'

# grafana dashboard
{
  "dashboard": {
    "title": "YICA/YiRage Monitoring",
    "panels": [
      {
        "title": "Optimization Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(yica_optimizations_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### Log Management
```bash
# Configure log rotation
sudo tee /etc/logrotate.d/yica << EOF
/var/log/yica/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 yica yica
}
EOF

# Centralized logging with ELK
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  elasticsearch:7.14.0

docker run -d \
  --name logstash \
  -p 5000:5000 \
  -v $(pwd)/logstash.conf:/usr/share/logstash/pipeline/logstash.conf \
  logstash:7.14.0
```

## 🔒 Security Considerations

### Network Security
```bash
# Firewall configuration
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8080/tcp  # YICA API
sudo ufw enable

# SSL/TLS setup
sudo certbot --nginx -d yica.example.com
```

### Container Security
```dockerfile
# Secure Dockerfile
FROM ubuntu:20.04

# Create non-root user
RUN groupadd -r yica && useradd -r -g yica yica

# Install security updates
RUN apt-get update && apt-get upgrade -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set secure permissions
COPY --chown=yica:yica . /app
USER yica

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1
```

### Access Control
```yaml
# RBAC configuration
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: yica-operator
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: yica-operator-binding
subjects:
- kind: User
  name: yica-operator
  apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: Role
  name: yica-operator
  apiGroup: rbac.authorization.k8s.io
```

## 🚨 Troubleshooting

### Common Deployment Issues

#### Container Startup Problems
```bash
# Check container logs
docker logs yica-container

# Check resource usage
docker stats yica-container

# Inspect container configuration
docker inspect yica-container

# Debug container interactively
docker run -it --rm yica/yirage:latest /bin/bash
```

#### Network Connectivity Issues
```bash
# Test port connectivity
telnet localhost 8080

# Check DNS resolution
nslookup yica.example.com

# Verify firewall rules
sudo iptables -L

# Test internal networking
docker network ls
docker network inspect bridge
```

#### Performance Issues
```bash
# Monitor system resources
htop
iotop
nethogs

# Check YICA performance
yirage diagnose --performance

# Profile specific operations
yirage profile --input test_model.py --output profile.json
```

### Recovery Procedures

#### Database Recovery
```bash
# Backup current state
yirage backup --output backup_$(date +%Y%m%d).tar.gz

# Restore from backup
yirage restore --input backup_20241201.tar.gz

# Verify integrity
yirage verify --check-all
```

#### Service Recovery
```bash
# Restart services
systemctl restart yica-optimizer
systemctl restart nginx

# Check service status
systemctl status yica-optimizer

# View service logs
journalctl -u yica-optimizer -f
```

## 📋 Deployment Checklists

### Pre-Deployment Checklist
- [ ] System requirements verified
- [ ] Dependencies installed
- [ ] Network ports configured
- [ ] Security settings applied
- [ ] Backup strategy defined
- [ ] Monitoring setup complete

### Post-Deployment Verification
- [ ] Services running correctly
- [ ] Health checks passing
- [ ] API endpoints accessible
- [ ] Performance benchmarks met
- [ ] Monitoring alerts configured
- [ ] Documentation updated

### Production Readiness
- [ ] Load testing completed
- [ ] Failover procedures tested
- [ ] Security audit passed
- [ ] Compliance requirements met
- [ ] Operations team trained
- [ ] Incident response plan ready

## 🔗 Related Documentation

- [Getting Started](../getting-started/) - Basic setup and concepts
- [Architecture](../architecture/) - System architecture overview
- [API Reference](../api/) - API documentation
- [Development](../development/) - Development environment

## 📞 Support

### Deployment Support
- **Documentation**: Complete deployment guides
- **Community**: GitHub Discussions and Discord
- **Professional**: Enterprise support available

### Emergency Contacts
- **Critical Issues**: emergency@yica-support.com
- **General Support**: support@yica-support.com
- **Community**: Discord #deployment-help

---

*For the latest deployment information and updates, check our [GitHub repository](https://github.com/yica-ai/yica-yirage) and [documentation website](https://yica-yirage.readthedocs.io/).*