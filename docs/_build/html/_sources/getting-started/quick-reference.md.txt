# YICA-QEMU Docker Quick Reference

## 🚀 One-Click Deployment
```bash
./scripts/docker_yica_deployment.sh
```

## 🌐 Access URLs
- **Web VNC**: http://localhost:6080 (password: yica)
- **VNC Client**: localhost:5900 (password: yica)
- **SSH Access**: ssh yica@localhost -p 2222

## 🔧 Common Commands

### Container Management
```bash
./scripts/yica_docker_manager.sh status    # Check status
./scripts/yica_docker_manager.sh shell     # Enter container
./scripts/yica_docker_manager.sh logs      # View logs
./scripts/yica_docker_manager.sh restart   # Restart container
./scripts/yica_docker_manager.sh stop      # Stop container
./scripts/yica_docker_manager.sh remove    # Remove container
```

### Deployment Management
```bash
./scripts/docker_yica_deployment.sh check   # Check environment
./scripts/docker_yica_deployment.sh build   # Build image
./scripts/docker_yica_deployment.sh start   # Start container
./scripts/docker_yica_deployment.sh verify  # Verify deployment
./scripts/docker_yica_deployment.sh clean   # Clean up resources
```

## 🧪 YICA Usage

### Launch QEMU in VNC Desktop
```bash
cd /home/yica/workspace
./qemu2.sh
```

### Python Environment Test
```bash
python3 -c "import sys; sys.path.insert(0, '/home/yica/workspace/yirage/python'); import yirage; print(f'YICA Version: {yirage.__version__}')"
```

### Basic YiRage Commands
```bash
# Check YiRage installation
yirage --version

# List available backends
yirage backends --list

# Run basic optimization
yirage optimize --input example.py --backend yica --output optimized.py

# Profile performance
yirage profile --input model.py --backend yica
```

## 📦 Package Management

### Python Package Installation
```bash
# Install in development mode
cd /home/yica/workspace/yirage/python
pip install -e .

# Install specific version
pip install yirage==2.0.0

# Upgrade to latest
pip install --upgrade yirage
```

### System Dependencies
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install additional dependencies
sudo apt install -y cmake build-essential python3-dev

# Install CUDA (if needed)
sudo apt install -y nvidia-cuda-toolkit
```

## 🔍 System Information

### Hardware Information
```bash
# Check CPU information
lscpu

# Check memory
free -h

# Check GPU (if available)
nvidia-smi

# Check disk space
df -h
```

### YICA Environment
```bash
# Check YICA configuration
yirage config --show

# Verify hardware compatibility
yirage hardware-check

# System diagnostics
yirage diagnose --full
```

## ⚠️ Troubleshooting

### VNC Connection Issues
```bash
# Restart VNC service
./scripts/yica_docker_manager.sh shell
vncserver -kill :1
vncserver :1 -geometry 1920x1080 -depth 24

# Check VNC status
ps aux | grep vnc

# Check port availability
netstat -tlnp | grep :6080
```

### Container Issues
```bash
# Check container status
docker ps -a | grep yica

# View container logs
docker logs yica-container

# Restart container
docker restart yica-container

# Remove and recreate
docker rm -f yica-container
./scripts/docker_yica_deployment.sh
```

### Performance Issues
```bash
# Check system resources
htop

# Monitor container resources
docker stats yica-container

# Check YICA performance
yirage benchmark --quick-test

# Profile specific operations
yirage profile --input test_model.py --detailed
```

### Network Connectivity
```bash
# Check port accessibility
curl -I http://localhost:6080

# Test SSH connection
ssh -p 2222 yica@localhost

# Check firewall settings
sudo ufw status

# Test internal network
ping host.docker.internal
```

## 📋 Quick Checklists

### Pre-Deployment Checklist
- [ ] Docker installed and running
- [ ] Sufficient disk space (>10GB)
- [ ] Required ports available (6080, 5900, 2222)
- [ ] Network connectivity working
- [ ] User permissions configured

### Post-Deployment Verification
- [ ] Container running successfully
- [ ] VNC accessible via web browser
- [ ] SSH connection working
- [ ] YICA environment functional
- [ ] Python packages installed correctly
- [ ] Basic optimization test passes

### Performance Optimization Checklist
- [ ] Sufficient memory allocated to container
- [ ] CPU cores properly assigned
- [ ] GPU passthrough configured (if applicable)
- [ ] Storage performance adequate
- [ ] Network latency acceptable

## 🛠️ Development Workflow

### Code Development
```bash
# Enter development environment
./scripts/yica_docker_manager.sh shell

# Navigate to workspace
cd /home/yica/workspace

# Edit code (using vim/nano or mount external editor)
vim yirage/src/example.cpp

# Build changes
cd build && make -j$(nproc)

# Test changes
./run_tests.sh
```

### Debugging
```bash
# Enable debug logging
export YICA_LOG_LEVEL=DEBUG

# Run with debugging
yirage optimize --debug --input test.py --backend yica

# Use GDB for C++ debugging
gdb --args yica_optimizer test_input.c

# Python debugging
python -m pdb -c continue script.py
```

## 📚 Useful Resources

### Documentation Links
- [Architecture Overview](../architecture/README.md)
- [API Reference](../api/README.md)
- [Deployment Guide](../deployment/README.md)
- **Troubleshooting Guide** (see development section)

### External Resources
- [Docker Documentation](https://docs.docker.com/)
- [VNC Setup Guide](https://wiki.archlinux.org/title/TigerVNC)
- [QEMU Documentation](https://qemu.readthedocs.io/)
- [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/)

### Community
- **GitHub Repository**: [https://github.com/yica-ai/yica-yirage](https://github.com/yica-ai/yica-yirage)
- **Issue Tracker**: [https://github.com/yica-ai/yica-yirage/issues](https://github.com/yica-ai/yica-yirage/issues)
- **Discussions**: [https://github.com/yica-ai/yica-yirage/discussions](https://github.com/yica-ai/yica-yirage/discussions)

## 📞 Support

### Getting Help
1. Check this quick reference guide
2. Consult the [full documentation](../README.md)
3. Search [existing issues](https://github.com/yica-ai/yica-yirage/issues)
4. Create a [new issue](https://github.com/yica-ai/yica-yirage/issues/new) with:
   - System information (`yirage diagnose --full`)
   - Error messages and logs
   - Steps to reproduce the problem

### Emergency Procedures
```bash
# Complete system reset
./scripts/docker_yica_deployment.sh clean
./scripts/docker_yica_deployment.sh

# Backup important data
docker cp yica-container:/home/yica/workspace/data ./backup/

# Restore from backup
docker cp ./backup/data yica-container:/home/yica/workspace/
```

---

**Server**: johnson.chen@10.11.60.58
**Workspace**: /home/johnson.chen/yica-docker-workspace
**Documentation Version**: v2.0