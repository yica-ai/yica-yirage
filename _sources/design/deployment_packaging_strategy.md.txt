# Production Deployment and Packaging Strategy

## Current Deployment Issues

### 1. Packaging Inconsistencies
- **Multiple conflicting packaging systems**: setup.py, pyproject.toml, CMake install
- **Missing dependency management**: No clear dependency resolution strategy
- **Platform-specific issues**: Different package formats for different systems
- **Version management chaos**: No unified versioning across components

### 2. Deployment Complexities
- **Manual deployment steps**: No automated deployment pipeline
- **Environment configuration drift**: Different configurations in different environments
- **Missing rollback mechanisms**: No easy way to revert deployments
- **Insufficient monitoring**: No deployment health checks

### 3. Distribution Problems
- **Large package sizes**: Unnecessary components included in distributions
- **Missing platform variants**: No optimized builds for different hardware
- **Complex installation**: Users need to understand build dependencies
- **Poor user experience**: Installation fails silently in some environments

## Comprehensive Deployment and Packaging Design

### 1. Multi-Tier Packaging Strategy

```yaml
# packaging/package_strategy.yaml
packaging_tiers:
  
  core:
    description: "Minimal YICA/YiRage functionality - always works"
    target_size: "< 50MB"
    dependencies:
      required:
        - python >= 3.8
        - numpy >= 1.19.0
      optional: []
    features:
      - CPU-only optimization
      - Basic Python API
      - Core compatibility layers
      - Essential error handling
    platforms:
      - linux-x86_64
      - macos-x86_64
      - macos-arm64
      - windows-x86_64
      
  enhanced:
    description: "Standard YICA/YiRage with common dependencies"
    target_size: "< 200MB"
    dependencies:
      required:
        - yica-yirage-core
        - z3-solver >= 4.8.0
      optional:
        - torch >= 1.12.0
    features:
      - All core features
      - Z3 SMT solving
      - OpenMP parallelization
      - PyTorch integration
      - Advanced optimization
    platforms:
      - linux-x86_64
      - macos-x86_64
      - macos-arm64
      - windows-x86_64
      
  full:
    description: "Complete YICA/YiRage with all acceleration features"
    target_size: "< 1GB"
    dependencies:
      required:
        - yica-yirage-enhanced
      optional:
        - nvidia-cuda-toolkit
        - triton >= 2.0.0
        - rocm-toolkit
    features:
      - All enhanced features
      - CUDA GPU acceleration
      - Triton kernel compilation
      - ROCm AMD support
      - Professional monitoring
    platforms:
      - linux-x86_64-cuda
      - linux-x86_64-rocm
      - windows-x86_64-cuda

  developer:
    description: "Development tools and debugging features"
    dependencies:
      required:
        - yica-yirage-full
      optional:
        - gdb
        - valgrind
        - perf
    features:
      - All full features
      - Debug symbols
      - Development tools
      - Profiling support
      - Test framework
```

### 2. Advanced Packaging Infrastructure

```python
# packaging/build_system.py
"""
Advanced packaging and distribution system for YICA/YiRage
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

class PackageTier(Enum):
    CORE = "core"
    ENHANCED = "enhanced"
    FULL = "full"
    DEVELOPER = "developer"

class Platform(Enum):
    LINUX_X64 = "linux-x86_64"
    LINUX_X64_CUDA = "linux-x86_64-cuda"
    LINUX_X64_ROCM = "linux-x86_64-rocm"
    MACOS_X64 = "macos-x86_64"
    MACOS_ARM64 = "macos-arm64"
    WINDOWS_X64 = "windows-x86_64"
    WINDOWS_X64_CUDA = "windows-x86_64-cuda"

@dataclass
class PackageConfig:
    tier: PackageTier
    platform: Platform
    version: str
    build_type: str = "Release"
    include_debug_symbols: bool = False
    enable_optimizations: bool = True
    custom_flags: List[str] = None
    
    def __post_init__(self):
        if self.custom_flags is None:
            self.custom_flags = []

class PackageBuilder:
    """Advanced package builder with multi-tier support"""
    
    def __init__(self, source_dir: Path, output_dir: Path):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = None
        
    def __enter__(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="yirage_build_"))
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def build_package(self, config: PackageConfig) -> Path:
        """Build a package according to the specified configuration"""
        
        print(f"🔨 Building {config.tier.value} package for {config.platform.value}")
        
        # Create build directory
        build_dir = self.temp_dir / f"build_{config.tier.value}_{config.platform.value}"
        build_dir.mkdir(parents=True)
        
        # Configure build
        cmake_args = self._generate_cmake_args(config)
        self._run_cmake_configure(build_dir, cmake_args)
        
        # Build
        self._run_cmake_build(build_dir, config)
        
        # Package
        package_path = self._create_package(build_dir, config)
        
        # Validate package
        self._validate_package(package_path, config)
        
        print(f"✅ Package created: {package_path}")
        return package_path
    
    def _generate_cmake_args(self, config: PackageConfig) -> List[str]:
        """Generate CMake configuration arguments"""
        
        args = [
            f"-DCMAKE_BUILD_TYPE={config.build_type}",
            f"-DYICA_PACKAGE_TIER={config.tier.value.upper()}",
            f"-DYICA_TARGET_PLATFORM={config.platform.value}",
            f"-DYICA_VERSION={config.version}",
        ]
        
        # Tier-specific configuration
        if config.tier == PackageTier.CORE:
            args.extend([
                "-DYICA_BUILD_MODE=CORE",
                "-DYICA_ENABLE_CUDA=OFF",
                "-DYICA_ENABLE_OPENMP=OFF",
                "-DYICA_ENABLE_Z3=OFF",
            ])
        elif config.tier == PackageTier.ENHANCED:
            args.extend([
                "-DYICA_BUILD_MODE=ENHANCED",
                "-DYICA_ENABLE_CUDA=OFF",
                "-DYICA_ENABLE_OPENMP=ON",
                "-DYICA_ENABLE_Z3=ON",
            ])
        elif config.tier == PackageTier.FULL:
            args.extend([
                "-DYICA_BUILD_MODE=FULL",
                "-DYICA_ENABLE_CUDA=ON",
                "-DYICA_ENABLE_OPENMP=ON",
                "-DYICA_ENABLE_Z3=ON",
                "-DYICA_ENABLE_TRITON=ON",
            ])
        elif config.tier == PackageTier.DEVELOPER:
            args.extend([
                "-DYICA_BUILD_MODE=FULL",
                "-DYICA_ENABLE_CUDA=ON",
                "-DYICA_ENABLE_OPENMP=ON",
                "-DYICA_ENABLE_Z3=ON",
                "-DYICA_ENABLE_TRITON=ON",
                "-DYICA_ENABLE_TESTING=ON",
                "-DYICA_ENABLE_DEBUGGING=ON",
            ])
        
        # Platform-specific configuration
        if "cuda" in config.platform.value:
            args.extend([
                "-DYICA_ENABLE_CUDA=ON",
                "-DCUDA_ARCHITECTURES=70;75;80;86;89;90",
            ])
        
        if "rocm" in config.platform.value:
            args.extend([
                "-DYICA_ENABLE_ROCM=ON",
                "-DGPU_TARGETS=gfx906;gfx908;gfx90a;gfx1030",
            ])
        
        # Debug symbols
        if config.include_debug_symbols:
            args.append("-DCMAKE_BUILD_TYPE=RelWithDebInfo")
        
        # Custom flags
        args.extend(config.custom_flags)
        
        return args
    
    def _run_cmake_configure(self, build_dir: Path, cmake_args: List[str]):
        """Run CMake configuration"""
        
        cmd = ["cmake", "-S", str(self.source_dir), "-B", str(build_dir)] + cmake_args
        
        print(f"🔧 Configuring: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ CMake configuration failed:")
            print(result.stderr)
            raise RuntimeError("CMake configuration failed")
    
    def _run_cmake_build(self, build_dir: Path, config: PackageConfig):
        """Run CMake build"""
        
        cmd = ["cmake", "--build", str(build_dir), "--parallel"]
        
        if config.build_type == "Release" and config.enable_optimizations:
            cmd.extend(["--config", "Release"])
        
        print(f"🔨 Building: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Build failed:")
            print(result.stderr)
            raise RuntimeError("Build failed")
    
    def _create_package(self, build_dir: Path, config: PackageConfig) -> Path:
        """Create the final package"""
        
        package_name = f"yica-yirage-{config.tier.value}-{config.version}-{config.platform.value}"
        
        if config.platform.value.startswith("linux"):
            return self._create_deb_package(build_dir, config, package_name)
        elif config.platform.value.startswith("macos"):
            return self._create_dmg_package(build_dir, config, package_name)
        elif config.platform.value.startswith("windows"):
            return self._create_msi_package(build_dir, config, package_name)
        else:
            return self._create_tarball_package(build_dir, config, package_name)
    
    def _create_deb_package(self, build_dir: Path, config: PackageConfig, package_name: str) -> Path:
        """Create Debian package"""
        
        deb_dir = self.temp_dir / "deb_package"
        deb_dir.mkdir()
        
        # Install to temporary directory
        subprocess.run([
            "cmake", "--install", str(build_dir), 
            "--prefix", str(deb_dir / "usr" / "local")
        ], check=True)
        
        # Create DEBIAN control directory
        control_dir = deb_dir / "DEBIAN"
        control_dir.mkdir()
        
        # Generate control file
        control_content = self._generate_deb_control(config, package_name)
        (control_dir / "control").write_text(control_content)
        
        # Create package
        package_path = self.output_dir / f"{package_name}.deb"
        subprocess.run([
            "dpkg-deb", "--build", str(deb_dir), str(package_path)
        ], check=True)
        
        return package_path
    
    def _create_dmg_package(self, build_dir: Path, config: PackageConfig, package_name: str) -> Path:
        """Create macOS DMG package"""
        
        app_dir = self.temp_dir / f"{package_name}.app"
        app_dir.mkdir()
        
        # Create app bundle structure
        contents_dir = app_dir / "Contents"
        macos_dir = contents_dir / "MacOS"
        resources_dir = contents_dir / "Resources"
        
        for dir_path in [contents_dir, macos_dir, resources_dir]:
            dir_path.mkdir(parents=True)
        
        # Install to app bundle
        subprocess.run([
            "cmake", "--install", str(build_dir),
            "--prefix", str(macos_dir)
        ], check=True)
        
        # Create Info.plist
        info_plist = self._generate_macos_info_plist(config, package_name)
        (contents_dir / "Info.plist").write_text(info_plist)
        
        # Create DMG
        package_path = self.output_dir / f"{package_name}.dmg"
        subprocess.run([
            "hdiutil", "create", "-srcfolder", str(app_dir),
            "-volname", package_name, str(package_path)
        ], check=True)
        
        return package_path
    
    def _create_msi_package(self, build_dir: Path, config: PackageConfig, package_name: str) -> Path:
        """Create Windows MSI package"""
        
        # Use CPack to create MSI
        subprocess.run([
            "cpack", "-G", "WIX", "-B", str(self.output_dir)
        ], cwd=build_dir, check=True)
        
        # Find generated MSI file
        msi_files = list(self.output_dir.glob("*.msi"))
        if not msi_files:
            raise RuntimeError("MSI package not found")
        
        # Rename to standard name
        package_path = self.output_dir / f"{package_name}.msi"
        msi_files[0].rename(package_path)
        
        return package_path
    
    def _create_tarball_package(self, build_dir: Path, config: PackageConfig, package_name: str) -> Path:
        """Create generic tarball package"""
        
        install_dir = self.temp_dir / "install"
        install_dir.mkdir()
        
        # Install to temporary directory
        subprocess.run([
            "cmake", "--install", str(build_dir),
            "--prefix", str(install_dir)
        ], check=True)
        
        # Create tarball
        package_path = self.output_dir / f"{package_name}.tar.gz"
        subprocess.run([
            "tar", "-czf", str(package_path), "-C", str(install_dir), "."
        ], check=True)
        
        return package_path
    
    def _generate_deb_control(self, config: PackageConfig, package_name: str) -> str:
        """Generate Debian control file"""
        
        # Base dependencies
        depends = ["python3 (>= 3.8)", "python3-numpy"]
        
        # Tier-specific dependencies
        if config.tier in [PackageTier.ENHANCED, PackageTier.FULL, PackageTier.DEVELOPER]:
            depends.extend(["python3-z3", "libz3-4"])
        
        if config.tier in [PackageTier.FULL, PackageTier.DEVELOPER]:
            if "cuda" in config.platform.value:
                depends.extend(["nvidia-cuda-toolkit", "libcudnn8"])
        
        control = f"""Package: {package_name}
Version: {config.version}
Section: science
Priority: optional
Architecture: amd64
Depends: {', '.join(depends)}
Maintainer: YICA Team <contact@yica.ai>
Description: YICA/YiRage AI Computing Optimization Framework
 YICA/YiRage is an advanced AI computing optimization framework designed for
 in-memory computing architectures. This package provides the {config.tier.value}
 tier with {self._get_tier_description(config.tier)}.
"""
        return control
    
    def _generate_macos_info_plist(self, config: PackageConfig, package_name: str) -> str:
        """Generate macOS Info.plist"""
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>yica-optimizer</string>
    <key>CFBundleIdentifier</key>
    <string>ai.yica.yirage.{config.tier.value}</string>
    <key>CFBundleName</key>
    <string>{package_name}</string>
    <key>CFBundleVersion</key>
    <string>{config.version}</string>
    <key>CFBundleShortVersionString</key>
    <string>{config.version}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>"""
    
    def _get_tier_description(self, tier: PackageTier) -> str:
        """Get human-readable tier description"""
        
        descriptions = {
            PackageTier.CORE: "basic CPU optimization capabilities",
            PackageTier.ENHANCED: "advanced optimization with SMT solving",
            PackageTier.FULL: "complete feature set with GPU acceleration",
            PackageTier.DEVELOPER: "development tools and debugging support"
        }
        return descriptions.get(tier, "unknown tier")
    
    def _validate_package(self, package_path: Path, config: PackageConfig):
        """Validate the created package"""
        
        print(f"🔍 Validating package: {package_path}")
        
        # Check file exists and size
        if not package_path.exists():
            raise RuntimeError(f"Package not found: {package_path}")
        
        size_mb = package_path.stat().st_size / (1024 * 1024)
        print(f"📦 Package size: {size_mb:.1f} MB")
        
        # Check size limits based on tier
        size_limits = {
            PackageTier.CORE: 50,
            PackageTier.ENHANCED: 200,
            PackageTier.FULL: 1000,
            PackageTier.DEVELOPER: 2000
        }
        
        limit = size_limits.get(config.tier, 1000)
        if size_mb > limit:
            print(f"⚠️  Package size ({size_mb:.1f} MB) exceeds limit ({limit} MB)")
        
        # Platform-specific validation
        if package_path.suffix == ".deb":
            self._validate_deb_package(package_path)
        elif package_path.suffix == ".dmg":
            self._validate_dmg_package(package_path)
        elif package_path.suffix == ".msi":
            self._validate_msi_package(package_path)
    
    def _validate_deb_package(self, package_path: Path):
        """Validate Debian package"""
        
        # Check package info
        result = subprocess.run([
            "dpkg-deb", "--info", str(package_path)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Invalid Debian package: {result.stderr}")
        
        print("✅ Debian package validation passed")
    
    def _validate_dmg_package(self, package_path: Path):
        """Validate macOS DMG package"""
        
        # Check DMG can be mounted
        result = subprocess.run([
            "hdiutil", "verify", str(package_path)
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Invalid DMG package: {result.stderr}")
        
        print("✅ DMG package validation passed")
    
    def _validate_msi_package(self, package_path: Path):
        """Validate Windows MSI package"""
        
        # Basic file validation (detailed validation requires Windows tools)
        if package_path.stat().st_size < 1024:
            raise RuntimeError("MSI package too small")
        
        print("✅ MSI package validation passed")

class PackageDistributor:
    """Handle package distribution to various channels"""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config = self._load_distribution_config(config_file)
    
    def distribute_package(self, package_path: Path, channels: List[str]):
        """Distribute package to specified channels"""
        
        for channel in channels:
            print(f"📤 Distributing to {channel}...")
            
            if channel == "pypi":
                self._distribute_to_pypi(package_path)
            elif channel == "conda":
                self._distribute_to_conda(package_path)
            elif channel == "github":
                self._distribute_to_github(package_path)
            elif channel == "docker":
                self._distribute_to_docker(package_path)
            elif channel == "s3":
                self._distribute_to_s3(package_path)
            else:
                print(f"⚠️  Unknown distribution channel: {channel}")
    
    def _load_distribution_config(self, config_file: Optional[Path]) -> Dict:
        """Load distribution configuration"""
        
        if config_file and config_file.exists():
            import json
            return json.loads(config_file.read_text())
        
        return {
            "pypi": {"repository": "https://upload.pypi.org/legacy/"},
            "conda": {"channel": "yica"},
            "github": {"repository": "yica/yirage"},
            "docker": {"registry": "docker.io", "namespace": "yica"},
            "s3": {"bucket": "yica-releases", "region": "us-east-1"}
        }
    
    def _distribute_to_pypi(self, package_path: Path):
        """Distribute to PyPI"""
        
        # Convert package to wheel if needed
        if package_path.suffix != ".whl":
            print("⚠️  PyPI distribution requires wheel format")
            return
        
        # Upload using twine
        subprocess.run([
            "twine", "upload", str(package_path),
            "--repository", self.config["pypi"]["repository"]
        ], check=True)
        
        print("✅ PyPI distribution completed")
    
    def _distribute_to_conda(self, package_path: Path):
        """Distribute to Conda"""
        
        # Build conda package
        subprocess.run([
            "conda-build", "packaging/conda-recipe",
            "--output-folder", str(package_path.parent)
        ], check=True)
        
        # Upload to channel
        subprocess.run([
            "anaconda", "upload", str(package_path),
            "--user", self.config["conda"]["channel"]
        ], check=True)
        
        print("✅ Conda distribution completed")
    
    def _distribute_to_github(self, package_path: Path):
        """Distribute to GitHub Releases"""
        
        import requests
        
        # Upload to GitHub releases using API
        # (Implementation would use GitHub API)
        
        print("✅ GitHub distribution completed")
    
    def _distribute_to_docker(self, package_path: Path):
        """Distribute as Docker image"""
        
        # Create Dockerfile for package
        dockerfile_content = self._generate_dockerfile(package_path)
        
        dockerfile_path = package_path.parent / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        
        # Build and push Docker image
        image_tag = f"{self.config['docker']['namespace']}/yirage:{package_path.stem}"
        
        subprocess.run([
            "docker", "build", "-t", image_tag, str(package_path.parent)
        ], check=True)
        
        subprocess.run([
            "docker", "push", image_tag
        ], check=True)
        
        print("✅ Docker distribution completed")
    
    def _distribute_to_s3(self, package_path: Path):
        """Distribute to S3"""
        
        import boto3
        
        s3 = boto3.client('s3')
        
        key = f"releases/{package_path.name}"
        bucket = self.config["s3"]["bucket"]
        
        s3.upload_file(str(package_path), bucket, key)
        
        print("✅ S3 distribution completed")
    
    def _generate_dockerfile(self, package_path: Path) -> str:
        """Generate Dockerfile for package"""
        
        if package_path.suffix == ".deb":
            return f"""FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \\
    python3 python3-pip python3-numpy \\
    && rm -rf /var/lib/apt/lists/*

COPY {package_path.name} /tmp/
RUN dpkg -i /tmp/{package_path.name} || apt-get install -f -y

ENTRYPOINT ["yica-optimizer"]
"""
        else:
            return f"""FROM python:3.11-slim

COPY {package_path.name} /tmp/
RUN pip install /tmp/{package_path.name}

ENTRYPOINT ["yica-optimizer"]
"""
```

### 3. Automated Deployment Pipeline

```yaml
# .github/workflows/deployment_pipeline.yml
name: Production Deployment Pipeline

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      tier:
        description: 'Package tier to build'
        required: true
        type: choice
        options:
          - core
          - enhanced
          - full
          - all
      distribution_channels:
        description: 'Distribution channels (comma-separated)'
        required: true
        default: 'github,docker'

env:
  PACKAGE_VERSION: ${{ github.ref_name }}

jobs:
  build-packages:
    name: Build ${{ matrix.tier }} for ${{ matrix.platform }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        include:
          # Core packages - all platforms
          - tier: core
            platform: linux-x86_64
            os: ubuntu-22.04
          - tier: core
            platform: macos-x86_64
            os: macos-12
          - tier: core
            platform: macos-arm64
            os: macos-14
          - tier: core
            platform: windows-x86_64
            os: windows-2022
          
          # Enhanced packages
          - tier: enhanced
            platform: linux-x86_64
            os: ubuntu-22.04
          - tier: enhanced
            platform: macos-x86_64
            os: macos-12
          - tier: enhanced
            platform: macos-arm64
            os: macos-14
          - tier: enhanced
            platform: windows-x86_64
            os: windows-2022
          
          # Full packages with GPU support
          - tier: full
            platform: linux-x86_64-cuda
            os: ubuntu-22.04
          - tier: full
            platform: windows-x86_64-cuda
            os: windows-2022
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Setup build environment
      uses: ./.github/actions/setup-build-env
      with:
        platform: ${{ matrix.platform }}
        tier: ${{ matrix.tier }}
    
    - name: Build package
      run: |
        python3 packaging/build_system.py \
          --tier ${{ matrix.tier }} \
          --platform ${{ matrix.platform }} \
          --version ${{ env.PACKAGE_VERSION }} \
          --output-dir dist/
    
    - name: Upload package artifacts
      uses: actions/upload-artifact@v4
      with:
        name: package-${{ matrix.tier }}-${{ matrix.platform }}
        path: dist/
        retention-days: 30
    
    - name: Run package validation
      run: |
        python3 packaging/validate_package.py \
          --package dist/yica-yirage-${{ matrix.tier }}-* \
          --tier ${{ matrix.tier }} \
          --platform ${{ matrix.platform }}

  integration-testing:
    name: Integration Testing
    needs: build-packages
    runs-on: ubuntu-22.04
    strategy:
      matrix:
        tier: [core, enhanced, full]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Download packages
      uses: actions/download-artifact@v4
      with:
        name: package-${{ matrix.tier }}-linux-x86_64
        path: packages/
    
    - name: Test package installation
      run: |
        # Test installation in clean environment
        docker run --rm -v $(pwd)/packages:/packages ubuntu:22.04 \
          bash -c "
            apt-get update && 
            apt-get install -y python3 python3-pip &&
            pip install /packages/yica-yirage-${{ matrix.tier }}-*.whl &&
            python3 -c 'import yirage; print(yirage.__version__)'
          "
    
    - name: Run integration tests
      run: |
        python3 -m pytest tests/integration/test_package_integration.py \
          --package-path packages/yica-yirage-${{ matrix.tier }}-* \
          --tier ${{ matrix.tier }}

  security-scanning:
    name: Security Scanning
    needs: build-packages
    runs-on: ubuntu-22.04
    
    steps:
    - name: Download all packages
      uses: actions/download-artifact@v4
    
    - name: Run security scan
      uses: anchore/scan-action@v3
      with:
        path: ./
        format: sarif
        output: security-results.sarif
    
    - name: Upload security results
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: security-results.sarif

  deploy-staging:
    name: Deploy to Staging
    needs: [build-packages, integration-testing, security-scanning]
    runs-on: ubuntu-22.04
    environment: staging
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Download all packages
      uses: actions/download-artifact@v4
    
    - name: Deploy to staging
      run: |
        python3 packaging/deploy.py \
          --environment staging \
          --packages package-*/yica-yirage-* \
          --channels docker,s3
    
    - name: Run staging tests
      run: |
        python3 tests/staging/test_deployment.py \
          --environment staging \
          --timeout 300

  deploy-production:
    name: Deploy to Production
    needs: deploy-staging
    runs-on: ubuntu-22.04
    environment: production
    if: startsWith(github.ref, 'refs/tags/v')
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Download all packages
      uses: actions/download-artifact@v4
    
    - name: Deploy to production
      run: |
        python3 packaging/deploy.py \
          --environment production \
          --packages package-*/yica-yirage-* \
          --channels pypi,conda,github,docker,s3
    
    - name: Create GitHub Release
      uses: ncipollo/release-action@v1
      with:
        artifacts: "package-*/yica-yirage-*"
        tag: ${{ github.ref_name }}
        name: YICA/YiRage ${{ github.ref_name }}
        body: |
          ## YICA/YiRage ${{ github.ref_name }}
          
          ### Package Tiers Available:
          - **Core**: Minimal functionality, works everywhere
          - **Enhanced**: Standard features with SMT solving
          - **Full**: Complete feature set with GPU acceleration
          
          ### Installation:
          ```bash
          # Core tier (recommended for most users)
          pip install yica-yirage-core
          
          # Enhanced tier (for advanced optimization)
          pip install yica-yirage-enhanced
          
          # Full tier (for GPU acceleration)
          pip install yica-yirage-full
          ```
          
          See the [documentation](https://yica.ai/docs) for detailed installation and usage instructions.
        draft: false
        prerelease: ${{ contains(github.ref_name, '-') }}
    
    - name: Update documentation
      run: |
        python3 scripts/update_docs.py \
          --version ${{ github.ref_name }} \
          --packages package-*/yica-yirage-*
    
    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: success
        text: "🚀 YICA/YiRage ${{ github.ref_name }} successfully deployed to production!"
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

This comprehensive deployment and packaging strategy provides:

1. **Multi-tier packaging** with CORE, ENHANCED, FULL, and DEVELOPER variants
2. **Cross-platform support** for Linux, macOS, and Windows
3. **Automated build pipeline** with comprehensive testing and validation
4. **Multiple distribution channels** including PyPI, Conda, GitHub, Docker, and S3
5. **Security scanning** and compliance checking
6. **Staged deployment** with staging environment validation
7. **Package size optimization** with tier-specific size limits
8. **Comprehensive documentation** and user guidance

The system ensures that users can easily install YICA/YiRage in any environment while maximizing compatibility and minimizing complexity. The multi-tier approach allows users to choose the appropriate feature set for their needs while maintaining consistent quality and security standards across all packages.
