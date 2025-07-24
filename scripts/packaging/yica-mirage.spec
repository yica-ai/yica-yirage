Name:           yica-mirage
Version:        1.0.0
Release:        1%{?dist}
Summary:        YICA-Mirage AI Computing Optimization Framework

License:        MIT
URL:            https://github.com/yica-ai/yica-mirage
Source0:        https://github.com/yica-ai/yica-mirage/archive/v%{version}.tar.gz

BuildRequires:  cmake >= 3.18
BuildRequires:  ninja-build
BuildRequires:  gcc-c++
BuildRequires:  python3-devel >= 3.8
BuildRequires:  python3-setuptools
BuildRequires:  python3-pip
BuildRequires:  python3-wheel
BuildRequires:  python3-pybind11-devel
BuildRequires:  z3-devel
BuildRequires:  llvm-devel
BuildRequires:  pkgconfig

Requires:       python3 >= 3.8
Requires:       python3-numpy >= 1.19.0
Requires:       python3-torch >= 1.12.0
Requires:       python3-z3
Requires:       z3

%description
YICA-Mirage is a high-performance AI computing optimization framework
designed for in-memory computing architectures.

Core Features:
- Mirage-based universal code optimization
- YICA in-memory computing architecture specific optimizations
- Automatic Triton code generation
- Multi-backend support (CPU/GPU/YICA)
- Intelligent performance tuning

%package        devel
Summary:        Development files for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}
Requires:       cmake >= 3.18
Requires:       z3-devel
Requires:       python3-pybind11-devel

%description    devel
This package contains header files and static libraries needed to develop
YICA-Mirage applications.

%package        doc
Summary:        Documentation for %{name}
BuildArch:      noarch

%description    doc
This package contains complete documentation for YICA-Mirage, including
API reference, tutorials, and examples.

%package        -n python3-%{name}
Summary:        Python3 bindings for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}
Requires:       python3-numpy >= 1.19.0
Requires:       python3-torch >= 1.12.0

%description    -n python3-%{name}
This package provides Python3 bindings for YICA-Mirage, allowing the use
of YICA optimization functionality in Python applications.

%prep
%autosetup -n %{name}-%{version}

%build
# Build C++ components
%cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_TESTS=OFF \
    -DPYTHON_EXECUTABLE=%{python3}
%cmake_build

# Build Python package
cd mirage/python
%py3_build

%install
# Install C++ components
%cmake_install

# Install Python package
cd mirage/python
%py3_install

# Install documentation
mkdir -p %{buildroot}%{_docdir}/%{name}
cp -r docs/* %{buildroot}%{_docdir}/%{name}/
cp README.md CHANGELOG.md %{buildroot}%{_docdir}/%{name}/

%files
%license LICENSE
%doc README.md CHANGELOG.md
%{_bindir}/yica-optimizer
%{_bindir}/yica-benchmark
%{_bindir}/yica-analyze
%{_libdir}/libyica_*.so.*

%files devel
%{_includedir}/yica/
%{_includedir}/mirage/yica/
%{_libdir}/libyica_*.so
%{_libdir}/libyica_*.a
%{_libdir}/cmake/yica-mirage/
%{_libdir}/pkgconfig/yica-mirage.pc

%files doc
%{_docdir}/%{name}/

%files -n python3-%{name}
%{python3_sitearch}/yica_mirage/
%{python3_sitearch}/yica_mirage-*.egg-info/

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%changelog
* Thu Jan 24 2025 YICA Team <contact@yica.ai> - 1.0.0-1
- Initial release
- Mirage-based universal code optimization
- YICA in-memory computing architecture support
- Automatic Triton code generation
- Multi-backend support (CPU/GPU/YICA)
- Python bindings and command-line tools 