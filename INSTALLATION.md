# Installation Guide

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Standard Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-business-analytics-dashboard.git
   cd ai-business-analytics-dashboard
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python BV.py
   ```

5. **Access the dashboard**
   - Open your browser
   - Navigate to `http://127.0.0.1:8050/`

## Troubleshooting Common Issues

### 🐼 Pandas Build Error

**Error**: `error occurred during the preparation of metadata (pyproject.toml) for the package pandas`

**Solutions**:

#### Option 1: Use Minimal Requirements
```bash
pip install -r requirements-minimal.txt
```

#### Option 2: Install Pre-compiled Binaries
```bash
# For Windows
pip install --only-binary=all pandas numpy

# For macOS/Linux
pip install --only-binary=all pandas numpy
```

#### Option 3: Use Conda (Recommended for Windows)
```bash
# Install Miniconda first, then:
conda create -n dashboard python=3.9
conda activate dashboard
conda install pandas numpy scikit-learn
pip install dash plotly flask
```

#### Option 4: Update Build Tools
```bash
# Update pip and setuptools
python -m pip install --upgrade pip setuptools wheel

# Install build dependencies
pip install --upgrade build
```

### 🔧 Other Common Issues

#### Python Version Issues
- **Problem**: "Python version not supported"
- **Solution**: Use Python 3.8-3.11 (avoid 3.12 for now)

#### Memory Issues
- **Problem**: "Out of memory" during installation
- **Solution**: 
  ```bash
  pip install --no-cache-dir -r requirements.txt
  ```

#### Permission Issues
- **Problem**: "Permission denied" on Windows
- **Solution**: Run as Administrator or use:
  ```bash
  pip install --user -r requirements.txt
  ```

#### Network Issues
- **Problem**: "Connection timeout" or "SSL errors"
- **Solution**:
  ```bash
  pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
  ```

## Alternative Installation Methods

### Using Conda (Recommended)

1. **Install Miniconda**
   - Download from: https://docs.conda.io/en/latest/miniconda.html

2. **Create environment**
   ```bash
   conda create -n dashboard python=3.9
   conda activate dashboard
   ```

3. **Install packages**
   ```bash
   conda install pandas numpy scikit-learn
   pip install dash plotly flask
   ```

### Using Docker

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   EXPOSE 8050
   
   CMD ["python", "BV.py"]
   ```

2. **Build and run**
   ```bash
   docker build -t dashboard .
   docker run -p 8050:8050 dashboard
   ```

### Using Poetry

1. **Install Poetry**
   ```bash
   pip install poetry
   ```

2. **Install dependencies**
   ```bash
   poetry install
   poetry run python BV.py
   ```

## Platform-Specific Instructions

### Windows

#### Prerequisites
- Install Visual Studio Build Tools
- Install Windows SDK

#### Installation Steps
```bash
# Install build tools
pip install --upgrade setuptools wheel

# Install packages
pip install -r requirements.txt
```

### macOS

#### Prerequisites
- Install Xcode Command Line Tools
```bash
xcode-select --install
```

#### Installation Steps
```bash
# Install packages
pip install -r requirements.txt
```

### Linux (Ubuntu/Debian)

#### Prerequisites
```bash
sudo apt-get update
sudo apt-get install python3-dev build-essential
```

#### Installation Steps
```bash
pip install -r requirements.txt
```

## Verification

After installation, verify everything works:

```bash
# Test imports
python -c "import dash, plotly, pandas, numpy, sklearn; print('All packages imported successfully!')"

# Run the application
python BV.py
```

## Performance Optimization

### For Large Datasets
```bash
# Install additional packages for better performance
pip install numba cython
```

### For Development
```bash
# Install development tools
pip install black flake8 mypy pytest
```

## Getting Help

If you still encounter issues:

1. **Check Python version**: `python --version`
2. **Check pip version**: `pip --version`
3. **Check system architecture**: `python -c "import platform; print(platform.architecture())"`
4. **Create issue** on GitHub with:
   - Error message
   - Python version
   - Operating system
   - Installation method used

## Common Error Messages and Solutions

| Error | Solution |
|-------|----------|
| `Microsoft Visual C++ 14.0 is required` | Install Visual Studio Build Tools |
| `Permission denied` | Use `--user` flag or run as admin |
| `SSL: CERTIFICATE_VERIFY_FAILED` | Use `--trusted-host` flags |
| `Out of memory` | Use `--no-cache-dir` flag |
| `No module named 'pandas'` | Reinstall with `pip install --force-reinstall pandas` |

## Support

- **GitHub Issues**: Create an issue with detailed error information
- **Documentation**: Check the README.md and API docs
- **Community**: Ask questions in GitHub Discussions 