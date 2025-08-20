#!/bin/bash

# FSI Risk Analysis with vLLM on AMD MI300X + roc-finance
# Startup script for advanced financial services implementation

set -e

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/fsi_roc_finance.log"
PID_FILE="$SCRIPT_DIR/fsi_roc_finance.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check if process is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Function to stop the service
stop_service() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        log "Stopping FSI roc-finance service (PID: $pid)..."
        kill "$pid"
        
        # Wait for graceful shutdown
        local count=0
        while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 30 ]; do
            sleep 1
            count=$((count + 1))
        done
        
        if ps -p "$pid" > /dev/null 2>&1; then
            warn "Graceful shutdown failed, forcing kill..."
            kill -9 "$pid"
        fi
        
        rm -f "$PID_FILE"
        log "FSI roc-finance service stopped"
    else
        warn "FSI roc-finance service is not running"
    fi
}

# Function to check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        info "Python version: $PYTHON_VERSION"
        
        # Check if version is 3.8 or higher
        if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
            log "✅ Python version requirement met"
        else
            error "❌ Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        error "❌ Python 3 not found"
        exit 1
    fi
    
    # Check for ROCm installation
    if command -v rocm-smi &> /dev/null; then
        log "✅ ROCm detected"
        rocm-smi --showproductname 2>/dev/null | head -5 || true
    else
        warn "⚠️  ROCm not detected - GPU acceleration may not be available"
    fi
    
    # Check for AMD GPU
    if lspci | grep -i amd | grep -i vga &> /dev/null; then
        log "✅ AMD GPU detected"
    else
        warn "⚠️  AMD GPU not detected - running in CPU mode"
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df "$SCRIPT_DIR" | awk 'NR==2 {print $4}')
    if [ "$AVAILABLE_SPACE" -gt 5242880 ]; then  # 5GB in KB
        log "✅ Sufficient disk space available"
    else
        warn "⚠️  Low disk space - recommend at least 5GB free"
    fi
    
    # Check memory
    AVAILABLE_MEM=$(free -m | awk 'NR==2{print $7}')
    if [ "$AVAILABLE_MEM" -gt 8192 ]; then  # 8GB
        log "✅ Sufficient memory available"
    else
        warn "⚠️  Low memory - recommend at least 8GB available"
    fi
}

# Function to setup environment
setup_environment() {
    log "Setting up environment for AMD MI300X..."
    
    # ROCm environment variables
    export PYTORCH_ROCM_ARCH=gfx942
    export HSA_OVERRIDE_GFX_VERSION=9.4.2
    export HIP_VISIBLE_DEVICES=0
    export ROCM_PATH=/opt/rocm
    export HSA_PATH=/opt/rocm
    
    # Python optimizations
    export PYTHONUNBUFFERED=1
    export PYTHONDONTWRITEBYTECODE=1
    
    # vLLM optimizations for AMD
    export VLLM_USE_ROCM=1
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    
    # Memory management
    export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
    
    info "Environment variables set for AMD MI300X optimization"
}

# Function to install dependencies
install_dependencies() {
    log "Installing/updating dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$SCRIPT_DIR/venv" ]; then
        log "Creating virtual environment..."
        python3 -m venv "$SCRIPT_DIR/venv"
    fi
    
    # Activate virtual environment
    source "$SCRIPT_DIR/venv/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    if [ -f "$SCRIPT_DIR/requirements_minimal.txt" ]; then
        log "Installing minimal requirements..."
        pip install -r "$SCRIPT_DIR/requirements_minimal.txt"
    elif [ -f "$SCRIPT_DIR/requirements_simplified.txt" ]; then
        log "Installing simplified requirements..."
        pip install -r "$SCRIPT_DIR/requirements_simplified.txt"
    elif [ -f "$SCRIPT_DIR/requirements_roc_finance.txt" ]; then
        log "Installing roc-finance requirements..."
        pip install -r "$SCRIPT_DIR/requirements_roc_finance.txt"
    else
        error "No requirements file found"
        exit 1
    fi
    
    log "✅ Dependencies installed successfully"
}

# Function to run health checks
health_check() {
    log "Running health checks..."
    
    # Activate virtual environment
    source "$SCRIPT_DIR/venv/bin/activate"
    
    # Test basic imports
    python3 -c "
import sys
print(f'Python version: {sys.version}')

# Test PyTorch
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} imported successfully')
    if torch.cuda.is_available():
        print(f'✅ CUDA/ROCm available: {torch.cuda.get_device_name()}')
    else:
        print('⚠️  CUDA/ROCm not available - running in CPU mode')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')

# Test vLLM
try:
    import vllm
    print(f'✅ vLLM {vllm.__version__} imported successfully')
except ImportError as e:
    print(f'⚠️  vLLM import failed: {e} - will use fallback')

# Test financial libraries
try:
    import yfinance
    print('✅ yfinance imported successfully')
except ImportError as e:
    print(f'❌ yfinance import failed: {e}')

try:
    import gradio
    print(f'✅ Gradio {gradio.__version__} imported successfully')
except ImportError as e:
    print(f'❌ Gradio import failed: {e}')

try:
    import pandas
    print(f'✅ Pandas {pandas.__version__} imported successfully')
except ImportError as e:
    print(f'❌ Pandas import failed: {e}')

print('Health check completed')
" || error "Health check failed"
}

# Function to start the service
start_service() {
    if is_running; then
        warn "FSI roc-finance service is already running (PID: $(cat "$PID_FILE"))"
        return 0
    fi
    
    log "Starting FSI Risk Analysis with vLLM + roc-finance..."
    
    # Activate virtual environment
    source "$SCRIPT_DIR/venv/bin/activate"
    
    # Start the service in background
    cd "$SCRIPT_DIR"
    nohup python3 fsi_roc_finance_vllm.py > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    
    # Wait a moment and check if it started successfully
    sleep 5
    if ps -p "$pid" > /dev/null 2>&1; then
        log "✅ FSI roc-finance service started successfully (PID: $pid)"
        log "🌐 Web interface should be available at: http://localhost:7860"
        log "📊 Monitor logs with: tail -f $LOG_FILE"
        return 0
    else
        error "❌ Failed to start FSI roc-finance service"
        rm -f "$PID_FILE"
        return 1
    fi
}

# Function to show status
show_status() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        log "FSI roc-finance service is running (PID: $pid)"
        
        # Show resource usage
        if command -v ps &> /dev/null; then
            local cpu_mem=$(ps -p "$pid" -o pid,pcpu,pmem,etime,cmd --no-headers 2>/dev/null || echo "Process info unavailable")
            info "Process info: $cpu_mem"
        fi
        
        # Check if web interface is accessible
        if command -v curl &> /dev/null; then
            if curl -s -o /dev/null -w "%{http_code}" http://localhost:7860 | grep -q "200"; then
                log "🌐 Web interface is accessible at http://localhost:7860"
            else
                warn "⚠️  Web interface may not be ready yet"
            fi
        fi
    else
        warn "FSI roc-finance service is not running"
    fi
}

# Function to show logs
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -n 50 "$LOG_FILE"
    else
        warn "Log file not found: $LOG_FILE"
    fi
}

# Main execution
main() {
    case "${1:-start}" in
        "start")
            log "🚀 Starting FSI Risk Analysis with roc-finance..."
            check_requirements
            setup_environment
            install_dependencies
            health_check
            start_service
            ;;
        "stop")
            stop_service
            ;;
        "restart")
            stop_service
            sleep 2
            main start
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "install")
            check_requirements
            setup_environment
            install_dependencies
            health_check
            ;;
        "health")
            setup_environment
            health_check
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|status|logs|install|health}"
            echo ""
            echo "Commands:"
            echo "  start    - Start the FSI roc-finance service"
            echo "  stop     - Stop the FSI roc-finance service"
            echo "  restart  - Restart the FSI roc-finance service"
            echo "  status   - Show service status"
            echo "  logs     - Show recent logs"
            echo "  install  - Install dependencies only"
            echo "  health   - Run health checks only"
            exit 1
            ;;
    esac
}

# Trap to handle cleanup on script exit
trap 'echo -e "\n${YELLOW}Script interrupted${NC}"' INT TERM

# Run main function
main "$@"
