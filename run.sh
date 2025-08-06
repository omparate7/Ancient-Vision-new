#!/bin/bash

echo "🎨 Ancient Vision - Starting Traditional Art Transformation App"
echo "=============================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

# Check Node.js
if ! command_exists node; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi

# Check npm
if ! command_exists npm; then
    echo -e "${RED}❌ npm is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating Python virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Install Python dependencies
echo -e "${YELLOW}📦 Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Create necessary directories
echo -e "${YELLOW}📁 Creating necessary directories...${NC}"
mkdir -p models
mkdir -p uploads
mkdir -p modules/statue_restoration/uploads

# Install Node.js dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}📦 Installing Node.js dependencies...${NC}"
    cd frontend
    npm install
    cd ..
fi

# Verify statue restoration module
echo -e "${YELLOW}🏛️ Verifying statue restoration module...${NC}"
if [ ! -f "modules/statue_restoration/statue_restoration.py" ]; then
    echo -e "${RED}❌ Statue restoration module not found${NC}"
    exit 1
fi

if [ ! -d "modules/statue_restoration/weights" ]; then
    echo -e "${RED}❌ Statue restoration weights not found${NC}"
    exit 1
fi

# Verify statue restoration module (lazy loading - no model loading test)
echo -e "${YELLOW}🧪 Verifying statue restoration setup (lazy loading)...${NC}"
cd modules/statue_restoration
python -c "
try:
    from statue_restoration import StatueRestorer
    restorer = StatueRestorer()
    print('✅ Statue restoration module ready - models will load on-demand')
    exit(0)
except Exception as e:
    print(f'❌ Statue restoration setup error: {e}')
    exit(1)
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️ Statue restoration setup failed, but continuing...${NC}"
fi
cd ../..

echo -e "${GREEN}✅ Setup complete!${NC}"

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $RESTORATION_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✅ All servers stopped${NC}"
    exit 0
}

# Function to kill processes on a specific port
kill_port() {
    local port=$1
    local pids=$(lsof -ti :$port)
    if [ ! -z "$pids" ]; then
        echo -e "${YELLOW}🔄 Killing existing processes on port $port...${NC}"
        echo $pids | xargs kill -9 2>/dev/null
        sleep 2
        echo -e "${GREEN}  ✓ Port $port is now free${NC}"
        return 0
    fi
    return 1
}

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${YELLOW}⚠️ Port $port is already in use${NC}"
        return 1
    fi
    return 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Check and free up ports
echo -e "${YELLOW}🔍 Checking and freeing up required ports...${NC}"

# Kill processes on required ports
kill_port 5001
kill_port 5002
kill_port 3000
kill_port 3001

echo -e "${GREEN}✅ All required ports are now available${NC}"

# Start Flask backend on port 5001
echo -e "${BLUE}🚀 Starting Art Transform API on port 5001...${NC}"
FLASK_RUN_PORT=5001 python app.py &
BACKEND_PID=$!
echo -e "${GREEN}  ✓ Art Transform API starting (PID: $BACKEND_PID)${NC}"

# Start Statue Restoration API on port 5002
echo -e "${BLUE}🏛️ Starting Statue Restoration API on port 5002...${NC}"
cd modules/statue_restoration
python api.py &
RESTORATION_PID=$!
echo -e "${GREEN}  ✓ Statue Restoration API starting (PID: $RESTORATION_PID)${NC}"
cd ../..

# Wait for backends to start
echo -e "${YELLOW}⏳ Waiting for backend services to start...${NC}"
sleep 5

# Check if backends are running
if kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${GREEN}  ✓ Art Transform API is running${NC}"
else
    echo -e "${RED}  ❌ Art Transform API failed to start${NC}"
fi

if kill -0 $RESTORATION_PID 2>/dev/null; then
    echo -e "${GREEN}  ✓ Statue Restoration API is running${NC}"
else
    echo -e "${RED}  ❌ Statue Restoration API failed to start${NC}"
fi

# Start React frontend on port 3000
echo -e "${BLUE}🚀 Starting React frontend on port 3000...${NC}"
cd frontend
PORT=3000 npm start &
FRONTEND_PID=$!
FRONTEND_PORT=3000
echo -e "${GREEN}  ✓ Frontend starting (PID: $FRONTEND_PID) on port $FRONTEND_PORT${NC}"
cd ..

echo -e "\n${GREEN}🎉 Ancient Vision is now running!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "${BLUE}📱 Frontend:                 http://localhost:${FRONTEND_PORT:-3000}${NC}"
echo -e "${BLUE}🔧 Art Transform API:        http://localhost:5001${NC}"
echo -e "${BLUE}🏛️ Statue Restoration API:   http://localhost:5002${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e "${YELLOW}Available Features:${NC}"
echo -e "${YELLOW}  • Traditional Art Transformation (Japanese & Indian styles)${NC}"
echo -e "${YELLOW}  • AI-Powered Statue Restoration${NC}"
echo -e "${YELLOW}  • Multi-model Support${NC}"
echo -e "${YELLOW}  • Advanced Style Controls${NC}"
echo -e "\n${YELLOW}💡 Tips:${NC}"
echo -e "${YELLOW}  • Use the tabs to switch between modules${NC}"
echo -e "${YELLOW}  • Upload images to get started${NC}"
echo -e "${YELLOW}  • Experiment with different art styles${NC}"
echo -e "\n${RED}Press Ctrl+C to stop all servers${NC}"

# Wait for background processes
wait
