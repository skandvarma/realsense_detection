#!/bin/bash
# Launch script for ROS2-based unified system (Camera + Detection + RTAB-Map)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== ROS2 Unified System Launcher ===${NC}"
echo ""

# Function to kill all processes on exit
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    # Kill processes by PID if they exist
    [[ -n $CAMERA_PID ]] && kill $CAMERA_PID 2>/dev/null
    [[ -n $DETECTION_PID ]] && kill $DETECTION_PID 2>/dev/null
    [[ -n $RTABMAP_PID ]] && kill $RTABMAP_PID 2>/dev/null
    exit
}
trap cleanup EXIT

# Step 1: Launch RealSense Camera Node
echo -e "${GREEN}[1/3] Starting ROS2 RealSense Camera Node...${NC}"
ros2 launch realsense2_camera rs_launch.py \
    align_depth.enable:=true \
    pointcloud.enable:=true &
CAMERA_PID=$!

echo "Waiting for camera to initialize..."
sleep 5

# Check if camera topics are available
echo -e "${YELLOW}Checking camera topics...${NC}"
if ! ros2 topic list | grep -q "/camera/camera/color/image_raw"; then
    echo -e "${RED}ERROR: Camera topics not found!${NC}"
    echo "Please check the RealSense camera connection and try again."
    exit 1
fi
echo -e "${GREEN}✓ Camera topics detected${NC}"

# Show available topics
echo ""
echo "Available camera topics:"
ros2 topic list | grep camera | head -10
echo ""

# Step 2: Start Detection System
echo -e "${GREEN}[2/3] Starting Detection System...${NC}"
echo "Detection system will use the ROS2 camera topics"
python3 run_ros2_unified_system.py --config config.yaml &
DETECTION_PID=$!

sleep 3

# Step 3: Ask user if they want to start RTAB-Map
echo ""
echo -e "${YELLOW}Do you want to start RTAB-Map SLAM? (y/n)${NC}"
read -r response

if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo -e "${GREEN}[3/3] Starting RTAB-Map...${NC}"
    
    # Ask for visualization preference
    echo -e "${YELLOW}Use rtabmap_viz (1) or rviz2 (2)? [default: 1]${NC}"
    read -r viz_choice
    viz_choice=${viz_choice:-1}
    
    if [[ "$viz_choice" == "2" ]]; then
        # Launch with RViz2
        ros2 launch rtabmap_launch rtabmap.launch.py \
             rtabmap_args:="--delete_db_on_start" \
             depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
             rgb_topic:=/camera/camera/color/image_raw \
             camera_info_topic:=/camera/camera/color/camera_info \
             approx_sync:=true \
             frame_id:=camera_link \
             use_odom:=false &
    else
        # Launch with rtabmap_viz (default)
        ros2 launch rtabmap_launch rtabmap.launch.py \
             rtabmap_args:="--delete_db_on_start" \
             depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
             rgb_topic:=/camera/camera/color/image_raw \
             camera_info_topic:=/camera/camera/color/camera_info \
             approx_sync:=true \
             frame_id:=camera_link \
             use_odom:=false &
    fi
    RTABMAP_PID=$!
    echo -e "${GREEN}✓ RTAB-Map started${NC}"
else
    echo -e "${YELLOW}Skipping RTAB-Map. You can start it manually later with:${NC}"
    echo "ros2 launch rtabmap_launch rtabmap.launch.py \\"
    echo "  rgb_topic:=/camera/camera/color/image_raw \\"
    echo "  depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \\"
    echo "  camera_info_topic:=/camera/camera/color/camera_info"
fi

# Show status
echo ""
echo -e "${GREEN}=== All systems launched ===${NC}"
echo "Camera PID: $CAMERA_PID"
echo "Detection PID: $DETECTION_PID"
[[ -n $RTABMAP_PID ]] && echo "RTAB-Map PID: $RTABMAP_PID"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all processes${NC}"
echo ""

# Monitor processes
echo -e "${GREEN}System Status:${NC}"
while true; do
    # Check if processes are still running
    if ! kill -0 $CAMERA_PID 2>/dev/null; then
        echo -e "${RED}Camera node stopped!${NC}"
        break
    fi
    
    if ! kill -0 $DETECTION_PID 2>/dev/null; then
        echo -e "${RED}Detection system stopped!${NC}"
        break
    fi
    
    if [[ -n $RTABMAP_PID ]] && ! kill -0 $RTABMAP_PID 2>/dev/null; then
        echo -e "${YELLOW}RTAB-Map stopped${NC}"
        RTABMAP_PID=""
    fi
    
    sleep 5
done

# If we get here, something stopped unexpectedly
echo -e "${RED}One or more systems stopped unexpectedly!${NC}"
cleanup
