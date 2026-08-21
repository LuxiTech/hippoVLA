#!/usr/bin/env bash
set -euo pipefail

env -i \
  HOME="/home/agilex" \
  USER="agilex" \
  LOGNAME="agilex" \
  SHELL="/bin/bash" \
  TERM="${TERM:-xterm-256color}" \
  DISPLAY="${DISPLAY:-}" \
  XAUTHORITY="${XAUTHORITY:-}" \
  PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  bash --noprofile --norc -c '
    set -eo pipefail
    source "/opt/ros/foxy/setup.bash"
    source "/home/agilex/cobot_magic/camera_ws_ros2/install/local_setup.bash"
    exec ros2 launch "/home/agilex/cobot_magic/camera_ws_ros2/src/realsense-ros/realsense2_camera/launch/rs_launch_lsx.py"
  '
