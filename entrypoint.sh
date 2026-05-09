#!/bin/bash

# Set up SSH authorized keys from environment variable (RunPod injects PUBLIC_KEY)
if [ -n "$PUBLIC_KEY" ]; then
    echo "Adding public key from environment..."
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

# Start SSH daemon
echo "Starting SSH daemon..."
/usr/sbin/sshd || echo "Warning: sshd failed to start"

# Start JupyterLab in background (accessible on port 8888)
echo "Starting JupyterLab on port 8888..."
cd /workspace/analysis
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token='' --ServerApp.password='' &

# Check JAX GPU detection
echo "Checking JAX GPU detection..."
python -c "import jax; devices = jax.devices(); print(f'JAX detected {len(devices)} device(s): {devices}')"

echo "Services started. Container ready."

# If no command provided, keep container alive
if [ $# -eq 0 ]; then
    exec tail -f /dev/null
else
    exec "$@"
fi
