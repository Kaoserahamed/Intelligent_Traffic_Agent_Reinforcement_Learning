FROM python:3.11-slim

# Install SUMO
# Note: SUMO isn't available in Debian slim repos; install via pip or pre-built binaries
# For local development, set SUMO_HOME env var or mount SUMO installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package
COPY traffic-rl-agents/pyproject.toml traffic-rl-agents/README.md* ./
COPY traffic-rl-agents/src/traffic_rl/ ./src/traffic_rl/
COPY traffic-rl-agents/sim/ ./sim/
COPY traffic-rl-agents/configs/ ./configs/

RUN pip install --no-cache-dir -e ".[dev]"

# SUMO binaries can be mounted or installed at runtime
ENV SUMO_HOME=/opt/sumo
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "traffic_rl.cli"]