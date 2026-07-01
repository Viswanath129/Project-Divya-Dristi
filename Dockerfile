# ==========================================
# Stage 1: Build the React frontend
# ==========================================
FROM node:20-alpine AS frontend-builder
WORKDIR /build

# Copy frontend dependency files
COPY app/package*.json ./
RUN npm ci

# Copy frontend source files
COPY app/ ./

# Build the frontend (outputs to /build/dist)
RUN npm run build

# ==========================================
# Stage 2: Build the FastAPI backend with PyTorch CPU
# ==========================================
FROM python:3.10-slim

# Install system dependencies needed for OpenCV, netCDF4, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements first for caching
COPY satellite-interpolation/requirements.txt ./

# Install CPU-optimized PyTorch and torchvision first to minimize image size,
# then install remaining backend dependencies.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY satellite-interpolation/ ./

# Copy built frontend assets to the expected location so FastAPI can serve them
COPY --from=frontend-builder /build/dist /workspace/../app/dist

# Expose port (Google Cloud Run uses PORT env variable)
ENV PORT=8080
EXPOSE 8080

# Copy entrypoint script
COPY docker-entrypoint.sh /workspace/docker-entrypoint.sh
RUN chmod +x /workspace/docker-entrypoint.sh

# Run using the entrypoint script
ENTRYPOINT ["/workspace/docker-entrypoint.sh"]
