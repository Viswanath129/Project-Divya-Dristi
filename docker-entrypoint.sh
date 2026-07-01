#!/bin/sh
# Start the FastAPI server using uvicorn, reading the PORT environment variable
exec uvicorn backend.api:app --host 0.0.0.0 --port "${PORT:-8080}"
