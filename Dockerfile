# 1) Base image
# Start from an official Python 3.10 image. slim means a lightweight version — only what's necessary, keeps the container small.
FROM python:3.10-slim

# 2) Set working directory
# Creates a folder called /app inside the container and sets it as the working directory. Now all commands from here run inside /app inside the container
WORKDIR /app

# 3) Install dependencies
# Copy requirements first (before copying code) — Docker caches this layer
# So if only your code changes, it won't reinstall all libraries every time
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#  4) Copy project files
# Copy everything from your project folder into the container's /app folder
COPY . .

# 5) Expose port
# Tell Docker this container listens on port 8000
EXPOSE 8000

# 6) Start the server
# Command that runs when the container starts
# uvicorn main:app  →  run the 'app' object from 'main.py'
# --host 0.0.0.0   →  accept requests from outside the container
# --port 8000      →  on port 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]