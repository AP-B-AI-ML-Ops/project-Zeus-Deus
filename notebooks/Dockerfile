FROM python:3.13-slim

# Set work directory
WORKDIR /workspace

# Install dependencies
COPY ../requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir jupyter
