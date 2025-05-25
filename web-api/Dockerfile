FROM python:3.11-slim

RUN pip install -U pip setuptools wheel

# Copy requirements first (Docker layer caching)
COPY train-deploy/requirements.txt .
RUN pip install -r requirements.txt

# Install Flask
RUN pip install flask

# Set workdir and copy app
WORKDIR /app
COPY web-api/app.py .

CMD ["python", "app.py"]