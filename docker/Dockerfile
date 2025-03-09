# Use python-3.10 slim image
FROM python:3.10-slim
# set the working directory
WORKDIR /app 
# Copy the requirements.
COPY requirements.txt .
# Install the dependencies
RUN pip install requirements.txt 
# Copy the entire project 
COPY . .
# Expose ports(FLASK, MLFlow)
EXPOSE 5000 8000 

