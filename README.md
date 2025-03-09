# Overview

The **CLIP-MLOps-Pipeline** is a Machine Learning Operations (MLOps) pipeline designed to train, evaluate, and deploy CLIP-based models for image-text processing. It includes modular components for data processing, model training, **experiment tracking with MLflow**, **API deployment with FastAPI**, and **CI/CD automation using Jenkins and Docker**.

# Features

**Modular Design:** Organized structure for models, data, training, and testing.

**MLflow Tracking:** Tracks model metrics and versioning.

**Dockerized Deployment:** API and training environments are containerized.

**CI/CD Integration:** Automated pipeline with Jenkins for testing and deployment.

**FastAPI for Model Serving:** Provides an efficient REST API for inference.

# Project Structure
```
📦 CLIP-MLOps-Pipeline
├── 📂 src/                     
│   ├── 📂 models/              
│   │   ├── image_encoder.py
│   │   ├── text_encoder.py
│   │   ├── transformer_encoder.py
│   │   ├── attn_head.py
│   │   ├── pos_embeds.py
│   │   ├── model.py
│   ├── 📂 data/                
│   │   ├── dataset.py
│   │   ├── parameters.json
│   ├── 📂 training/            
│   │   ├── train.py
│   │   ├── test.py
│   ├── 📂 api/                 
│   │   ├── main.py             
│   │   ├── Dockerfile          
│   │   ├── requirements.txt
│   ├── 📂 tests/               
│   │   ├── test_model.py
│   │   ├── test_api.py
├── 📂 mlflow/                  
│   ├── tracking.py           
├── 📂 docker/
│   ├── Dockerfile
├── 📂 ci-cd/
│   ├── Jenkinsfile
├── README.md
├── requirements.txt 
├── .gitignore
```