import pytest 
import json 
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.api.main import app

@pytest.fixture 
def client():
    """Create a test client for Flask App"""
    app.testing = True
    return app.test_client() 

def test_home_route(client):
    """Test if the home page returns a successfull response"""
    response = client.get("/")
    assert response.status_code == 200 
    assert "CLIP model API is running! upload a image in '/predict' page to predictions" in response.data.decode("utf-8")

def test_predict_no_valid_image(client):
    """Test the API with invalid input"""
    response = client.post(
        "/predict", data = {}
    )

    assert response.status_code == 400 
    json_data = json.loads(response.data) 
    assert "Error" in json_data 