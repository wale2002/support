import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_classify_customer():
    response = client.post("/classify", json={"complaint": "My bill is $50 too high this month."})
    assert response.status_code == 200
    data = response.json()
    assert data["classification"] == "customer"
    assert "solution" in data

def test_classify_fibre():
    response = client.post("/classify", json={"complaint": "Fibre connection is down due to outage."})
    assert response.status_code == 200
    data = response.json()
    assert data["classification"] == "fibre"

def test_empty_complaint():
    response = client.post("/classify", json={"complaint": ""})
    assert response.status_code == 400