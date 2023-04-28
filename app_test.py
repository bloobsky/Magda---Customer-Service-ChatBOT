import pytest
from flask import session, url_for

from app import app

@pytest.fixture(scope='module')
def test_client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_login_page(test_client):
    response = test_client.get('/login')
    assert response.status_code == 200
    assert b"Login" in response.data


def test_login_failure(test_client):
    response = test_client.post('/login', data=dict(
        username='invalid',
        password='invalid'
    ), follow_redirects=True)
    assert response.status_code == 200
    assert b"Invalid username or password." in response.data


def test_invalid_route(test_client):
    response = test_client.get('/invalid_route')
    assert response.status_code == 404
    assert b"Not Found" in response.data
