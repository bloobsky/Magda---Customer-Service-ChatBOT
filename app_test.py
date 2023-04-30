from app import app

import unittest
import pytest


class TestApp(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_login_page(self):
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Login", response.data)

    def test_login_failure(self):
        response = self.client.post('/login', data=dict(
            username='invalid',
            password='invalid'
        ), follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Invalid username or password.", response.data)

    def test_invalid_route(self):
        response = self.client.get('/invalid_route')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b"Not Found", response.data)


if __name__ == '__main__':
    # Unit Testing (Works only when Flask is running)
    unittest.main()

    # Pytest can be run offlinec
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
