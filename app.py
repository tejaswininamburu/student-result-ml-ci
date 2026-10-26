import unittest
from app import app

class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_prediction_success(self):
        response = self.client.post("/predict", json={
            "age": 30,
            "usage_hours": 40,
            "subscription_type": 1
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.get_json())

    def test_missing_field_validation(self):
        response = self.client.post("/predict", json={"age": 30})
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())

if __name__ == "__main__":
    unittest.main()
