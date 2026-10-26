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
        # Sending sample data matching your dataset's columns
        payload = {
            "Sessions": 50,
            "Messages_Sent": 200,
            "Files_Uploaded": 5
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction", response.get_json())

if __name__ == "__main__":
    unittest.main()
