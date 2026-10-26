
import unittest

from app import app, FEATURES


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_valid_prediction(self):
        sample = {
            "Age": 25,
            "Country": "India",
            "Plan": "Plus",
            "Primary_Use_Case": "Learning",
            "Device": "Android",
            "Sessions": 50,
            "Avg_Session_Minutes": 20.5,
            "Messages_Sent": 500,
            "Files_Uploaded": 10,
            "Images_Generated": 5,
            "Favorite_Model": "Auto",
            "Monthly_Spend_USD": 20,
            "Notifications_Enabled": "Yes",
            "Dark_Mode": "Yes"
        }

        response = self.client.post(
            "/predict",
            json=sample
        )

        self.assertEqual(response.status_code, 200)

        result = response.get_json()

        self.assertIn("prediction", result)
        self.assertIn(
    result["prediction"],
    [
        "Very Satisfied",
        "Satisfied",
        "Neutral",
        "Dissatisfied",
        "Very Dissatisfied"
    ]
)
        self.assertEqual(result["target"], "Satisfaction")

    def test_missing_fields(self):
        response = self.client.post(
            "/predict",
            json={
                "Age": 25,
                "Country": "India"
            }
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "missing_fields",
            response.get_json()
        )

    def test_empty_request(self):
        response = self.client.post(
            "/predict",
            json={}
        )

        self.assertEqual(response.status_code, 400)

    def test_prediction_contains_all_required_fields(self):
        self.assertEqual(len(FEATURES), 14)
        self.assertIn("Age", FEATURES)
        self.assertNotIn("Satisfaction", FEATURES)


if __name__ == "__main__":
    unittest.main()
