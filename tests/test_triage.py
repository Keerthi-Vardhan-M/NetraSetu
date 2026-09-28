import unittest

from src.triage import determine_triage


class TriageTests(unittest.TestCase):
    def test_grade_four_is_urgent(self):
        self.assertEqual(determine_triage(1, 4, [])["priority"], "URGENT")

    def test_emergency_symptom_overrides_low_grade(self):
        self.assertEqual(determine_triage(0, 0, ["Sudden vision loss"])["priority"], "URGENT")

    def test_ungradable_routes_to_review(self):
        self.assertEqual(determine_triage(None, 0, [], ungradable=True)["priority"], "MANUAL REVIEW")

    def test_grade_two_is_referral(self):
        self.assertEqual(determine_triage(2, 1, [])["priority"], "REFERRAL")


if __name__ == "__main__":
    unittest.main()

