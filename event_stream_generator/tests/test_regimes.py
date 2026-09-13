import random
import unittest

from event_stream_generator.temporal.history import HistoryState
from event_stream_generator.regimes import get_regime


class TemporalRegimeTests(unittest.TestCase):
    def setUp(self):
        self.history = HistoryState.empty()

    def test_all_first_phase_regimes_sample_positive_delta(self):
        cases = {
            "exponential": {"lambda": 1.2},
            "gamma": {"shape": 2.0, "scale": 0.5},
            "lognormal": {"mu": 0.1, "sigma": 0.4},
            "weibull": {"shape": 1.5, "scale": 1.2},
        }
        for name, params in cases.items():
            with self.subTest(name=name):
                regime = get_regime(name)
                self.assertTrue(regime.validate_params(params))
                values = [
                    regime.sample(params, self.history, random.Random(seed))
                    for seed in range(10)
                ]
                self.assertTrue(all(value > 0.0 for value in values))
                self.assertEqual(regime.describe(params)["name"], name)

    def test_invalid_regime_parameters_are_rejected(self):
        invalid_cases = {
            "exponential": {"lambda": 0.0},
            "gamma": {"shape": -1.0, "scale": 0.5},
            "lognormal": {"mu": 0.0, "sigma": 0.0},
            "weibull": {"shape": 1.0, "scale": -0.2},
        }
        for name, params in invalid_cases.items():
            with self.subTest(name=name):
                self.assertFalse(get_regime(name).validate_params(params))


if __name__ == "__main__":
    unittest.main()
