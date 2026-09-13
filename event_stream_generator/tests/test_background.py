import random
import unittest

from event_stream_generator.simulation.background import generate_background_events


class BackgroundTests(unittest.TestCase):
    def test_background_events_are_not_in_causal_chain(self):
        events = generate_background_events(
            stream_id="stream_bg",
            start=0.0,
            end=20.0,
            density="medium",
            rates={"low": 0.05, "medium": 0.25, "high": 0.8},
            max_background_events=20,
            rng=random.Random(3),
        )

        self.assertTrue(events)
        for event in events:
            self.assertTrue(event.is_background)
            self.assertIsNone(event.parent_id)
            self.assertIsNone(event.root_id)
            self.assertEqual(event.cascade_depth, 0)
            self.assertTrue(event.abstract_type.startswith("BG"))


if __name__ == "__main__":
    unittest.main()
