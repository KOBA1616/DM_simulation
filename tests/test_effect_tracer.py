import json
import unittest
import time
from dm_toolkit.effect_tracer import EffectTracer, TraceEventType

class TestEffectTracer(unittest.TestCase):
    def test_basic_tracing(self):
        tracer = EffectTracer()
        tracer.start_effect("FireBall", {"target": "CreatureA"})
        tracer.step("Applying damage", {"amount": 5000})
        tracer.end_effect("FireBall")

        history = tracer.get_history()
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]['event_type'], 'START_EFFECT')
        self.assertEqual(history[0]['description'], 'Start effect: FireBall')
        self.assertEqual(history[1]['event_type'], 'STEP')
        self.assertEqual(history[1]['description'], 'Applying damage')
        self.assertEqual(history[2]['event_type'], 'END_EFFECT')

    def test_json_output(self):
        tracer = EffectTracer()
        tracer.info("Test Info")
        json_output = tracer.to_json()
        data = json.loads(json_output)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['description'], 'Test Info')

    def test_flowchart_export(self):
        tracer = EffectTracer()
        tracer.start_effect("Root")
        tracer.step("Step 1")
        tracer.start_effect("Nested")
        tracer.end_effect("Nested")
        tracer.end_effect("Root")

        flow_data = tracer.export_for_flowchart()
        steps = flow_data['steps']
        self.assertEqual(len(steps), 5)
        # Check depth
        self.assertEqual(steps[0]['depth'], 0) # Start Root
        self.assertEqual(steps[1]['depth'], 1) # Step 1
        self.assertEqual(steps[2]['depth'], 1) # Start Nested
        self.assertEqual(steps[3]['depth'], 1) # End Nested (depth is at end of block, or should it be?)
        # Let's check implementation logic:
        # Start: depth (before increment) -> 0
        # Step: depth -> 1
        # Start Nested: depth -> 1
        # End Nested: depth (after decrement) -> 1. Wait.

        # Implementation:
        # START: item['depth'] = depth; depth += 1  => depth=0, then depth=1
        # STEP: item['depth'] = depth => depth=1
        # START: item['depth'] = depth; depth += 1 => depth=1, then depth=2
        # END: depth -= 1; item['depth'] = depth => depth=1, then item['depth']=1
        # END: depth -= 1; item['depth'] = depth => depth=0, then item['depth']=0

        self.assertEqual(steps[0]['depth'], 0)
        self.assertEqual(steps[1]['depth'], 1)
        self.assertEqual(steps[2]['depth'], 1)
        self.assertEqual(steps[3]['depth'], 1)
        self.assertEqual(steps[4]['depth'], 0)

    def test_disable(self):
        tracer = EffectTracer()
        tracer.disable()
        tracer.info("Should not appear")
        self.assertEqual(len(tracer.get_history()), 0)
        tracer.enable()
        tracer.info("Should appear")
        self.assertEqual(len(tracer.get_history()), 1)

if __name__ == '__main__':
    unittest.main()
