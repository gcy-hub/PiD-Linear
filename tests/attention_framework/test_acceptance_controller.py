"""The acceptance gate must never pass missing or failed mandatory work."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('acceptance', Path(__file__).resolve().parents[2] / 'scripts/accept_attention_framework.py')
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


class ControllerTest(unittest.TestCase):
    def test_incomplete_and_failed_evidence(self):
        self.assertEqual(set(acceptance.summarize({}).values()), {'blocked'})
        result = {'common': {'verification': 3, 'status': 'failed'}}
        self.assertEqual(acceptance.summarize(result)['2'], 'failed')
        self.assertEqual(acceptance.summarize(result)['3'], 'failed')

    def test_missing_and_planned_commands_do_not_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(acceptance.run_case('pid', None, Path(directory), execute=True, gpu='0')['status'], 'blocked')
            binding = {'argv': ['/nonexistent'], 'config': {'batch': 1}, 'resources': {'weights': 'local'}}
            result = acceptance.run_case('pid', binding, Path(directory), execute=False, gpu='0')
            self.assertEqual(result['status'], 'blocked')
            self.assertFalse(list(Path(directory).iterdir()))

    def test_exit_failure_and_missing_executable(self):
        import sys
        with tempfile.TemporaryDirectory() as directory:
            binding = {'argv': [sys.executable, '-c', 'raise SystemExit(7)'], 'config': {}, 'resources': {}}
            result = acceptance.run_case('config_data', binding, Path(directory), execute=True, gpu='')
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['returncode'], 7)
            binding['argv'] = ['/nonexistent']
            self.assertEqual(acceptance.run_case('config_data', binding, Path(directory), execute=True, gpu='')['status'], 'blocked')

    def test_planning_reports_every_verification_and_blocks_publish(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(acceptance.main(['--reports', directory]), 1)
            report = json.loads((Path(directory) / 'acceptance.json').read_text())
            self.assertEqual(set(report['verification']), set('12345678'))
            self.assertEqual(set(report['results']), set(acceptance.SCENARIOS))
            self.assertFalse(report['publish_allowed'])
            self.assertTrue(all(v == 'blocked' for v in report['verification'].values()))
            self.assertNotIn('gpu_inventory', report)

    def test_binding_boundary(self):
        for bad in ({'typo': {}}, {'pid': {'argv': 'shell command'}}, {'pid': {'argv': [], 'config': {}, 'resources': {}}}):
            with self.assertRaises(ValueError):
                acceptance.validate_bindings(bad)


if __name__ == '__main__':
    unittest.main()
