"""Binding generation and complete mandatory aggregation, without GPU execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / 'scripts'))
from generate_attention_bindings import generate
from accept_attention_framework import summarize, SCENARIOS


class BindingsTest(unittest.TestCase):
    def test_generate_all_scenarios_and_launcher_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'resources.json'
            path.write_text(json.dumps(dict(models={f: str(ROOT / f'configs/models/{f}.yaml')
                for f in ('pid', 'flux2', 'flux3')}, data=str(ROOT / 'configs/data/default.yaml'),
                handoff_argv=[sys.executable, str(ROOT / 'plans/attention-framework/check_handoff.py')])))
            bindings = generate(path)
            self.assertEqual(set(bindings), set(SCENARIOS))
            self.assertTrue(bindings['distributed']['argv'][0].endswith('check_distributed.py'))
            for name, item in bindings.items():
                if name != 'distributed':
                    self.assertEqual(item['argv'][0], sys.executable)
                self.assertTrue(Path(item['argv'][0 if name == 'distributed' else 1]).is_file())
            path.write_text(json.dumps(dict(models={'pid': 'missing'})))
            with self.assertRaises(ValueError):
                generate(path)

    def test_partial_verification_cannot_pass(self):
        result = summarize({'pid': {'verification': 2, 'status': 'passed'}})
        self.assertEqual(result['2'], 'blocked')
        self.assertEqual(result['4'], 'blocked')
        all_pass = {name: {'status': 'passed'} for name in SCENARIOS}
        self.assertEqual(set(summarize(all_pass).values()), {'passed'})
        del all_pass['pid_cycle']
        self.assertEqual(summarize(all_pass)['5'], 'blocked')
        self.assertEqual(summarize(all_pass)['6'], 'blocked')
