"""Run the shared config/CPU numerical acceptance against the contract."""
import runpy
from pathlib import Path
import unittest


class ContractTest(unittest.TestCase):
    def test_contract_reference(self):
        root = Path(__file__).resolve().parents[2]
        runpy.run_path(str(root / 'plans/attention-framework/checks/contract_cpu.py'), run_name='__main__')


if __name__ == '__main__':
    unittest.main()
