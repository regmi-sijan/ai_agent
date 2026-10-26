import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agent.run_logger import RunLogger
from agent.tools.physics_tools import inspect_numerical_dataset, inspect_data_folder
from agent.utils import ResilientChatSession
from gui.server import app
from fastapi.testclient import TestClient


class ProjectTests(unittest.TestCase):
    def test_sample_dataset(self):
        sample = Path(__file__).resolve().parents[1] / 'sample_physics_data.csv'
        report = json.loads(inspect_numerical_dataset(str(sample)))
        self.assertEqual(report['estimated_total_rows'], 10)
        self.assertIn('particle_label', report['columns'])
        self.assertEqual(sum(report['missing_values'].values()), 0)

    def test_parquet_inspection(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.parquet'
            pd.DataFrame({'energy': [1.0, 2.0]}).to_parquet(path)
            report = json.loads(inspect_numerical_dataset(str(path)))
            self.assertEqual(report['shape'], [2, 1])
            self.assertEqual(report['columns'], ['energy'])

    def test_folder_discovery_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / 'nested'
            nested.mkdir()
            (nested / 'events.csv').write_text('x,y\n1,2\n')
            report = json.loads(inspect_data_folder(directory))
            self.assertEqual(report['total_datasets_found'], 1)
            self.assertIn('not found', inspect_numerical_dataset(str(nested / 'missing.csv')))

    def test_run_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / 'input.csv'
            data.write_text('x\n1\n')
            logger = RunLogger(base_dir=base / 'outputs')
            logger.archive_dataset(str(data))
            logger.log_user_directive('Compare a baseline')
            result = logger.finalize_run(str(data), 'test', None, 'COMPLETED_MAX_ITERATIONS', 'plan', 'result')
            self.assertEqual((result / 'data' / 'input.csv').read_text(), data.read_text())
            self.assertEqual(json.loads((result / 'results.json').read_text())['user_directives_count'], 1)
            self.assertTrue((result / 'summary.md').exists())
            self.assertEqual(RunLogger(base_dir=base / 'outputs').run_id, 'run_002')

    def test_manual_tools_and_fallback_history(self):
        client = Mock()
        first, second = Mock(), Mock()
        first.send_message.side_effect = RuntimeError('429 RESOURCE_EXHAUSTED')
        first.get_history.return_value = ['previous conversation']
        second.send_message.return_value = 'done'
        client.chats.create.side_effect = [first, second]
        with patch('agent.utils._apply_rate_limit_throttle'):
            session = ResilientChatSession(client, 'test', models=['primary', 'fallback'])
            self.assertEqual(session.send_message('hello'), 'done')
        calls = client.chats.create.call_args_list
        self.assertTrue(calls[0].kwargs['config'].automatic_function_calling.disable)
        self.assertEqual(calls[1].kwargs['model'], 'fallback')
        self.assertEqual(calls[1].kwargs['history'], ['previous conversation'])
        first.send_message.assert_called_once_with('hello')
        second.send_message.assert_called_once_with('hello')

    def test_dashboard_without_api_calls(self):
        with TestClient(app) as client:
            response = client.get('/')
            self.assertEqual(response.status_code, 200)
            self.assertIn('html', response.text.lower())
            self.assertEqual(client.get('/api/stream/nonexistent').status_code, 404)
            self.assertFalse(client.get('/api/check_path', params={'path': '/nonexistent/example.csv'}).json()['exists'])


if __name__ == '__main__':
    unittest.main()
