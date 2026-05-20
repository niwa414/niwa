import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from miet_claw.mcp_server import MietClawMCPServer


ROOT = Path(__file__).resolve().parents[1]


def _write_message(proc: subprocess.Popen, payload: dict) -> None:
    raw = json.dumps(payload).encode('utf-8')
    proc.stdin.write(f'Content-Length: {len(raw)}\r\n\r\n'.encode('utf-8'))
    proc.stdin.write(raw)
    proc.stdin.flush()


def _read_message(proc: subprocess.Popen) -> dict:
    headers = {}
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError('MCP server closed stdout unexpectedly')
        if line in (b'\r\n', b'\n'):
            break
        name, value = line.decode('utf-8').split(':', 1)
        headers[name.strip().lower()] = value.strip()
    size = int(headers['content-length'])
    payload = proc.stdout.read(size)
    return json.loads(payload.decode('utf-8'))


def _write_jsonl_message(proc: subprocess.Popen, payload: dict) -> None:
    proc.stdin.write(json.dumps(payload).encode('utf-8') + b'\n')
    proc.stdin.flush()


def _read_jsonl_message(proc: subprocess.Popen) -> dict:
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError('MCP server closed stdout unexpectedly')
    return json.loads(line.decode('utf-8'))


class MietClawMCPServerTests(unittest.TestCase):
    def test_mcp_server_lists_tools_and_can_call_them(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = dict(os.environ)
            env['PYTHONPATH'] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
            proc = subprocess.Popen(
                [
                    'python3',
                    '-m',
                    'miet_claw.cli',
                    'mcp-server',
                    '--project-root',
                    str(ROOT),
                    '--workspace-root',
                    str(Path(tmpdir) / 'autonomy'),
                    '--output-dir',
                    str(ROOT / '.runs'),
                    '--provider',
                    'local',
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            try:
                _write_message(
                    proc,
                    {
                        'jsonrpc': '2.0',
                        'id': 1,
                        'method': 'initialize',
                        'params': {'protocolVersion': '2025-03-26', 'clientInfo': {'name': 'test', 'version': '0.0'}},
                    },
                )
                initialize = _read_message(proc)
                self.assertEqual(initialize['result']['serverInfo']['name'], 'mietclaw-mcp')

                _write_message(proc, {'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {}})

                _write_message(proc, {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}})
                tools_list = _read_message(proc)
                tool_names = [item['name'] for item in tools_list['result']['tools']]
                self.assertIn('miet_kmc_bridge', tool_names)
                self.assertIn('miet_autonomy_draft', tool_names)
                self.assertIn('miet_runtime_doctor', tool_names)
                self.assertIn('miet_moire_run', tool_names)
                self.assertIn('miet_moire_compare', tool_names)
                self.assertIn('miet_moire_diffusion_sweep', tool_names)
                self.assertIn('miet_moire_lammps', tool_names)
                self.assertIn('miet_moire_kmc', tool_names)

                _write_message(
                    proc,
                    {
                        'jsonrpc': '2.0',
                        'id': 3,
                        'method': 'tools/call',
                        'params': {'name': 'miet_list_runs', 'arguments': {'output_dir': str(ROOT / '.runs'), 'limit': 2}},
                    },
                )
                runs_result = _read_message(proc)
                self.assertFalse(runs_result['result']['isError'])
                self.assertIn('runs', runs_result['result']['structuredContent'])

                _write_message(
                    proc,
                    {
                        'jsonrpc': '2.0',
                        'id': 4,
                        'method': 'tools/call',
                        'params': {
                            'name': 'miet_autonomy_draft',
                            'arguments': {
                                'prompt': 'Create a KMC only diffusion job for MCP server test at 830 K with Fe=0.62 eV, Cu=0.54 eV, Ni=0.53 eV.',
                                'provider': 'local',
                                'workspace_root': str(Path(tmpdir) / 'autonomy'),
                            },
                        },
                    },
                )
                draft_result = _read_message(proc)
                self.assertFalse(draft_result['result']['isError'])
                self.assertEqual(draft_result['result']['structuredContent']['mode'], 'kmc_only')
            finally:
                proc.terminate()
                proc.wait(timeout=10)
                if proc.stdin:
                    proc.stdin.close()
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()

    def test_mcp_server_supports_jsonl_stdio_transport(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = dict(os.environ)
            env['PYTHONPATH'] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
            proc = subprocess.Popen(
                [
                    'python3',
                    '-m',
                    'miet_claw.cli',
                    'mcp-server',
                    '--project-root',
                    str(ROOT),
                    '--workspace-root',
                    str(Path(tmpdir) / 'autonomy'),
                    '--output-dir',
                    str(ROOT / '.runs'),
                    '--provider',
                    'local',
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            try:
                _write_jsonl_message(
                    proc,
                    {
                        'jsonrpc': '2.0',
                        'id': 1,
                        'method': 'initialize',
                        'params': {'protocolVersion': '2025-06-18', 'clientInfo': {'name': 'codex', 'version': '0.128.0'}},
                    },
                )
                initialize = _read_jsonl_message(proc)
                self.assertEqual(initialize['result']['serverInfo']['name'], 'mietclaw-mcp')
                self.assertEqual(initialize['result']['protocolVersion'], '2025-06-18')

                _write_jsonl_message(proc, {'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {}})

                _write_jsonl_message(proc, {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}})
                tools_list = _read_jsonl_message(proc)
                tool_names = [item['name'] for item in tools_list['result']['tools']]
                self.assertIn('miet_runtime_doctor', tool_names)
                self.assertIn('miet_moire_kmc', tool_names)
            finally:
                proc.terminate()
                proc.wait(timeout=10)
                if proc.stdin:
                    proc.stdin.close()
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()

    def test_mcp_server_can_dispatch_runtime_doctor_and_moire_run(self):
        server = MietClawMCPServer(
            project_root=str(ROOT),
            workspace_root=str(ROOT / '.autonomy-mcp'),
            output_dir=str(ROOT / '.runs'),
            provider='local',
        )

        with mock.patch('miet_claw.mcp_server.get_local_model_status', return_value={'healthy': True, 'default_model': 'demo-model'}):
            with mock.patch('miet_claw.mcp_server.collect_runtime_doctor', return_value={'checks': {'kmc_binary_exists': True}}):
                result = server._dispatch_tool('miet_runtime_doctor', {})
        self.assertFalse(result['isError'])
        self.assertIn('checks', result['structuredContent'])

        fake_summary = {
            'event_json': '/tmp/event.json',
            'source_case_dir': '/tmp/case',
            'copied_case_dir': '/tmp/work/case',
            'generated_lammps_input': '/tmp/work/case/generated_in.neb.mietclaw',
            'generated_barrier_script': '/tmp/work/case/extract_barrier.mietclaw.sh',
            'neb_txt': '/tmp/work/case/neb.txt',
            'lammps': {'status': 'executed', 'log': '/tmp/work/lammps_run.out'},
            'postprocess': {'log': '/tmp/work/neb_postprocess.out'},
            'summary_json': '/tmp/work/summary.json',
            'kmc': {
                'barrier_eV': 0.51,
                'files': {'state_values_sites': '/tmp/work/kmc_bridge/state.repo.values.sites', 'input_kmc': '/tmp/work/kmc_bridge/generated_kmc.repo.in', 'run_out': '/tmp/work/kmc_bridge/run.out'},
                'parsed_run': {'accepted_events': 9, 'final_time': 1e-10},
            },
        }
        with mock.patch('miet_claw.mcp_server.run_moire_lammps_to_kmc', return_value=fake_summary):
            result = server._dispatch_tool(
                'miet_moire_run',
                {'event_json': '/tmp/event.json', 'case_dir': '/tmp/case', 'workdir': '/tmp/work', 'validate': True},
            )
        self.assertFalse(result['isError'])
        self.assertEqual(result['structuredContent']['kmc']['barrier_eV'], 0.51)

        with mock.patch('miet_claw.mcp_server.run_moire_lammps_to_kmc', return_value=fake_summary) as mocked_moire_run:
            result = server._dispatch_tool(
                'miet_moire_run',
                {
                    'event_json': '/tmp/event.json',
                    'case_dir': '/tmp/case',
                    'workdir': '/tmp/work',
                    'validate': True,
                    'kmc_seeds': [4101, 4102],
                    'ovito': True,
                },
            )
        self.assertFalse(result['isError'])
        self.assertEqual(mocked_moire_run.call_args.kwargs['kmc_seeds'], [4101, 4102])
        self.assertTrue(mocked_moire_run.call_args.kwargs['render_ovito'])

        fake_compare_summary = {
            'status': 'completed',
            'mode': 'moire_event_compare',
            'case_dir': '/tmp/case',
            'workdir': '/tmp/work',
            'event_count': 2,
            'completed_count': 2,
            'event_runs': [
                {'label': 'event_a', 'status': 'completed', 'barrier_eV': 0.31, 'summary_json': '/tmp/work/event_a/summary.json'},
                {'label': 'event_b', 'status': 'completed', 'barrier_eV': 0.58, 'summary_json': '/tmp/work/event_b/summary.json'},
            ],
            'barrier_ranking': [
                {'rank': 1, 'label': 'event_a', 'barrier_eV': 0.31},
                {'rank': 2, 'label': 'event_b', 'barrier_eV': 0.58},
            ],
            'summary_json': '/tmp/work/summary.json',
            'comparison_json': '/tmp/work/comparison.json',
        }
        with mock.patch('miet_claw.mcp_server.run_moire_event_compare', return_value=fake_compare_summary) as mocked_compare:
            result = server._dispatch_tool(
                'miet_moire_compare',
                {
                    'case_dir': '/tmp/case',
                    'event_jsons': ['/tmp/a.json', '/tmp/b.json'],
                    'workdir': '/tmp/work',
                    'kmc_seeds': [3401, 3402],
                    'ovito': True,
                },
            )
        self.assertFalse(result['isError'])
        self.assertEqual(mocked_compare.call_args.kwargs['event_jsons'], ['/tmp/a.json', '/tmp/b.json'])
        self.assertEqual(mocked_compare.call_args.kwargs['kmc_seeds'], [3401, 3402])
        self.assertTrue(mocked_compare.call_args.kwargs['render_ovito'])

        fake_diffusion_summary = {
            'status': 'completed',
            'mode': 'moire_diffusion_sweep',
            'case_dir': '/tmp/case',
            'event_json': '/tmp/event.json',
            'workdir': '/tmp/work',
            'barrier_eV': 0.454657,
            'temperatures_k': [800.0, 900.0],
            'kmc_seeds': [3401, 3402],
            'temperature_runs': [
                {'label': '800 K', 'status': 'completed', 'diffusion_coefficient': 8.0e-10},
                {'label': '900 K', 'status': 'completed', 'diffusion_coefficient': 9.0e-10},
            ],
            'files': {'diffusion_vs_temperature_svg': '/tmp/work/diffusion_vs_temperature.svg'},
            'summary_json': '/tmp/work/summary.json',
        }
        with mock.patch('miet_claw.mcp_server.run_moire_diffusion_sweep', return_value=fake_diffusion_summary) as mocked_diffusion:
            result = server._dispatch_tool(
                'miet_moire_diffusion_sweep',
                {
                    'event_json': '/tmp/event.json',
                    'case_dir': '/tmp/case',
                    'workdir': '/tmp/work',
                    'temperatures_k': [800, 900],
                    'kmc_seeds': [3401, 3402],
                    'run_time': '1e-6',
                    'stats_step': '1e-7',
                    'ovito': True,
                },
            )
        self.assertFalse(result['isError'])
        self.assertEqual(mocked_diffusion.call_args.kwargs['temperatures_k'], [800, 900])
        self.assertEqual(mocked_diffusion.call_args.kwargs['kmc_seeds'], [3401, 3402])
        self.assertEqual(mocked_diffusion.call_args.kwargs['run_time'], '1e-6')
        self.assertEqual(mocked_diffusion.call_args.kwargs['stats_step'], '1e-7')
        self.assertTrue(mocked_diffusion.call_args.kwargs['render_ovito'])

        fake_lammps_summary = {
            'status': 'completed',
            'source_case_dir': '/tmp/case',
            'copied_case_dir': '/tmp/work/lammps_case',
            'neb_txt': '/tmp/work/lammps_case/neb.txt',
            'barrier_eV': 0.51,
            'lammps': {'status': 'executed', 'log': '/tmp/work/lammps_run.out'},
            'postprocess': {'log': '/tmp/work/neb_postprocess.out'},
            'summary_json': '/tmp/work/lammps_summary.json',
        }
        with mock.patch('miet_claw.mcp_server.run_moire_lammps_case', return_value=fake_lammps_summary):
            result = server._dispatch_tool(
                'miet_moire_lammps',
                {'case_dir': '/tmp/case', 'workdir': '/tmp/work'},
            )
        self.assertFalse(result['isError'])
        self.assertEqual(result['structuredContent']['barrier_eV'], 0.51)

        with mock.patch('miet_claw.mcp_server.run_moire_lammps_case', return_value=fake_lammps_summary) as mocked_lammps:
            result = server._dispatch_tool(
                'miet_moire_lammps',
                {'case_dir': '/tmp/case', 'workdir': '/tmp/work', 'ovito': True},
            )
        self.assertFalse(result['isError'])
        self.assertTrue(mocked_lammps.call_args.kwargs['render_ovito'])

        fake_kmc_summary = {
            'status': 'completed',
            'barrier_eV': 0.51,
            'files': {'state_values_sites': '/tmp/work/kmc_bridge/state.repo.values.sites', 'input_kmc': '/tmp/work/kmc_bridge/generated_kmc.repo.in', 'run_out': '/tmp/work/kmc_bridge/run.out'},
            'parsed_run': {'accepted_events': 9, 'final_time': 1e-10},
            'state_transform': {'converted_pair_markers': 24, 'pair_marker_host_type': 1},
            'barrier_assignment': {'Mo': 0.51, 'Re': 0.51},
            'runtime_health': {'status': 'ok', 'warnings': [], 'checks': {'returncode_ok': True}},
        }
        with mock.patch('miet_claw.mcp_server.run_moire_repo_kmc', return_value=fake_kmc_summary):
            result = server._dispatch_tool(
                'miet_moire_kmc',
                {'event_json': '/tmp/event.json', 'barrier_eV': 0.51, 'workdir': '/tmp/work/kmc_bridge'},
            )
        self.assertFalse(result['isError'])
        self.assertEqual(result['structuredContent']['parsed_run']['accepted_events'], 9)

        with mock.patch('miet_claw.mcp_server.run_moire_repo_kmc', return_value=fake_kmc_summary) as mocked_moire_kmc:
            result = server._dispatch_tool(
                'miet_moire_kmc',
                {
                    'event_json': '/tmp/event.json',
                    'barrier_eV': 0.51,
                    'workdir': '/tmp/work/kmc_bridge',
                    'kmc_seed': 4101,
                    'ovito': True,
                },
            )
        self.assertFalse(result['isError'])
        self.assertEqual(mocked_moire_kmc.call_args.kwargs['kmc_seed'], 4101)
        self.assertTrue(mocked_moire_kmc.call_args.kwargs['render_ovito'])


if __name__ == '__main__':
    unittest.main()
