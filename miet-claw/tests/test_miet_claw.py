import csv
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from miet_claw.autonomy import extract_material_name, materialize_autonomy_workspace, run_autonomy_job
from miet_claw.bridge import run_kmc_lookup_bridge
from miet_claw.chat import (
    MietClawChatSession,
    ToolExecutionOutcome,
    _resolve_model_alias,
    compare_recent_runs,
    get_log_excerpt,
    inspect_run,
    list_runs,
    list_artifacts,
    run_chat_once_payload,
)
from miet_claw.executor import ExecutionCancelled, run_job
from miet_claw.frontends.shell_commands import SHELL_COMMAND_HANDLERS, missing_shell_command_handlers
from miet_claw.neb_runtime import parse_neb_terse_output
from miet_claw.local_profile import get_local_model_settings, get_runtime_settings, load_local_agent_profile
from miet_claw.moire_runtime import (
    MoReWorkflowError,
    run_moire_diffusion_sweep,
    run_moire_event_compare,
    run_moire_lammps_case,
    run_moire_lammps_to_kmc,
    run_moire_repo_kmc,
)
from miet_claw.planner import build_plan_payload
from miet_claw.runtime.approval import ToolApprovalContext, decide_tool_approval
from miet_claw.runtime.snapshot import build_chat_once_payload as build_runtime_chat_payload
from miet_claw.runtime.context import build_engine_context as build_runtime_engine_context
from miet_claw.runtime.snapshot import build_snapshot_from_turn_result
from miet_claw.runtime.permissions import permission_profile_for_intent, permission_scope_for_intent
from miet_claw.runtime.query_engine import MietQueryEngine, QueryTurnResult
from miet_claw.runtime.query_loop import run_agent_query_loop, run_tool_event_loop
from miet_claw.runtime.router_eval import run_router_golden_eval
from miet_claw.runtime.runtime_eval import run_runtime_health_golden_eval
from miet_claw.runtime.shell_command_registry import canonical_shell_command, shell_command_names, shell_command_summaries
from miet_claw.runtime.tool_dispatch import dispatch_mcp_tool, execute_chat_tool_intent_outcome
from miet_claw.runtime.types import (
    AssistantActionBlock,
    AssistantActionBlockEvent,
    FinalAnswerBlock,
    PermissionDecisionEvent,
    ToolBudget,
    ToolRequestBlock,
    ToolResultBlock,
    ToolResultBlockEvent,
    ToolTurnState,
    TurnFinishEvent,
)
from miet_claw.runtime.tool_registry import get_chat_tool_definition, get_tool_definition
from miet_claw.shell_runtime import SHELL_COMMANDS, build_shell_status, collect_runtime_doctor
from miet_claw.specs import load_job_spec
from miet_claw.tool_router import (
    ToolPlan,
    ToolIntent,
    heuristic_tool_intent,
    heuristic_tool_plan,
    parse_agent_decision,
    parse_tool_intent,
    parse_tool_plan,
    should_skip_tool_router,
    should_try_tool_plan,
)
from miet_claw.transforms import compile_event_table, compute_rate_hz, derive_kmc_barrier_map, render_kmc_input


ROOT = Path(__file__).resolve().parents[1]


class MietClawTests(unittest.TestCase):
    @staticmethod
    def _write_minimal_moire_case(case_dir: Path) -> None:
        (case_dir / 'data.lmp').write_text(
            '\n'.join(
                [
                    'LAMMPS data file',
                    '',
                    '2 atoms',
                    '2 atom types',
                    '',
                    '0.0 1.0 xlo xhi',
                    '0.0 1.0 ylo yhi',
                    '0.0 1.0 zlo zhi',
                    '',
                    'Atoms',
                    '',
                    '1 1 0.0 0.0 0.0',
                    '2 2 0.5 0.5 0.5',
                ]
            )
            + '\n',
            encoding='utf-8',
        )
        (case_dir / 'final.mosia').write_text(
            '\n'.join(
                [
                    '2',
                    '1 0.1 0.0 0.0',
                    '2 0.6 0.5 0.5',
                ]
            )
            + '\n',
            encoding='utf-8',
        )
        (case_dir / 'MoRe.eam.fs').write_text('eam', encoding='utf-8')

    def test_rate_is_positive(self):
        rate = compute_rate_hz(0.65, 1.0e13, 773.0)
        self.assertGreater(rate, 0)
        self.assertLess(rate, 1.0e13)

    def test_chain_plan_contains_expected_steps(self):
        spec = load_job_spec(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'))
        step_ids = [step['id'] for step in build_plan_payload(spec)]
        self.assertEqual(
            step_ids,
            ['md.run', 'chain.compile_event_table', 'kmc.prepare_input', 'kmc.run', 'explain.summary', 'archive.results'],
        )

    def test_chain_plan_includes_resume_and_checkpoint_metadata(self):
        spec = load_job_spec(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'))
        payload = build_plan_payload(spec)
        by_id = {item['id']: item for item in payload}
        self.assertTrue(by_id['md.run']['mutating'])
        self.assertTrue(by_id['md.run']['resumable'])
        self.assertEqual(by_id['md.run']['checkpoint_kind'], 'state_store')
        self.assertTrue(by_id['kmc.run']['resumable'])
        self.assertEqual(by_id['archive.results']['checkpoint_kind'], 'archive')

    def test_event_table_and_barrier_map(self):
        payload = json.loads((ROOT / 'examples/sample-data/md/barriers.fe-cu-ni.json').read_text(encoding='utf-8'))
        rows = compile_event_table(payload, 773.0)
        barrier_map = derive_kmc_barrier_map(rows)
        self.assertEqual(set(barrier_map.keys()), {'Fe', 'Cu', 'Ni'})
        self.assertAlmostEqual(barrier_map['Fe'], 0.65)

    def test_render_kmc_input_contains_expected_commands(self):
        spec = load_job_spec(str(ROOT / 'examples/jobs/kmc_only.json'))
        template = spec['kmc']['template']
        text = render_kmc_input(template, {'Fe': 0.65, 'Cu': 0.56, 'Ni': 0.55}, 773.0, cluster_ref='cluster.xyz')
        self.assertIn('app_style vacancy', text)
        self.assertIn('plugin diffusion', text)
        self.assertIn('replace cluster.xyz', text)
        self.assertIn('barrier   Fe 0.650000', text)

    def test_dry_run_chain_generates_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = run_job(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'), tmpdir, dry_run=True)
            self.assertTrue((run_dir / 'state.json').exists())
            self.assertTrue((run_dir / 'artifacts/chain/event_table.csv').exists())
            self.assertTrue((run_dir / 'artifacts/kmc/generated_kmc.in').exists())
            self.assertTrue((run_dir / 'archive/manifest.json').exists())
            self.assertTrue((run_dir / 'explain/summary.md').exists())
            summary_text = (run_dir / 'explain/summary.md').read_text(encoding='utf-8')
            self.assertIn('执行来源 / 真实性', summary_text)
            self.assertIn('KMC: mode=dry-run', summary_text)
            inspected = inspect_run(run_dir)
            self.assertTrue(inspected['has_simulated_outputs'])
            self.assertEqual(inspected['execution_provenance']['stages']['kmc']['diffusion_mode'], 'simulated')

            with (run_dir / 'artifacts/chain/event_table.csv').open('r', encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['species'], 'Fe')

    def test_run_job_emits_progress_and_checkpoint_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_events = []
            checkpoint_events = []
            run_dir = run_job(
                str(ROOT / 'examples/jobs/md_to_kmc_chain.json'),
                tmpdir,
                dry_run=True,
                progress_callback=lambda stage, payload: progress_events.append((stage, payload)),
                checkpoint_callback=lambda stage, payload: checkpoint_events.append((stage, payload)),
            )
            self.assertTrue(run_dir.exists())
            stages = [stage for stage, _ in progress_events]
            self.assertIn('executor.job.start', stages)
            self.assertIn('executor.step.start', stages)
            self.assertIn('executor.step.complete', stages)
            self.assertIn('executor.job.complete', stages)
            checkpoint_stages = [stage for stage, _ in checkpoint_events]
            self.assertIn('executor.step.complete', checkpoint_stages)
            self.assertIn('executor.job.complete', checkpoint_stages)

    def test_run_job_requires_resume_or_overwrite_for_existing_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = run_job(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'), tmpdir, dry_run=True)
            self.assertTrue(first.exists())
            with self.assertRaises(RuntimeError) as ctx:
                run_job(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'), tmpdir, dry_run=True)
            self.assertIn('resume=True', str(ctx.exception))
            resumed = run_job(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'), tmpdir, dry_run=True, resume=True)
            self.assertEqual(resumed, first)

    def test_run_job_resume_records_recovery_summary_and_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = load_job_spec(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'))
            run_dir = run_job(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'), tmpdir, dry_run=True)
            resumed = run_job(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'), tmpdir, dry_run=True, resume=True)
            self.assertEqual(resumed, run_dir)
            state = json.loads((Path(tmpdir) / spec['job_id'] / 'state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['status'], 'completed')
            self.assertIn('resume_summary', state.get('job', {}))
            self.assertIn('md.run', state['job']['resume_summary']['completed_steps'])
            checkpoint_kinds = [item.get('kind') for item in state.get('checkpoints', [])]
            self.assertIn('executor.job.resume', checkpoint_kinds)
            self.assertIn('recovery_plan', state.get('job', {}))
            recovery_by_step = {item['step_id']: item for item in state['job']['recovery_plan']['steps']}
            self.assertEqual(recovery_by_step['md.run']['action'], 'reuse_completed')

    def test_run_job_cancel_hook_can_stop_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ExecutionCancelled):
                run_job(
                    str(ROOT / 'examples/jobs/md_to_kmc_chain.json'),
                    tmpdir,
                    dry_run=True,
                    cancel_check=lambda step_id, run_dir, state: "cancelled by test"
                    if step_id == 'chain.compile_event_table'
                    else None,
                )

    def test_run_job_cancel_persists_cancelled_state_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = load_job_spec(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'))
            with self.assertRaises(ExecutionCancelled):
                run_job(
                    str(ROOT / 'examples/jobs/md_to_kmc_chain.json'),
                    tmpdir,
                    dry_run=True,
                    cancel_check=lambda step_id, run_dir, state: "cancelled by test"
                    if step_id == 'chain.compile_event_table'
                    else None,
                )
            state = json.loads((Path(tmpdir) / spec['job_id'] / 'state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['status'], 'cancelled')
            self.assertEqual(state['steps']['chain.compile_event_table']['status'], 'cancelled')
            checkpoint_kinds = [item.get('kind') for item in state.get('checkpoints', [])]
            self.assertIn('executor.step.cancelled', checkpoint_kinds)
            self.assertIn('executor.job.cancelled', checkpoint_kinds)

    def test_run_job_resume_builds_step_level_recovery_plan_for_failed_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = load_job_spec(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'))
            run_dir = run_job(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'), tmpdir, dry_run=True)
            state_path = Path(tmpdir) / spec['job_id'] / 'state.json'
            state = json.loads(state_path.read_text(encoding='utf-8'))
            state['status'] = 'failed'
            state['steps']['kmc.run']['status'] = 'failed'
            state['steps']['kmc.run']['error'] = 'simulated failure'
            state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')

            resumed = run_job(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'), tmpdir, dry_run=True, resume=True)
            self.assertEqual(resumed, run_dir)
            new_state = json.loads(state_path.read_text(encoding='utf-8'))
            recovery_by_step = {item['step_id']: item for item in new_state['job']['recovery_plan']['steps']}
            self.assertEqual(recovery_by_step['kmc.run']['action'], 'restart_resumable_step')
            checkpoint_kinds = [item.get('kind') for item in new_state.get('checkpoints', [])]
            self.assertIn('executor.job.recovery_plan', checkpoint_kinds)
            self.assertIn('executor.step.resume', checkpoint_kinds)

    def test_run_job_resume_rebuilds_completed_step_when_artifact_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = load_job_spec(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'))
            run_dir = run_job(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'), tmpdir, dry_run=True)
            missing_path = run_dir / 'artifacts' / 'kmc' / 'generated_kmc.in'
            missing_path.unlink()

            resumed = run_job(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'), tmpdir, dry_run=True, resume=True)
            self.assertEqual(resumed, run_dir)
            state = json.loads((Path(tmpdir) / spec['job_id'] / 'state.json').read_text(encoding='utf-8'))
            recovery_by_step = {item['step_id']: item for item in state['job']['recovery_plan']['steps']}
            self.assertEqual(recovery_by_step['kmc.prepare_input']['action'], 'rebuild_from_checkpoint')
            self.assertFalse(recovery_by_step['kmc.prepare_input']['artifacts_valid'])
            self.assertIn('artifacts/kmc/generated_kmc.in', recovery_by_step['kmc.prepare_input']['missing_outputs'])
            self.assertEqual(recovery_by_step['kmc.prepare_input']['decision_reason'], 'completed-artifacts-missing')

    def test_run_job_resume_cascades_recovery_to_downstream_completed_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = load_job_spec(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'))
            run_dir = run_job(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'), tmpdir, dry_run=True)
            (run_dir / 'artifacts' / 'kmc' / 'generated_kmc.in').unlink()

            resumed = run_job(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'), tmpdir, dry_run=True, resume=True)
            self.assertEqual(resumed, run_dir)
            state = json.loads((Path(tmpdir) / spec['job_id'] / 'state.json').read_text(encoding='utf-8'))
            recovery_by_step = {item['step_id']: item for item in state['job']['recovery_plan']['steps']}
            self.assertEqual(recovery_by_step['kmc.run']['action'], 'restart_resumable_step')
            self.assertEqual(recovery_by_step['kmc.run']['decision_reason'], 'upstream-step-replanned')
            self.assertEqual(recovery_by_step['kmc.run']['invalidated_by'], 'kmc.prepare_input')
            self.assertEqual(recovery_by_step['explain.summary']['invalidated_by'], 'kmc.run')

    def test_run_job_resume_detects_drifted_artifact_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = load_job_spec(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'))
            run_dir = run_job(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'), tmpdir, dry_run=True)
            generated = run_dir / 'artifacts' / 'kmc' / 'generated_kmc.in'
            generated.write_text(generated.read_text(encoding='utf-8') + '\n# drifted by test\n', encoding='utf-8')

            resumed = run_job(str(ROOT / 'examples/jobs/md_to_kmc_chain.json'), tmpdir, dry_run=True, resume=True)
            self.assertEqual(resumed, run_dir)
            state = json.loads((Path(tmpdir) / spec['job_id'] / 'state.json').read_text(encoding='utf-8'))
            recovery_by_step = {item['step_id']: item for item in state['job']['recovery_plan']['steps']}
            self.assertEqual(recovery_by_step['kmc.prepare_input']['action'], 'rebuild_from_checkpoint')
            self.assertIn('artifacts/kmc/generated_kmc.in', recovery_by_step['kmc.prepare_input']['drifted_outputs'])
            self.assertEqual(recovery_by_step['kmc.prepare_input']['decision_reason'], 'artifact-fingerprint-mismatch')

    def test_md_command_can_generate_barriers_before_chaining(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            script = tmp / 'emit_barriers.py'
            script.write_text(
                """
import json
from pathlib import Path

payload = {
    "source": "generated-by-test",
    "attempt_frequency_hz": 1.0e13,
    "events": [
        {"event_id": "jump_fe", "species": "Fe", "barrier_ev": 0.65, "kmc_barrier_key": "Fe"},
        {"event_id": "jump_cu", "species": "Cu", "barrier_ev": 0.56, "kmc_barrier_key": "Cu"},
    ],
}
Path("barriers.json").write_text(json.dumps(payload), encoding="utf-8")
print("barriers generated")
""".strip(),
                encoding='utf-8',
            )
            spec_path = tmp / 'md_command.json'
            spec_path.write_text(
                json.dumps(
                    {
                        'job_id': 'md_command_exec',
                        'mode': 'md_only',
                        'md': {
                            'engine': 'lammps',
                            'command': ['python3', str(script)],
                            'working_dir': './workspace',
                            'barriers_source': 'barriers.json',
                        },
                    }
                ),
                encoding='utf-8',
            )

            run_dir = run_job(str(spec_path), tmpdir, dry_run=False)
            execution = json.loads((run_dir / 'artifacts/md/md_execution.json').read_text(encoding='utf-8'))
            barriers = json.loads((run_dir / 'artifacts/md/barriers.json').read_text(encoding='utf-8'))

            self.assertEqual(execution['mode'], 'executed')
            self.assertEqual(execution['returncode'], 0)
            self.assertEqual(barriers['source'], 'generated-by-test')
            self.assertIn('barriers generated', (run_dir / 'artifacts/md/md_execution.log').read_text(encoding='utf-8'))

    def test_run_command_parses_resume_overwrite_and_dry_run_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            with mock.patch.object(session, '_run_manual_tool_intent', return_value='ok') as run_manual_mock:
                reply = session._handle_command('/run --resume --overwrite --dry-run 继续这个算例')
            self.assertEqual(reply, 'ok')
            intent = run_manual_mock.call_args.args[0]
            self.assertEqual(intent.action, 'run')
            self.assertTrue(intent.params['resume_existing'])
            self.assertTrue(intent.params['overwrite_existing'])
            self.assertTrue(intent.params['dry_run_only'])

    def test_followups_command_lists_pending_followups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            source_turn_id = session._runtime_state.start_turn('先看最近一次 run')
            session._runtime_state.finish_turn(source_turn_id, reply='已完成', used_tools=False, status='chat')
            queued = session._runtime_state.queue_followup(
                source_turn_id,
                {
                    'kind': 'followup_prompt',
                    'text': '继续展开 kmc 日志',
                    'source_status': 'chat',
                    'source_finish_reason': 'chat',
                },
            )

            reply = session._handle_command('/followups')

            self.assertIn('Queued follow-ups', reply)
            self.assertIn(queued['followup_id'], reply)
            self.assertIn('继续展开 kmc 日志', reply)

    def test_resume_and_retry_commands_delegate_to_query_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            resume_result = QueryTurnResult(turn_id='turn-resume', reply='resume reply', used_tools=False, status='chat')
            retry_result = QueryTurnResult(turn_id='turn-retry', reply='retry reply', used_tools=False, status='chat')
            with mock.patch.object(engine, 'resolve_turn_reference', return_value='turn-source') as resolve_mock:
                with mock.patch.object(engine, 'resume_turn', return_value=resume_result) as resume_mock:
                    reply = session._handle_command('/resume latest 继续分析')
            self.assertEqual(reply, 'resume reply')
            resolve_mock.assert_called_once_with('latest')
            resume_mock.assert_called_once_with('turn-source', prompt='继续分析')

            with mock.patch.object(engine, 'resolve_turn_reference', return_value='turn-source') as resolve_mock:
                with mock.patch.object(engine, 'retry_turn', return_value=retry_result) as retry_mock:
                    reply = session._handle_command('/retry turn-source')
            self.assertEqual(reply, 'retry reply')
            resolve_mock.assert_called_once_with('turn-source')
            retry_mock.assert_called_once_with('turn-source', prompt=None)

    def test_continue_commands_delegate_to_query_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            with mock.patch.object(engine, 'continue_queued_followup', return_value='continued reply') as continue_mock:
                reply = session._handle_command('/continue followup-123')
            self.assertEqual(reply, 'continued reply')
            continue_mock.assert_called_once_with('followup-123')

            with mock.patch.object(engine, 'drain_queued_followups', return_value='drained replies') as drain_mock:
                reply = session._handle_command('/continue-all 2')
            self.assertEqual(reply, 'drained replies')
            drain_mock.assert_called_once_with(limit=2)

    def test_shell_status_includes_followup_and_abort_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            turn_id = session._runtime_state.start_turn('继续分析')
            session._runtime_state.finish_turn(turn_id, reply='done', used_tools=False, status='chat')
            session._runtime_state.queue_followup(
                turn_id,
                {'kind': 'followup_intent', 'text': '继续自动检查当前 run 的 `kmc` 日志。', 'action': 'logs', 'params': {'run': 'current', 'target': 'kmc'}, 'auto_continue': True},
            )
            session._runtime_state.aborted_turns.append({'turn_id': 'turn-aborted', 'reason': 'user stop'})

            reply = session._format_shell_status()

            self.assertIn('- queued follow-ups: 1', reply)
            self.assertIn('- runnable follow-ups: 1', reply)
            self.assertIn('- auto follow-ups: 1', reply)
            self.assertIn('- aborted turns: 1', reply)

    def test_runtime_state_rebuilds_memory_summary_and_prunes_stale_runtime_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            state = session._runtime_state
            source_turn_id = state.start_turn('先看最近一次 run')
            state.finish_turn(source_turn_id, reply='旧结论', used_tools=True, status='tool')
            retry_turn_id = state.begin_resumed_turn('重新分析这次 run', source_turn_id=source_turn_id, mode='retry')
            state.finish_turn(retry_turn_id, reply='新结论', used_tools=True, status='tool')
            for idx in range(15):
                item = state.queue_followup(source_turn_id, {'kind': 'followup_prompt', 'text': f'继续看第 {idx} 条日志'})
                state.consume_queued_followup(item['followup_id'], status='completed')
            state.aborted_turns = [{'turn_id': f'turn-aborted-{idx}', 'reason': 'stop'} for idx in range(15)]

            summary = state.rebuild_memory_summary(live_turn_window=0)
            state.prune_stale_runtime_state()

            self.assertEqual(summary['archived_turn_count'], 2)
            self.assertEqual(summary['stale_memory_count'], 1)
            self.assertEqual(state.memory_records[0]['turn_id'], source_turn_id)
            self.assertTrue(state.memory_records[0]['stale'])
            self.assertLessEqual(len([item for item in state.queued_followups if bool(item.get('consumed'))]), 12)
            self.assertEqual(len(state.aborted_turns), 12)

    def test_build_engine_context_exposes_memory_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            archived_turn_id = session._runtime_state.start_turn('先总结最近一次 run')
            session._runtime_state.record_turn_message(archived_turn_id, 'user', '先总结最近一次 run')
            session._runtime_state.finish_turn(archived_turn_id, reply='旧总结', used_tools=False, status='chat')
            session._runtime_state.set_turn_finish_reason(archived_turn_id, status='chat', reason='chat')
            session._runtime_state.rebuild_memory_summary(live_turn_window=0)

            context = session._build_engine_context()

            self.assertIn('memory_context', context)
            self.assertEqual(context['memory_context']['archived_turn_count'], 1)
            self.assertEqual(context['memory_context']['fresh_memory_count'], 1)
            self.assertIn('Earlier working memory', context['memory_context']['summary'])

    def test_local_model_messages_include_memory_summary_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            archived_turn_id = session._runtime_state.start_turn('先总结最近一次 run')
            session._runtime_state.finish_turn(archived_turn_id, reply='旧总结', used_tools=False, status='chat')
            session._runtime_state.set_turn_finish_reason(archived_turn_id, status='chat', reason='chat')
            session._runtime_state.rebuild_memory_summary(live_turn_window=0)

            messages = session._build_local_model_messages()

            memory_messages = [
                message for message in messages
                if 'working-memory summary condenses older conversation state' in message.get('content', '')
            ]
            self.assertEqual(len(memory_messages), 1)
            self.assertIn('Earlier working memory from archived turns', memory_messages[0]['content'])

    def _make_synthetic_run(
        self,
        output_root: Path,
        job_id: str,
        *,
        mode: str = 'md_to_kmc_chain',
        temperature_k: float = 800.0,
        barriers=None,
        jump_frequency: float = 1.0e9,
        diffusion_coefficient: float = 1.0e6,
    ) -> Path:
        run_dir = output_root / job_id
        (run_dir / 'artifacts' / 'md').mkdir(parents=True, exist_ok=True)
        (run_dir / 'artifacts' / 'kmc').mkdir(parents=True, exist_ok=True)
        steps = {
            'md.run': {'status': 'completed'},
            'chain.compile_event_table': {'status': 'completed'},
            'kmc.prepare_input': {'status': 'completed'},
            'kmc.run': {'status': 'completed'},
        }
        (run_dir / 'state.json').write_text(
            json.dumps(
                {
                    'job_id': job_id,
                    'mode': mode,
                    'steps': steps,
                }
            ),
            encoding='utf-8',
        )
        (run_dir / 'job_spec.resolved.json').write_text(
            json.dumps(
                {
                    'job_id': job_id,
                    'mode': mode,
                    'material_system': {'name': job_id},
                    'kmc': {'temperature_k': temperature_k},
                }
            ),
            encoding='utf-8',
        )
        barrier_rows = barriers or {'Fe': 0.875, 'Cu': 0.536, 'Ni': 0.550}
        (run_dir / 'artifacts' / 'md' / 'barriers.json').write_text(
            json.dumps(
                {
                    'metadata': {'workflow_kind': mode, 'barrier_source_mode': 'lammps-neb'},
                    'events': [
                        {'species': species, 'barrier_ev': value, 'barrier_source': 'neb'}
                        for species, value in barrier_rows.items()
                    ],
                }
            ),
            encoding='utf-8',
        )
        (run_dir / 'artifacts' / 'kmc' / 'diffusion.csv').write_text(
            '\n'.join(
                [
                    'No.,jumps,msd,simulation_time,jump frequency,diffusion coefficient',
                    f'0,44,0.77,2.4e-08,{jump_frequency},{diffusion_coefficient}',
                ]
            )
            + '\n',
            encoding='utf-8',
        )
        return run_dir

    def _make_moire_summary_run(
        self,
        output_root: Path,
        job_id: str,
        *,
        barrier_ev: float = 0.59798,
        accepted_events: int = 6,
    ) -> Path:
        run_dir = output_root / job_id
        (run_dir / "kmc_bridge").mkdir(parents=True, exist_ok=True)
        (run_dir / "lammps_case").mkdir(parents=True, exist_ok=True)
        (run_dir / "lammps_run.out").write_text("LAMMPS ran\n", encoding="utf-8")
        (run_dir / "kmc_bridge" / "run.out").write_text("KMC ran\n", encoding="utf-8")
        summary = {
            "status": "completed",
            "source_case_dir": "/tmp/MoRe/Re_0.07/model_4",
            "copied_case_dir": str(run_dir / "lammps_case"),
            "generated_lammps_input": str(run_dir / "lammps_case" / "generated_in.neb.mietclaw"),
            "generated_barrier_script": str(run_dir / "lammps_case" / "extract_barrier.mietclaw.sh"),
            "neb_txt": str(run_dir / "lammps_case" / "neb.txt"),
            "barrier_eV": barrier_ev,
            "lammps": {
                "status": "executed",
                "log": str(run_dir / "lammps_run.out"),
            },
            "postprocess": {
                "status": "executed",
            },
            "kmc": {
                "status": "completed",
                "barrier_eV": barrier_ev,
                "barrier_assignment": {"Mo": barrier_ev, "Re": barrier_ev},
                "parsed_run": {
                    "accepted_events": accepted_events,
                    "final_time": 1.0e-10,
                },
                "files": {
                    "run_out": str(run_dir / "kmc_bridge" / "run.out"),
                },
                "runtime_health": {"status": "ok", "warnings": []},
            },
            "runtime_health": {"status": "ok", "warnings": []},
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return run_dir

    def test_inspect_run_includes_temperature_and_kmc_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_synthetic_run(
                Path(tmpdir),
                'inspect_metrics_case',
                temperature_k=835.0,
                jump_frequency=2.5e9,
                diffusion_coefficient=7.8e6,
            )
            info = inspect_run(run_dir)
            self.assertEqual(info['temperature_k'], 835.0)
            self.assertEqual(info['latest_diffusion']['jump_frequency'], 2.5e9)
            self.assertEqual(info['latest_diffusion']['diffusion_coefficient'], 7.8e6)

    def test_list_runs_includes_moire_summary_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_moire_summary_run(tmp, 'moire_case_run')
            runs = list_runs(tmp, limit=5)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]['job_id'], 'moire_case_run')
            self.assertEqual(runs[0]['mode'], 'moire_lammps_to_kmc')
            self.assertEqual(runs[0]['status'], 'completed')

    def test_inspect_run_reads_moire_summary_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = self._make_moire_summary_run(tmp, 'moire_case_run', barrier_ev=0.611, accepted_events=9)
            info = inspect_run(run_dir)
            self.assertEqual(info['kind'], 'moire_lammps_to_kmc')
            self.assertEqual(info['status'], 'completed')
            self.assertEqual(info['accepted_events'], 9)
            self.assertAlmostEqual(info['barrier_eV'], 0.611)
            self.assertEqual(info['events'][0]['species'], 'Mo')

    def test_chat_helpers_list_artifacts_and_logs_for_moire_summary_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = self._make_moire_summary_run(Path(tmpdir), 'moire_case_run')
            artifacts = list_artifacts(run_dir)
            log_info = get_log_excerpt(run_dir, target='md', max_lines=5)
            self.assertIn('summary.json', artifacts)
            self.assertIn('lammps_run.out', artifacts)
            self.assertTrue(log_info['available'])
            self.assertEqual(log_info['target'], 'md')
            self.assertIn('LAMMPS ran', log_info['content'])

    def test_compare_recent_runs_prompt_reports_temperature_and_kmc_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_synthetic_run(
                tmp,
                'chain_780k',
                temperature_k=780.0,
                jump_frequency=1.1e9,
                diffusion_coefficient=3.2e6,
            )
            self._make_synthetic_run(
                tmp,
                'chain_900k',
                temperature_k=900.0,
                jump_frequency=4.6e9,
                diffusion_coefficient=9.4e6,
            )
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            reply = session.handle_line('请比较最近两个 MD→KMC run，告诉我 barrier、jump frequency、diffusion coefficient 和温度怎么变化。')
            self.assertIn('temperature: 900.0 K', reply)
            self.assertIn('temperature: 780.0 K', reply)
            self.assertIn('jump_frequency=4.6e+09', reply)
            self.assertIn('diffusion_coefficient=9.4e+06', reply)
            self.assertIn('更高温的是 chain_900k', reply)

    def test_autonomy_workspace_materializes_generated_scripts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = materialize_autonomy_workspace(
                prompt='Create a native MD to KMC vacancy diffusion job for "Autonomy Draft Demo" at 910 K owned by miet.',
                project_root=str(ROOT),
                workspace_root=tmpdir,
                provider='local',
            )

            spec_path = Path(report['generated_files']['job_spec'])
            spec = json.loads(spec_path.read_text(encoding='utf-8'))
            md_script = Path(report['generated_files']['md_script'])
            kmc_preview = Path(report['generated_files']['kmc_preview_input'])
            neb_campaign = Path(report['generated_files']['md_neb_campaign'])
            neb_primary_input = Path(report['generated_files']['md_neb_primary_input'])

            self.assertTrue(spec_path.exists())
            self.assertTrue(md_script.exists())
            self.assertTrue(kmc_preview.exists())
            self.assertTrue(neb_campaign.exists())
            self.assertTrue(neb_primary_input.exists())
            self.assertEqual(spec['mode'], 'md_to_kmc_chain')
            self.assertEqual(spec['material_system']['name'], 'Autonomy Draft Demo')
            self.assertAlmostEqual(spec['kmc']['temperature_k'], 910.0)
            self.assertEqual(spec['md']['command'][-1], str(md_script))
            self.assertIn('mietclaw-autonomy-neb-workflow', md_script.read_text(encoding='utf-8'))
            self.assertIn('fix             nebfix mobile neb', neb_primary_input.read_text(encoding='utf-8'))
            self.assertIn('run hint', Path(report['generated_files']['md_neb_readme']).read_text(encoding='utf-8'))
            self.assertTrue(Path(report['generated_files']['plan_script']).exists())
            self.assertTrue(Path(report['generated_files']['run_script']).exists())

    def test_autonomy_run_dry_run_only_completes_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_autonomy_job(
                prompt='Run a KMC only Fe-Cu-Ni diffusion job at 805 K with Fe=0.61 eV, Cu=0.53 eV, Ni=0.52 eV.',
                project_root=str(ROOT),
                workspace_root=tmpdir,
                provider='local',
                dry_run_only=True,
            )

            validation_run_dir = Path(report['execution']['validation_run_dir'])
            state = json.loads((validation_run_dir / 'state.json').read_text(encoding='utf-8'))
            spec = json.loads(Path(report['generated_files']['job_spec']).read_text(encoding='utf-8'))

            self.assertEqual(spec['mode'], 'kmc_only')
            self.assertAlmostEqual(spec['kmc']['temperature_k'], 805.0)
            self.assertAlmostEqual(spec['kmc']['template']['precomputed_barriers']['Fe'], 0.61)
            self.assertEqual(state['steps']['kmc.run']['status'], 'completed')
            self.assertIsNone(report['execution']['final_run_dir'])
            self.assertTrue((validation_run_dir / 'artifacts/kmc/generated_kmc.in').exists())
            self.assertIsNotNone(report['execution']['validation_recovery'])
            self.assertGreater(report['execution']['validation_recovery']['checkpoint_count'], 0)
            self.assertIn('recovery_plan', report['execution']['validation_recovery'])

    def test_autonomy_extracts_owned_by_owner_phrase(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = materialize_autonomy_workspace(
                prompt='Create a native MD to KMC vacancy diffusion job for "Owner Parse Demo" at 880 K owned by alice.',
                project_root=str(ROOT),
                workspace_root=tmpdir,
                provider='local',
            )

            spec = json.loads(Path(report['generated_files']['job_spec']).read_text(encoding='utf-8'))
            self.assertEqual(report['facts']['owner'], 'alice')
            self.assertEqual(spec['material_system']['owner'], 'alice')

    def test_parse_neb_terse_output_extracts_barrier(self):
        stdout = """
LAMMPS (29 Aug 2024)
Running on 7 partitions of processors
Reading NEB coordinate file(s) ...
Setting up regular NEB ...
    Step     MaxReplicaForce  MaxAtomForce      GradV0         GradV1         GradVc          EBF            EBR            RDT
         0   0.37897973       0.047875211    0.37897973     0.37897973     1.8217471e-15  0.86433994     0.86433994     2.4725025
        10   0.00022976827    1.7597819e-08  0.00022976827  0.00022976827  1.5827981e-15  0.87467129     0.87467129     2.3612504
Setting up climbing ...
Climbing replica = 4
    Step     MaxReplicaForce  MaxAtomForce      GradV0         GradV1         GradVc          EBF            EBR            RDT
        10   0.00022976827    1.7597819e-08  0.00022976827  0.00022976827  1.5827981e-15  0.87467129     0.87467129     2.3612504
        11   0.00013085286    5.7074907e-09  0.00013085286  0.00013085286  1.5313433e-15  0.87467129     0.87467129     2.3612821
"""
        parsed = parse_neb_terse_output(stdout)
        self.assertTrue(parsed['parsed'])
        self.assertEqual(parsed['climbing_replica'], 4)
        self.assertAlmostEqual(parsed['barrier_forward_ev'], 0.87467129)
        self.assertAlmostEqual(parsed['barrier_reverse_ev'], 0.87467129)
        self.assertEqual(parsed['converged_step'], 11)

    def test_chat_helpers_list_artifacts_and_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'demo_run'
            (run_dir / 'artifacts' / 'md').mkdir(parents=True, exist_ok=True)
            (run_dir / 'artifacts' / 'kmc').mkdir(parents=True, exist_ok=True)
            (run_dir / 'state.json').write_text(json.dumps({'steps': {}}), encoding='utf-8')
            (run_dir / 'artifacts' / 'md' / 'md_execution.log').write_text('line1\nline2\nline3\n', encoding='utf-8')
            (run_dir / 'artifacts' / 'kmc' / 'generated_kmc.in').write_text('kmc input', encoding='utf-8')

            artifacts = list_artifacts(run_dir)
            log_info = get_log_excerpt(run_dir, target='md', max_lines=2)

            self.assertIn('artifacts/kmc/generated_kmc.in', artifacts)
            self.assertTrue(log_info['available'])
            self.assertEqual(log_info['target'], 'md')
            self.assertEqual(log_info['content'], 'line2\nline3')

    def test_build_shell_status_uses_summary_runs_for_latest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = self._make_moire_summary_run(tmp, 'moire_case_run')
            status = build_shell_status(
                project_root=ROOT,
                workspace_root=tmp,
                output_dir=tmp,
                provider='auto',
                selected_model='demo-model',
                local_status={'healthy': True, 'default_model': 'demo-model'},
                current_run_dir=None,
            )
            self.assertEqual(status['latest_run_dir'], str(run_dir))
            self.assertEqual(status['tool_approval_policy'], 'allow_all')

    def test_tool_router_detects_runs_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            intent = heuristic_tool_intent('请列出最近的 runs', Path(tmpdir))
            self.assertIsNotNone(intent)
            self.assertEqual(intent.action, 'runs')

    def test_tool_router_detects_bridge_request_from_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            event_json = tmp / 'event.json'
            neb_dir = tmp / 'neb_case'
            neb_dir.mkdir()
            neb_txt = neb_dir / 'neb.txt'
            event_json.write_text('{}', encoding='utf-8')
            neb_txt.write_text('#reaction_coordinate de\n0 0\n0.5 0.42\n', encoding='utf-8')
            prompt = f'请把 {event_json} 和 {neb_txt} bridge 成 KMC lookup，并验证一下。'
            intent = heuristic_tool_intent(prompt, tmp)
            self.assertIsNotNone(intent)
            self.assertEqual(intent.action, 'bridge_kmc_lookup')
            self.assertEqual(intent.params['event_json'], str(event_json))
            self.assertEqual(intent.params['neb_txt'], str(neb_txt))
            self.assertTrue(intent.params['validate'])

    def test_tool_router_detects_lammps_barrier_to_kmc_workflow_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = '我现在想用这个agent来实现lammps计算迁移能垒，用代码仓库里的kmc软件通过lammps传递的迁移能垒结果来继续kmc模拟'
            intent = heuristic_tool_intent(prompt, Path(tmpdir))
            self.assertIsNotNone(intent)
            self.assertEqual(intent.action, 'draft')

    def test_tool_router_detects_direct_launch_for_lammps_barrier_to_kmc_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt = '请直接运行一个 native LAMMPS 迁移能垒计算，并把结果传给代码仓库里的 KMC 软件继续模拟。'
            intent = heuristic_tool_intent(prompt, Path(tmpdir))
            self.assertIsNotNone(intent)
            self.assertEqual(intent.action, 'run')

    def test_tool_router_detects_moire_run_with_event_and_case_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            event_json = tmp / 'event.json'
            case_dir = tmp / 'MoRe' / 'Re_0.07' / 'model_4'
            case_dir.mkdir(parents=True)
            event_json.write_text('{}', encoding='utf-8')
            prompt = f'请直接在本机上跑 MoRe 的 LAMMPS，然后把结果接到 KMC：{event_json} {case_dir}'
            intent = heuristic_tool_intent(prompt, tmp)
            self.assertIsNotNone(intent)
            self.assertEqual(intent.action, 'moire_run')
            self.assertEqual(intent.params['event_json'], str(event_json))
            self.assertEqual(intent.params['case_dir'], str(case_dir))

    def test_tool_router_detects_moire_run_seed_ensemble_and_ovito(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            case_dir = tmp / 'MoRe' / 'Re_0.07' / 'model_4'
            case_dir.mkdir(parents=True)
            prompt = f'请直接在本机上跑 MoRe 的 LAMMPS，然后把结果接到 KMC，用 OVITO 可视化，并使用随机种子 4101,4102,4103：{case_dir}'
            intent = heuristic_tool_intent(prompt, tmp)
            self.assertIsNotNone(intent)
            self.assertEqual(intent.action, 'moire_run')
            self.assertEqual(intent.params['case_dir'], str(case_dir))
            self.assertEqual(intent.params['kmc_seeds'], [4101, 4102, 4103])
            self.assertTrue(intent.params['ovito'])

    def test_tool_router_detects_moire_compare_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            case_dir = tmp / 'MoRe' / 'Re_0.07' / 'model_4'
            case_dir.mkdir(parents=True)
            event_a = tmp / 'event_a.json'
            event_b = tmp / 'event_b.json'
            event_a.write_text('{}', encoding='utf-8')
            event_b.write_text('{}', encoding='utf-8')
            prompt = f'请比较这两个 MoRe event 的 barrier，并接着跑 KMC，用 OVITO 可视化，随机种子 4101,4102：{case_dir} {event_a} {event_b}'
            intent = heuristic_tool_intent(prompt, tmp)
            self.assertIsNotNone(intent)
            self.assertEqual(intent.action, 'moire_compare')
            self.assertEqual(intent.params['case_dir'], str(case_dir))
            self.assertEqual(intent.params['event_jsons'], [str(event_a), str(event_b)])
            self.assertEqual(intent.params['kmc_seeds'], [4101, 4102])
            self.assertTrue(intent.params['ovito'])

    def test_tool_router_detects_moire_diffusion_sweep_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            case_dir = tmp / 'MoRe' / 'Re_0.07' / 'model_4'
            case_dir.mkdir(parents=True)
            event_json = tmp / 'event.json'
            event_json.write_text('{}', encoding='utf-8')
            prompt = (
                f'请先用这个 event 做 MoRe 的 LAMMPS barrier，再扫温 700K 800K 900K，'
                f'算扩散系数与温度的关系，并用 OVITO 可视化，随机种子 4101,4102：{event_json} {case_dir}'
            )
            intent = heuristic_tool_intent(prompt, tmp)
            self.assertIsNotNone(intent)
            self.assertEqual(intent.action, 'moire_diffusion_sweep')
            self.assertEqual(intent.params['event_json'], str(event_json))
            self.assertEqual(intent.params['case_dir'], str(case_dir))
            self.assertEqual(intent.params['temperatures_k'], [700.0, 800.0, 900.0])
            self.assertEqual(intent.params['kmc_seeds'], [4101, 4102])
            self.assertTrue(intent.params['ovito'])

    def test_tool_router_can_use_default_moire_paths_when_prompt_has_no_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            case_dir = tmp / 'MoRe' / 'Re_0.07' / 'model_4'
            case_dir.mkdir(parents=True)
            prompt = '我希望你把 MoRe的迁移能垒计算出来并把迁移能垒作为kmc软件的输入进行kmc模拟'
            with mock.patch.dict('os.environ', {'MIETCLAW_MOIRE_CASE_DIR': str(case_dir)}, clear=False):
                intent = heuristic_tool_intent(prompt, tmp)
            self.assertIsNotNone(intent)
            self.assertEqual(intent.action, 'moire_run')
            self.assertEqual(intent.params['case_dir'], str(case_dir.resolve()))
            self.assertNotIn('event_json', intent.params)

    def test_extract_material_name_uses_generic_native_chain_label(self):
        prompt = '我现在想用这个agent来实现lammps计算迁移能垒，用代码仓库里的kmc软件通过lammps传递的迁移能垒结果来继续kmc模拟'
        self.assertEqual(extract_material_name(prompt), 'LAMMPS to KMC migration barrier workflow')

    def test_parse_tool_intent_from_model_json(self):
        intent = parse_tool_intent('{"action":"runs","reply":null}')
        self.assertIsNotNone(intent)
        self.assertEqual(intent.action, 'runs')

    def test_tool_plan_detects_log_summary_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = heuristic_tool_plan('请检查最新那个 run 的 KMC 日志并总结一下。', Path(tmpdir))
            self.assertIsNotNone(plan)
            self.assertEqual([step.action for step in plan.steps], ['inspect', 'logs'])
            self.assertTrue(plan.summarize)

    def test_tool_router_detects_compare_latest_two_runs_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            intent = heuristic_tool_intent(
                '请比较最近两个 MD→KMC run，告诉我 barrier、jump frequency、diffusion coefficient 和温度怎么变化。',
                Path(tmpdir),
            )
            self.assertIsNotNone(intent)
            self.assertEqual(intent.action, 'compare_runs')
            self.assertEqual(intent.params['run'], 'latest_two')
            self.assertEqual(intent.params['mode'], 'md_to_kmc_chain')

    def test_tool_plan_diagnoses_latest_md_to_kmc_run_with_matching_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = heuristic_tool_plan(
                '请检查最新这个 MD→KMC run 是否真的正常结束。不要只看 completed 状态；还要检查 LAMMPS 执行记录、KMC 执行记录和相关日志。如果有异常痕迹，请直接指出。',
                Path(tmpdir),
            )
            self.assertIsNotNone(plan)
            self.assertEqual([step.action for step in plan.steps], ['inspect', 'logs', 'logs'])
            self.assertEqual(plan.steps[0].params['run'], 'latest')
            self.assertEqual(plan.steps[0].params['mode'], 'md_to_kmc_chain')
            self.assertEqual(plan.steps[1].params, {'run': 'current', 'target': 'md'})
            self.assertEqual(plan.steps[2].params, {'run': 'current', 'target': 'kmc'})
            self.assertTrue(plan.summarize)

    def test_parse_tool_plan_from_model_json(self):
        plan = parse_tool_plan(
            '{"steps":[{"action":"inspect","params":{"run":"latest"}},{"action":"logs","params":{"run":"latest","target":"kmc"}}],"summarize":true,"reply":null}'
        )
        self.assertIsNotNone(plan)
        self.assertEqual([step.action for step in plan.steps], ['inspect', 'logs'])
        self.assertTrue(plan.summarize)

    def test_skip_tool_router_for_plain_chat(self):
        self.assertTrue(should_skip_tool_router('你好，简单介绍一下你自己。'))
        self.assertTrue(should_skip_tool_router('What can you do?'))
        self.assertFalse(should_skip_tool_router('请列出最近的 runs。'))
        self.assertTrue(should_try_tool_plan('请检查最新那个 run 的 KMC 日志并总结一下。'))

    def test_router_golden_eval_fixture_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = run_router_golden_eval(ROOT / 'examples' / 'evals' / 'router_golden.json', output_dir=tmpdir)
        self.assertTrue(payload['ok'])
        self.assertEqual(payload['failed'], 0)
        self.assertGreaterEqual(payload['passed'], 8)

    def test_runtime_health_golden_eval_fixture_passes(self):
        payload = run_runtime_health_golden_eval(ROOT / 'examples' / 'evals' / 'runtime_health_golden.json')
        self.assertTrue(payload['ok'])
        self.assertEqual(payload['failed'], 0)
        self.assertGreaterEqual(payload['passed'], 4)

    def test_model_alias_resolves_27b(self):
        models = [
            'Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-4bit',
            'Qwen3.5-122B-A10B-4bit',
        ]
        self.assertEqual(_resolve_model_alias('27b', models), models[0])
        self.assertEqual(_resolve_model_alias('122b', models), models[1])

    def test_local_agent_profile_defaults_to_repo_config(self):
        profile = load_local_agent_profile()
        settings = get_local_model_settings()
        self.assertEqual(profile['agent_name'], 'mietclaw')
        self.assertEqual(settings['preferred_model'], '27b')
        self.assertEqual(settings['base_url'], 'http://127.0.0.1:8000')

    def test_local_agent_profile_can_be_overridden_by_env_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / 'local-agent.json'
            profile_path.write_text(
                json.dumps(
                    {
                        'agent_name': 'custom-agent',
                        'local_model': {
                            'base_url': 'http://127.0.0.1:9000',
                            'api_key': 'custom-key',
                            'preferred_model': '122b',
                        },
                        'runtime': {
                            'conda_exec': '/tmp/custom-conda',
                            'conda_env': 'custom-env',
                            'mpi_procs': 7,
                        },
                        'moire': {
                            'case_dir': 'relative-case',
                            'kmc_retry_attempts': 2,
                        },
                    }
                ),
                encoding='utf-8',
            )
            with mock.patch.dict('os.environ', {'MIETCLAW_LOCAL_PROFILE_FILE': str(profile_path)}, clear=False):
                settings = get_local_model_settings()
                runtime = get_runtime_settings(Path(tmpdir))
            self.assertEqual(settings['agent_name'], 'custom-agent')
            self.assertEqual(settings['base_url'], 'http://127.0.0.1:9000')
            self.assertEqual(settings['preferred_model'], '122b')
            self.assertEqual(runtime['conda_exec'], str(Path('/tmp/custom-conda').resolve()))
            self.assertEqual(runtime['conda_env'], 'custom-env')
            self.assertEqual(runtime['mpi_procs'], 7)
            self.assertEqual(runtime['moire_case_dir'], str((Path(tmpdir) / 'relative-case').resolve()))
            self.assertEqual(runtime['kmc_retry_attempts'], 2)

    def test_chat_falls_back_to_normal_reply_when_router_returns_chat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            healthy = {
                'healthy': True,
                'default_model': 'demo-model',
                'models': ['demo-model'],
                'base_url': 'http://127.0.0.1:8000',
            }
            with mock.patch('miet_claw.chat.chat_with_local_model') as mocked_chat:
                mocked_chat.side_effect = [
                    {'content': '{"action":"chat","reply":"我是工具路由器"}', 'model': 'demo-model'},
                    {'content': '你好，我是 mietclaw。', 'model': 'demo-model'},
                ]
                with mock.patch.object(session, '_refresh_local_model_status', return_value=healthy):
                    reply = session.handle_line('解释一下这个系统是做什么的。')

            self.assertEqual(reply, '你好，我是 mietclaw。')

    def test_multi_step_tool_plan_executes_and_summarizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'demo_run'
            (run_dir / 'artifacts' / 'md').mkdir(parents=True, exist_ok=True)
            (run_dir / 'artifacts' / 'kmc').mkdir(parents=True, exist_ok=True)
            (run_dir / 'explain').mkdir(parents=True, exist_ok=True)
            (run_dir / 'state.json').write_text(json.dumps({'steps': {'kmc.run': {'status': 'completed'}}}), encoding='utf-8')
            (run_dir / 'job_spec.resolved.json').write_text(
                json.dumps({'mode': 'kmc_only', 'material_system': {'name': 'Demo'}}), encoding='utf-8'
            )
            (run_dir / 'artifacts' / 'kmc' / 'log.spparks').write_text('step1\nstep2\nstep3\n', encoding='utf-8')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            healthy = {
                'healthy': True,
                'default_model': 'demo-model',
                'models': ['demo-model'],
                'base_url': 'http://127.0.0.1:8000',
            }
            with mock.patch('miet_claw.chat.chat_with_local_model') as mocked_chat:
                mocked_chat.return_value = {'content': '这个 run 已完成，KMC 日志末尾是 step3。', 'model': 'demo-model'}
                with mock.patch.object(session, '_refresh_local_model_status', return_value=healthy):
                    reply = session.handle_line('请检查最新那个 run 的 KMC 日志并总结一下。')
            self.assertIn('KMC 日志', reply)

    def test_agent_loop_can_continue_then_finish(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'demo_run'
            (run_dir / 'artifacts' / 'kmc').mkdir(parents=True, exist_ok=True)
            (run_dir / 'state.json').write_text(json.dumps({'steps': {'kmc.run': {'status': 'completed'}}}), encoding='utf-8')
            (run_dir / 'job_spec.resolved.json').write_text(
                json.dumps({'mode': 'kmc_only', 'material_system': {'name': 'Demo Agent Loop'}}), encoding='utf-8'
            )
            (run_dir / 'artifacts' / 'kmc' / 'log.spparks').write_text('normal\nMPI_ABORT\n', encoding='utf-8')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            healthy = {
                'healthy': True,
                'default_model': 'demo-model',
                'models': ['demo-model'],
                'base_url': 'http://127.0.0.1:8000',
            }
            with mock.patch('miet_claw.chat.chat_with_local_model') as mocked_chat:
                mocked_chat.side_effect = [
                    {'content': '{"status":"continue","step":{"action":"logs","params":{"run":"latest","target":"kmc"}},"reply":null}', 'model': 'demo-model'},
                    {'content': '{"status":"finish","step":null,"reply":"这个 run 有异常，KMC 日志里出现了 MPI_ABORT。"}', 'model': 'demo-model'},
                ]
                with mock.patch.object(session, '_refresh_local_model_status', return_value=healthy):
                    reply = session.handle_line('帮我判断最新那个 run 是正常结束还是异常退出，如果有必要再看 KMC 日志。')
            self.assertIn('MPI_ABORT', reply)

    def test_agent_loop_forces_kmc_log_check_for_normality_question(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'demo_run'
            (run_dir / 'artifacts' / 'kmc').mkdir(parents=True, exist_ok=True)
            (run_dir / 'state.json').write_text(json.dumps({'steps': {'kmc.run': {'status': 'completed'}}}), encoding='utf-8')
            (run_dir / 'job_spec.resolved.json').write_text(
                json.dumps({'mode': 'kmc_only', 'material_system': {'name': 'Demo Force Log'}}), encoding='utf-8'
            )
            (run_dir / 'artifacts' / 'kmc' / 'log.spparks').write_text('tail\nMPI_ABORT\n', encoding='utf-8')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            healthy = {
                'healthy': True,
                'default_model': 'demo-model',
                'models': ['demo-model'],
                'base_url': 'http://127.0.0.1:8000',
            }
            with mock.patch('miet_claw.chat.chat_with_local_model') as mocked_chat:
                mocked_chat.return_value = {
                    'content': '{"status":"finish","step":null,"reply":"这个 run 不是正常结束，KMC 日志里有 MPI_ABORT。"}',
                    'model': 'demo-model',
                }
                with mock.patch.object(session, '_refresh_local_model_status', return_value=healthy):
                    reply = session.handle_line('帮我判断最新那个 run 是正常结束还是异常退出，如果有必要再看 KMC 日志。')
            self.assertIn('MPI_ABORT', reply)
            self.assertEqual(mocked_chat.call_count, 1)

    def test_agent_loop_forces_kmc_log_check_for_failure_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'demo_run'
            (run_dir / 'artifacts' / 'kmc').mkdir(parents=True, exist_ok=True)
            (run_dir / 'state.json').write_text(json.dumps({'steps': {'kmc.run': {'status': 'failed'}}}), encoding='utf-8')
            (run_dir / 'job_spec.resolved.json').write_text(
                json.dumps({'mode': 'kmc_only', 'material_system': {'name': 'Demo Root Cause'}}), encoding='utf-8'
            )
            (run_dir / 'artifacts' / 'kmc' / 'log.spparks').write_text('setup\nMPI_ABORT: rank 0\n', encoding='utf-8')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            healthy = {
                'healthy': True,
                'default_model': 'demo-model',
                'models': ['demo-model'],
                'base_url': 'http://127.0.0.1:8000',
            }
            with mock.patch('miet_claw.chat.chat_with_local_model') as mocked_chat:
                mocked_chat.return_value = {
                    'content': '{"status":"finish","step":null,"reply":"根因是 KMC 日志里的 MPI_ABORT。"}',
                    'model': 'demo-model',
                }
                with mock.patch.object(session, '_refresh_local_model_status', return_value=healthy):
                    reply = session.handle_line('帮我找出最新那个 run 为什么失败，如果有必要继续看日志。')
            self.assertIn('MPI_ABORT', reply)
            self.assertEqual(mocked_chat.call_count, 1)

    def test_md_to_kmc_diagnosis_prompt_uses_latest_matching_chain_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            kmc_only_dir = tmp / 'latest_kmc_only'
            (kmc_only_dir / 'artifacts' / 'kmc').mkdir(parents=True, exist_ok=True)
            (kmc_only_dir / 'state.json').write_text(json.dumps({'steps': {'kmc.run': {'status': 'completed'}}}), encoding='utf-8')
            (kmc_only_dir / 'job_spec.resolved.json').write_text(
                json.dumps({'mode': 'kmc_only', 'material_system': {'name': '830K KMC'}, 'kmc': {'temperature_k': 830.0}}),
                encoding='utf-8',
            )
            (kmc_only_dir / 'artifacts' / 'kmc' / 'log.spparks').write_text('kmc only latest\n', encoding='utf-8')

            chain_dir = tmp / 'lammps_kmc_demo_4_diagnose'
            (chain_dir / 'artifacts' / 'md').mkdir(parents=True, exist_ok=True)
            (chain_dir / 'artifacts' / 'kmc').mkdir(parents=True, exist_ok=True)
            (chain_dir / 'state.json').write_text(
                json.dumps(
                    {
                        'steps': {
                            'md.run': {'status': 'completed'},
                            'kmc.run': {'status': 'completed'},
                        }
                    }
                ),
                encoding='utf-8',
            )
            (chain_dir / 'job_spec.resolved.json').write_text(
                json.dumps({'mode': 'md_to_kmc_chain', 'material_system': {'name': 'Diagnose Chain'}, 'kmc': {'temperature_k': 810.0}}),
                encoding='utf-8',
            )
            (chain_dir / 'artifacts' / 'md' / 'barriers.json').write_text(
                json.dumps({'metadata': {'workflow_kind': 'md_to_kmc_chain', 'barrier_source_mode': 'lammps-neb'}, 'events': []}),
                encoding='utf-8',
            )
            (chain_dir / 'artifacts' / 'md' / 'md_execution.log').write_text('md chain log\n', encoding='utf-8')
            (chain_dir / 'artifacts' / 'kmc' / 'log.spparks').write_text('kmc chain log\nMPI_ABORT\n', encoding='utf-8')

            os.utime(chain_dir, (100, 100))
            os.utime(kmc_only_dir, (200, 200))

            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': False}):
                reply = session.handle_line(
                    '请检查最新这个 MD→KMC run 是否真的正常结束。不要只看 completed 状态；还要检查 LAMMPS 执行记录、KMC 执行记录和相关日志。如果有异常痕迹，请直接指出。'
                )
            self.assertIn('lammps_kmc_demo_4_diagnose', reply)
            self.assertIn('temperature: 810.0 K', reply)
            self.assertIn('md_execution.log', reply)
            self.assertIn('log.spparks', reply)
            self.assertNotIn('latest_kmc_only', reply)
            self.assertNotIn('830.0 K', reply)

    def test_forced_log_target_prefers_md_when_md_step_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            outputs = [
                (
                    mock.Mock(action='inspect', params={}),
                    'Run detail\n- workflow: md_only\n- steps:\n  - md.run: failed\n',
                )
            ]
            self.assertEqual(session._forced_log_target('帮我看看为什么失败了。', outputs), 'md')

    def test_bridge_runner_reads_summary_from_helper_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            helper = tmp / 'fake_bridge.py'
            helper.write_text(
                "\n".join(
                    [
                        "import argparse",
                        "import json",
                        "from pathlib import Path",
                        "",
                        "parser = argparse.ArgumentParser()",
                        "parser.add_argument('--event-json')",
                        "parser.add_argument('--workdir')",
                        "parser.add_argument('--neb-txt')",
                        "parser.add_argument('--barrier')",
                        "parser.add_argument('--validate', action='store_true')",
                        "args = parser.parse_args()",
                        "",
                        "workdir = Path(args.workdir)",
                        "workdir.mkdir(parents=True, exist_ok=True)",
                        "(workdir / 'summary.json').write_text(json.dumps({",
                        "    'status': 'validated',",
                        "    'barrier_eV': 0.42,",
                        "    'files': {",
                        "        'barriers_tsv': str(workdir / 'barriers.tsv'),",
                        "        'state_values_sites': str(workdir / 'state.values.sites'),",
                        "        'input_ml': str(workdir / 'input.ml'),",
                        "        'run_out': str(workdir / 'run.out'),",
                        "    },",
                        "    'validation': {'lookup_hits': 2, 'live_ml_misses': 0},",
                        "    'validation_passed': True,",
                        "}), encoding='utf-8')",
                        "(workdir / 'run.out').write_text('Loop time of 0.1 on 1 procs\\\\n', encoding='utf-8')",
                    ]
                ),
                encoding='utf-8',
            )
            workdir = tmp / 'workdir'
            with mock.patch.dict(
                'os.environ',
                {
                    'MIETCLAW_KMC_BRIDGE_SCRIPT': str(helper),
	                    'MIETCLAW_KMC_BRIDGE_PYTHON': sys.executable,
	                },
	                clear=False,
	            ):
	                summary = run_kmc_lookup_bridge(
	                    event_json=str(tmp / 'event.json'),
	                    neb_txt=str(tmp / 'neb.txt'),
	                    workdir=str(workdir),
	                    validate=True,
	                )
            self.assertTrue(summary['validation_passed'])
            self.assertTrue(summary['safe_validation_passed'])
            self.assertEqual(summary['runtime_health']['status'], 'ok')

    def test_status_command_reports_shell_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / 'demo_run'
            run_dir.mkdir()
            (run_dir / 'state.json').write_text(json.dumps({'steps': {}}), encoding='utf-8')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            healthy = {
                'healthy': True,
                'default_model': 'demo-model',
                'models': ['demo-model'],
                'base_url': 'http://127.0.0.1:8000',
            }
            with mock.patch.object(session, '_refresh_local_model_status', return_value=healthy):
                reply = session.handle_line('/status')
            self.assertIn('Shell status', reply)
            self.assertIn('built-in tools', reply)

    def test_shell_command_registry_drives_help_and_dispatch_handlers(self):
        self.assertEqual(missing_shell_command_handlers(), [])
        self.assertEqual(canonical_shell_command('/quit'), '/exit')
        self.assertIn('/status', shell_command_names())
        self.assertIn('/status', SHELL_COMMAND_HANDLERS)
        self.assertEqual(SHELL_COMMANDS, shell_command_summaries())
        self.assertIn('/model [model-id]', [item['command'] for item in SHELL_COMMANDS])

    def test_chat_session_can_load_history_from_web_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            session.load_history(
                [
                    {'role': 'user', 'content': '  第一条消息  '},
                    {'role': 'assistant', 'content': '  第一条回复  '},
                    {'role': 'system', 'content': '忽略我'},
                    {'role': 'user', 'content': ''},
                ]
            )
            self.assertEqual(session.history, [('user', '第一条消息'), ('assistant', '第一条回复')])

    def test_run_chat_once_payload_reports_tool_result_for_moire_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            case_dir = tmp / 'case'
            case_dir.mkdir()
            summary_json = tmp / 'workdir' / 'summary.json'
            summary_json.parent.mkdir(parents=True, exist_ok=True)
            fake_summary = {
                'status': 'completed',
                'event_json': str(tmp / 'event.json'),
                'source_case_dir': str(case_dir),
                'copied_case_dir': '/tmp/work/lammps_case',
                'generated_lammps_input': '/tmp/work/lammps_case/generated_in.neb.mietclaw',
                'generated_barrier_script': '/tmp/work/lammps_case/extract_barrier.mietclaw.sh',
                'neb_txt': '/tmp/work/lammps_case/neb.txt',
                'barrier_eV': 0.51,
                'runtime_health': {'status': 'ok', 'warnings': [], 'checks': {'returncode_ok': True}},
                'summary_json': str(summary_json),
                'kmc': {
                    'status': 'completed',
                    'barrier_eV': 0.51,
                    'files': {
                        'state_values_sites': '/tmp/work/kmc_bridge/state.repo.values.sites',
                        'input_kmc': '/tmp/work/kmc_bridge/generated_kmc.repo.in',
                        'run_out': '/tmp/work/kmc_bridge/run.out',
                    },
                    'state_transform': {'converted_pair_markers': 24, 'pair_marker_host_type': 1},
                    'barrier_assignment': {'Mo': 0.51, 'Re': 0.51},
                    'parsed_run': {'accepted_events': 9, 'final_time': 1e-10},
                    'runtime_health': {'status': 'ok', 'warnings': [], 'checks': {'returncode_ok': True}},
                },
            }
            summary_json.write_text(json.dumps(fake_summary), encoding='utf-8')

            with mock.patch.dict('os.environ', {'MIETCLAW_MOIRE_CASE_DIR': str(case_dir)}, clear=False):
                with mock.patch.object(
                    MietClawChatSession,
                    '_call_local_mcp_tool',
                    return_value={'structuredContent': fake_summary},
                ):
                    payload = run_chat_once_payload(
                        project_root=str(ROOT),
                        workspace_root=tmpdir,
                        output_dir=tmpdir,
                        provider='local',
                        prompt='我希望你把 MoRe的迁移能垒计算出来并把迁移能垒作为kmc软件的输入进行kmc模拟',
                        history_messages=[{'role': 'assistant', 'content': '之前的上下文'}],
                    )

            self.assertEqual(payload['kind'], 'tool')
            self.assertIn('MoRe LAMMPS → KMC complete', payload['reply'])
            self.assertTrue(payload['used_tools'])
            self.assertEqual(payload['session']['approval_policy'], 'allow_all')
            self.assertEqual(payload['current']['active_kind'], 'moire_summary')
            self.assertTrue(payload['current_moire_summary'])
            self.assertTrue(payload['current']['moire_summary'])
            self.assertTrue(Path(payload['current_moire_summary']['summary_json']).exists())
            self.assertEqual(payload['message']['role'], 'assistant')
            self.assertEqual(payload['message']['kind'], 'tool')
            self.assertEqual(payload['message']['session']['approvalPolicy'], 'allow_all')
            self.assertEqual(payload['message']['current']['activeKind'], 'moire_summary')
            self.assertTrue(payload['message']['currentMoireSummary'])
            self.assertFalse(payload['message']['previewConfirmable'])
            self.assertFalse(payload['message']['previewExecuted'])
            self.assertIn('runtime_snapshot', [card['type'] for card in payload['message']['cards']])

    def test_runtime_chat_payload_enriches_report_and_run_cards(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = tmp / 'demo-run'
            run_dir.mkdir()
            (run_dir / 'summary.json').write_text(json.dumps({'status': 'completed'}), encoding='utf-8')

            job_spec = {
                'job_id': 'demo-job',
                'mode': 'kmc_only',
                'kmc': {
                    'temperature_k': 773.0,
                    'template': {'precomputed_barriers': 'barriers.tsv'},
                },
            }
            job_spec_path = tmp / 'job_spec.generated.json'
            job_spec_path.write_text(json.dumps(job_spec), encoding='utf-8')
            notes_path = tmp / 'notes.md'
            notes_path.write_text('# Notes\nThis run was prepared from a generated report.\n', encoding='utf-8')

            report = {
                'job_id': 'demo-job',
                'mode': 'kmc_only',
                'generated_files': {
                    'job_spec': str(job_spec_path),
                    'notes': str(notes_path),
                },
                'job_spec': job_spec,
            }

            trace_state = ToolTurnState(budget=ToolBudget(max_steps=4, max_mutating_steps=2, max_failures=1))
            request = ToolRequestBlock(
                request_id='inspect-latest',
                intent=ToolIntent(action='inspect', params={'run': 'latest'}),
                source='legacy_router_model',
            )
            trace_state.trace.add(
                AssistantActionBlockEvent(
                    block=AssistantActionBlock(
                        source='legacy_router_model',
                        raw_content='{"action":"inspect"}',
                        tool_requests=[request],
                    )
                )
            )
            trace_state.trace.add(
                ToolResultBlockEvent(
                    block=ToolResultBlock(
                        request_id='inspect-latest',
                        intent=ToolIntent(action='inspect', params={'run': 'latest'}),
                        output='Run detail\n- status: completed',
                        ok=True,
                        source='legacy_router',
                    )
                )
            )
            trace_state.trace.add(TurnFinishEvent(status='finish', reason='tool event loop produced a final answer', reply='done'))

            payload = build_runtime_chat_payload(
                reply='done',
                progress_lines=['[progress] demo'],
                transcript_path=tmp / 'chat.md',
                selected_model='demo-model',
                history_length=4,
                current_run_dir=run_dir,
                current_report=report,
                current_bridge_summary=None,
                current_moire_summary=None,
                current_moire_compare_summary=None,
                current_moire_diffusion_summary=None,
                last_tool_turn_state=trace_state,
            )

            self.assertEqual(payload['message']['kind'], 'tool')
            self.assertEqual(payload['message']['current']['activeKind'], 'run')
            self.assertEqual(payload['message']['currentPlan'][0]['id'], 'kmc.prepare_input')
            self.assertIn('This run was prepared', payload['message']['currentReportNotes'])
            self.assertEqual(payload['message']['currentRunDetail']['runDir'], str(run_dir.resolve()))
            self.assertEqual(payload['message']['currentRunDetail']['status'], 'Completed')
            self.assertEqual(payload['message']['toolTraceSummary']['toolStepCount'], 1)
            self.assertEqual(payload['message']['toolTraceSummary']['finishStatus'], 'finish')
            self.assertTrue(payload['message']['toolTraceId'].startswith('trace-'))
            self.assertEqual(payload['message']['toolEvidence'][0]['action'], 'inspect')
            self.assertEqual(payload['message']['toolTraceReplay'][0]['kind'], 'assistant_action_block')
            self.assertEqual(payload['message']['toolTraceReplay'][-1]['kind'], 'turn_finish')
            self.assertEqual(payload['message']['toolTimeline'][0]['stage'], 'assistant')
            self.assertEqual(payload['message']['toolTimeline'][-1]['stage'], 'finish')
            self.assertEqual(payload['message']['toolTimeline'][0]['transcriptRef']['eventIndex'], 1)
            transparency_card = payload['message']['cards'][0]
            timeline_card = payload['message']['cards'][1]
            runtime_card = payload['message']['cards'][2]
            self.assertEqual(
                [card['type'] for card in payload['message']['cards']],
                ['transparency', 'tool_timeline', 'runtime_snapshot', 'run_result'],
            )
            self.assertEqual(transparency_card['toolTraceSummary']['toolStepCount'], 1)
            self.assertEqual(transparency_card['toolTraceReplay'][0]['kind'], 'assistant_action_block')
            self.assertEqual(timeline_card['timeline'][0]['stage'], 'assistant')
            self.assertEqual(runtime_card['toolEvidence'][0]['action'], 'inspect')
            self.assertEqual(runtime_card['toolTraceReplay'][-1]['kind'], 'turn_finish')

    def test_collect_runtime_doctor_can_use_mocked_probes(self):
        fake_local_status = {'healthy': True, 'default_model': 'demo-model'}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fake_conda = tmp / 'conda'
            fake_kmc = tmp / 'misa-kmc'
            fake_case = tmp / 'moire-case'
            fake_eam = tmp / 'MoRe.eam.fs'
            fake_bridge = tmp / 'run_lammps_kmc_bridge.py'
            fake_conda.write_text('#!/bin/sh\n', encoding='utf-8')
            fake_kmc.write_text('#!/bin/sh\n', encoding='utf-8')
            fake_case.mkdir()
            fake_eam.write_text('', encoding='utf-8')
            fake_bridge.write_text('', encoding='utf-8')
            env = {
                'MIETCLAW_CONDA_EXEC': str(fake_conda),
                'MIETCLAW_MOIRE_KMC_BINARY': str(fake_kmc),
                'MIETCLAW_MOIRE_CASE_DIR': str(fake_case),
                'MIETCLAW_MOIRE_EAM_FILE': str(fake_eam),
                'MIETCLAW_KMC_BRIDGE_SCRIPT': str(fake_bridge),
            }
            with mock.patch.dict('os.environ', env, clear=False):
                with mock.patch('miet_claw.shell_runtime._run_probe') as mocked_probe:
                    mocked_probe.side_effect = [
                        {'ok': True, 'output': '/env/bin/lmp'},
                        {'ok': True, 'output': '/env/bin/mpirun'},
                    ]
                    payload = collect_runtime_doctor(ROOT, local_status=fake_local_status)
        self.assertIn('checks', payload)
        self.assertIn('probes', payload)
        self.assertTrue(payload['checks']['kmc_binary_exists'])

    def test_moire_runtime_runs_lammps_then_bridge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            case_dir = tmp / 'case'
            case_dir.mkdir()
            self._write_minimal_moire_case(case_dir)
            event_json = tmp / 'event.json'
            event_json.write_text(
                json.dumps(
                    {
                        'pair_type': 'MoRe',
                        're_concentration': 0.5,
                        'box_lo': [0.0, 0.0, 0.0],
                        'box_hi': [1.0, 1.0, 1.0],
                        'initsite': {'site_id': 1, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                        'jumpsite': {'site_id': 2, 'atom_type': 2, 'x': 0.5, 'y': 0.5, 'z': 0.5},
                        'normal_sites': [],
                        'other_pair_sites': [],
                    }
                ),
                encoding='utf-8',
            )
            fake_conda = tmp / 'fake-conda'
            fake_conda.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
            fake_conda.chmod(0o755)

            fake_bridge_summary = {
                'barrier_eV': 0.42,
                'files': {
                    'state_values_sites': str(tmp / 'workdir' / 'kmc_bridge' / 'state.repo.values.sites'),
                    'input_kmc': str(tmp / 'workdir' / 'kmc_bridge' / 'generated_kmc.repo.in'),
                    'run_out': str(tmp / 'workdir' / 'kmc_bridge' / 'run.out'),
                },
                'parsed_run': {'accepted_events': 9, 'final_time': 1e-10},
                'runtime_health': {'status': 'ok', 'warnings': [], 'checks': {'returncode_ok': True}},
            }

            subprocess_results = [
                mock.Mock(returncode=0, stdout='lammps ok'),
                mock.Mock(returncode=0, stdout='post ok'),
            ]

            def fake_run(*args, **kwargs):
                cmd = args[0]
                cwd = Path(kwargs['cwd'])
                result = subprocess_results.pop(0)
                if cmd[:4] == [str(fake_conda), 'run', '-n', 'miet-stack']:
                    return result
                if cmd == ['bash', 'extract_barrier.mietclaw.sh']:
                    (cwd / 'neb.txt').write_text('#reaction_crodinate de\n0 0\n1 0.42\n', encoding='utf-8')
                    return result
                return result

            with mock.patch('miet_claw.moire_runtime.subprocess.run', side_effect=fake_run):
                with mock.patch('miet_claw.moire_runtime.run_moire_repo_kmc', return_value=fake_bridge_summary) as mocked_bridge:
                    summary = run_moire_lammps_to_kmc(
                        event_json=str(event_json),
                        case_dir=str(case_dir),
                        workdir=str(tmp / 'workdir'),
                        validate=True,
                        conda_exec=fake_conda,
                    )

            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['kmc']['barrier_eV'], 0.42)
            self.assertTrue(summary['lammps_event_binding']['matches_requested_event'])
            self.assertEqual(summary['lammps_event_binding']['expected_pair']['vacancy']['site_id'], 1)
            self.assertEqual(summary['lammps_event_binding']['expected_pair']['jump']['site_id'], 2)
            self.assertEqual(summary['lammps_model']['mode'], 'generated_from_event_json')
            self.assertTrue(str(mocked_bridge.call_args.kwargs['data_lmp']).endswith('source_data.lmp'))
            self.assertTrue((tmp / 'workdir' / 'summary.json').exists())
            self.assertEqual(summary['kmc']['parsed_run']['accepted_events'], 9)
            self.assertTrue((tmp / 'workdir' / 'lammps_case' / 'ovito_initial.xyz').exists())
            self.assertTrue((tmp / 'workdir' / 'lammps_case' / 'ovito_final.xyz').exists())

    def test_run_moire_event_compare_collects_barriers_and_kmc_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            case_dir = tmp / 'case'
            case_dir.mkdir()
            event_a = tmp / 'event_a.json'
            event_b = tmp / 'event_b.json'
            event_a.write_text('{}', encoding='utf-8')
            event_b.write_text('{}', encoding='utf-8')

            def fake_chain(*, event_json, case_dir, workdir, **kwargs):
                event_path = Path(event_json)
                run_dir = Path(workdir)
                run_dir.mkdir(parents=True, exist_ok=True)
                barrier = 0.31 if event_path.name == 'event_a.json' else 0.58
                accepted = 5 if event_path.name == 'event_a.json' else 9
                final_time = 1e-10 if event_path.name == 'event_a.json' else 2e-10
                summary = {
                    'status': 'completed',
                    'event_json': str(event_path),
                    'source_case_dir': str(case_dir),
                    'copied_case_dir': str(run_dir / 'lammps_case'),
                    'generated_lammps_input': str(run_dir / 'lammps_case' / 'generated_in.neb.mietclaw'),
                    'generated_barrier_script': str(run_dir / 'lammps_case' / 'extract_barrier.mietclaw.sh'),
                    'neb_txt': str(run_dir / 'lammps_case' / 'neb.txt'),
                    'barrier_eV': barrier,
                    'runtime_health': {'status': 'ok', 'warnings': [], 'checks': {'returncode_ok': True}},
                    'summary_json': str(run_dir / 'summary.json'),
                    'lammps_model': {'mode': 'generated_from_event_json'},
                    'lammps_visualization': {
                        'status': 'completed',
                        'initial_snapshot': str(run_dir / 'lammps_case' / 'ovito_initial.png'),
                        'final_snapshot': str(run_dir / 'lammps_case' / 'ovito_final.png'),
                    },
                    'lammps_stage': {'summary_json': str(run_dir / 'lammps_summary.json')},
                    'kmc': {
                        'status': 'completed',
                        'summary_json': str(run_dir / 'kmc_bridge' / 'kmc_summary.json'),
                        'parsed_run': {
                            'accepted_events': accepted,
                            'rejected_events': 0,
                            'final_time': final_time,
                            'final_energy': -10.0 - barrier,
                            'loop_time_seconds': 0.001 + barrier,
                        },
                        'ensemble': {'count': 2, 'completed_count': 2, 'seeds': [3401, 3402]},
                        'visualization': {
                            'status': 'completed',
                            'per_seed': [{'output_png': str(run_dir / 'kmc_bridge' / 'seed_3401' / 'ovito_snapshot.png')}],
                        },
                        'runtime_health': {'status': 'ok', 'warnings': [], 'checks': {'returncode_ok': True}},
                    },
                }
                Path(summary['summary_json']).write_text(json.dumps(summary), encoding='utf-8')
                return summary

            with mock.patch('miet_claw.moire_runtime.run_moire_lammps_to_kmc', side_effect=fake_chain):
                summary = run_moire_event_compare(
                    case_dir=str(case_dir),
                    event_jsons=[str(event_a), str(event_b)],
                    workdir=str(tmp / 'compare'),
                    kmc_seeds=[3401, 3402],
                    run_kmc=True,
                    render_ovito=True,
                )

            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['mode'], 'moire_event_compare')
            self.assertEqual(summary['event_count'], 2)
            self.assertEqual(summary['completed_count'], 2)
            self.assertEqual(summary['barrier_ranking'][0]['label'], 'event_a')
            self.assertAlmostEqual(summary['barrier_span_eV'], 0.27)
            self.assertEqual(summary['kmc_metrics']['accepted_events']['count'], 2)
            self.assertTrue((tmp / 'compare' / 'summary.json').exists())
            self.assertTrue((tmp / 'compare' / 'comparison.json').exists())

    def test_moire_lammps_case_rebuilds_dynamic_assets_from_event_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            case_dir = tmp / 'case'
            case_dir.mkdir()
            self._write_minimal_moire_case(case_dir)
            event_a = tmp / 'event_a.json'
            event_a.write_text(
                json.dumps(
                    {
                        'pair_type': 'MoRe',
                        're_concentration': 0.5,
                        'box_lo': [0.0, 0.0, 0.0],
                        'box_hi': [1.0, 1.0, 1.0],
                        'initsite': {'site_id': 1, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                        'jumpsite': {'site_id': 2, 'atom_type': 2, 'x': 0.5, 'y': 0.5, 'z': 0.5},
                        'normal_sites': [],
                        'other_pair_sites': [],
                    }
                ),
                encoding='utf-8',
            )
            event_b = tmp / 'event_b.json'
            event_b.write_text(
                json.dumps(
                    {
                        'pair_type': 'MoRe',
                        're_concentration': 0.5,
                        'box_lo': [0.0, 0.0, 0.0],
                        'box_hi': [1.0, 1.0, 1.0],
                        'initsite': {'site_id': 2, 'x': 0.5, 'y': 0.5, 'z': 0.5},
                        'jumpsite': {'site_id': 1, 'atom_type': 1, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                        'normal_sites': [],
                        'other_pair_sites': [],
                    }
                ),
                encoding='utf-8',
            )
            fake_conda = tmp / 'fake-conda'
            fake_conda.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
            fake_conda.chmod(0o755)

            def run_case(event_path: Path, workdir_name: str):
                subprocess_results = [
                    mock.Mock(returncode=0, stdout='lammps ok'),
                    mock.Mock(returncode=0, stdout='post ok'),
                ]

                def fake_run(*args, **kwargs):
                    cmd = args[0]
                    cwd = Path(kwargs['cwd'])
                    result = subprocess_results.pop(0)
                    if cmd[:4] == [str(fake_conda), 'run', '-n', 'miet-stack']:
                        return result
                    if cmd == ['bash', 'extract_barrier.mietclaw.sh']:
                        (cwd / 'neb.txt').write_text('#reaction_crodinate de\n0 0\n1 0.25\n', encoding='utf-8')
                        return result
                    return result

                with mock.patch('miet_claw.moire_runtime.subprocess.run', side_effect=fake_run):
                    return run_moire_lammps_case(
                        case_dir=str(case_dir),
                        workdir=str(tmp / workdir_name),
                        event_json=str(event_path),
                        conda_exec=fake_conda,
                    )

            summary_a = run_case(event_a, 'work_a')
            summary_b = run_case(event_b, 'work_b')

            data_a = (tmp / 'work_a' / 'lammps_case' / 'data.lmp').read_text(encoding='utf-8')
            data_b = (tmp / 'work_b' / 'lammps_case' / 'data.lmp').read_text(encoding='utf-8')
            final_a = (tmp / 'work_a' / 'lammps_case' / 'final.mosia').read_text(encoding='utf-8')
            final_b = (tmp / 'work_b' / 'lammps_case' / 'final.mosia').read_text(encoding='utf-8')
            def parse_atom_ids(data_text: str):
                lines = data_text.splitlines()
                start = lines.index('Atoms')
                atom_ids = []
                for raw in lines[start + 1 :]:
                    line = raw.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5 or not parts[0].isdigit():
                        continue
                    atom_ids.append(int(parts[0]))
                return atom_ids

            atom_ids_a = parse_atom_ids(data_a)
            atom_ids_b = parse_atom_ids(data_b)

            self.assertEqual(summary_a['model']['mode'], 'generated_from_event_json')
            self.assertEqual(summary_b['model']['mode'], 'generated_from_event_json')
            self.assertEqual(summary_a['model']['vacancy_site_id'], 1)
            self.assertEqual(summary_b['model']['vacancy_site_id'], 2)
            self.assertTrue((tmp / 'work_a' / 'lammps_case' / 'source_data.lmp').exists())
            self.assertTrue((tmp / 'work_b' / 'lammps_case' / 'source_data.lmp').exists())
            self.assertTrue(summary_a['kmc_data_lmp_assist'].endswith('source_data.lmp'))
            self.assertTrue(summary_b['kmc_data_lmp_assist'].endswith('source_data.lmp'))
            self.assertNotEqual(data_a, data_b)
            self.assertNotEqual(final_a, final_b)
            self.assertEqual(len(atom_ids_a), len(set(atom_ids_a)))
            self.assertEqual(len(atom_ids_b), len(set(atom_ids_b)))

    def test_moire_lammps_case_parses_barrier_and_writes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            case_dir = tmp / 'case'
            case_dir.mkdir()
            self._write_minimal_moire_case(case_dir)
            fake_conda = tmp / 'fake-conda'
            fake_conda.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
            fake_conda.chmod(0o755)

            subprocess_results = [
                mock.Mock(returncode=0, stdout='lammps ok'),
                mock.Mock(returncode=0, stdout='post ok'),
            ]

            def fake_run(*args, **kwargs):
                cmd = args[0]
                cwd = Path(kwargs['cwd'])
                result = subprocess_results.pop(0)
                if cmd[:4] == [str(fake_conda), 'run', '-n', 'miet-stack']:
                    return result
                if cmd == ['bash', 'extract_barrier.mietclaw.sh']:
                    (cwd / 'neb.txt').write_text('#reaction_crodinate de\n0 0\n0.5 0.31\n1 0.27\n', encoding='utf-8')
                    return result
                return result

            with mock.patch('miet_claw.moire_runtime.subprocess.run', side_effect=fake_run):
                summary = run_moire_lammps_case(
                    case_dir=str(case_dir),
                    workdir=str(tmp / 'workdir'),
                    conda_exec=fake_conda,
                )

            self.assertEqual(summary['status'], 'completed')
            self.assertAlmostEqual(summary['barrier_eV'], 0.31)
            self.assertTrue((tmp / 'workdir' / 'lammps_summary.json').exists())
            self.assertTrue((tmp / 'workdir' / 'lammps_case' / 'generated_in.neb.mietclaw').exists())
            self.assertTrue((tmp / 'workdir' / 'lammps_case' / 'extract_barrier.mietclaw.sh').exists())
            self.assertIn('write_dump      all custom neb.initial.dump id type x y z', (tmp / 'workdir' / 'lammps_case' / 'generated_in.neb.mietclaw').read_text(encoding='utf-8'))
            self.assertIn('write_dump      all custom neb.final.dump id type x y z', (tmp / 'workdir' / 'lammps_case' / 'generated_in.neb.mietclaw').read_text(encoding='utf-8'))
            self.assertTrue((tmp / 'workdir' / 'lammps_case' / 'ovito_initial.xyz').exists())
            self.assertTrue((tmp / 'workdir' / 'lammps_case' / 'ovito_final.xyz').exists())

    def test_moire_lammps_case_can_render_ovito_snapshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            case_dir = tmp / 'case'
            case_dir.mkdir()
            self._write_minimal_moire_case(case_dir)
            fake_conda = tmp / 'fake-conda'
            fake_conda.write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
            fake_conda.chmod(0o755)

            subprocess_results = [
                mock.Mock(returncode=0, stdout='lammps ok'),
                mock.Mock(returncode=0, stdout='post ok'),
            ]

            def fake_run(*args, **kwargs):
                cmd = args[0]
                cwd = Path(kwargs['cwd'])
                result = subprocess_results.pop(0)
                if cmd[:4] == [str(fake_conda), 'run', '-n', 'miet-stack']:
                    return result
                if cmd == ['bash', 'extract_barrier.mietclaw.sh']:
                    (cwd / 'neb.txt').write_text('#reaction_crodinate de\n0 0\n0.5 0.31\n1 0.27\n', encoding='utf-8')
                    return result
                return result

            def fake_render(path, output, python_cmd):
                output.write_text(f'rendered from {path.name}', encoding='utf-8')
                return {'returncode': 0, 'output': ''}

            with mock.patch('miet_claw.moire_runtime.subprocess.run', side_effect=fake_run):
                with mock.patch('miet_claw.moire_runtime._resolve_ovito_python', return_value='python3'):
                    with mock.patch('miet_claw.moire_runtime._render_ovito_snapshot', side_effect=fake_render):
                        summary = run_moire_lammps_case(
                            case_dir=str(case_dir),
                            workdir=str(tmp / 'workdir'),
                            conda_exec=fake_conda,
                            render_ovito=True,
                        )

            self.assertEqual(summary['visualization']['status'], 'completed')
            self.assertTrue((tmp / 'workdir' / 'lammps_case' / 'ovito_initial.png').exists())
            self.assertTrue((tmp / 'workdir' / 'lammps_case' / 'ovito_final.png').exists())

    def test_run_moire_repo_kmc_generates_transparent_repo_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            event_json = tmp / 'event.json'
            event_json.write_text(
                json.dumps(
                    {
                        'pair_type': 'MoRe',
                        're_concentration': 0.07,
                        'pair_displacement': 0.3,
                        'box_lo': [0.0, 0.0, 0.0],
                        'box_hi': [2.0, 1.0, 1.0],
                        'initsite': {'site_id': 1, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                        'jumpsite': {'site_id': 2, 'atom_type': 1, 'x': 0.5, 'y': 0.5, 'z': 0.5},
                        'normal_sites': [
                            {'site_id': 2, 'atom_type': 1, 'x': 0.5, 'y': 0.5, 'z': 0.5},
                            {'site_id': 3, 'atom_type': 2, 'x': 1.0, 'y': 0.0, 'z': 0.0},
                        ],
                        'other_pair_sites': [
                            {'site_id': 4, 'pair_type': 'MoRe', 'x': 1.5, 'y': 0.5, 'z': 0.5},
                        ],
                    }
                ),
                encoding='utf-8',
            )
            data_lmp = tmp / 'data.lmp'
            data_lmp.write_text(
                '\n'.join(
                    [
                        'LAMMPS data file',
                        '',
                        '3 atoms',
                        '2 atom types',
                        '',
                        '0.0 2.0 xlo xhi',
                        '0.0 1.0 ylo yhi',
                        '0.0 1.0 zlo zhi',
                        '',
                        'Atoms',
                        '',
                        '1 1 0.5 0.5 0.5',
                        '2 2 1.0 0.0 0.0',
                        '3 2 1.5 0.5 0.5',
                    ]
                )
                + '\n',
                encoding='utf-8',
            )
            fake_kmc = tmp / 'misa-kmc'
            fake_kmc.write_text(
                '#!/bin/bash\n'
                'cat <<\'EOF\'\n'
                '      Time    Naccept    Nreject    Nsweeps        CPU      Energy\tCu_alone\n'
                '         0          0          0          0          0 -10.0\t0\n'
                '     1e-10          9          0          0   0.000601 -11.0\t1\n'
                'Loop time of 0.000603 on 1 procs\n'
                'EOF\n',
                encoding='utf-8',
            )
            fake_kmc.chmod(0o755)
            fake_eam = tmp / 'MoRe.eam.fs'
            fake_eam.write_text('eam', encoding='utf-8')

            summary = run_moire_repo_kmc(
                barrier_eV=0.59798,
                event_json=str(event_json),
                workdir=str(tmp / 'repo_kmc'),
                data_lmp=str(data_lmp),
                misa_kmc_binary=fake_kmc,
                eam_file=fake_eam,
            )

            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['parsed_run']['accepted_events'], 9)
            self.assertEqual(summary['state_transform']['converted_pair_markers'], 1)
            self.assertEqual(summary['state_transform']['pair_sites_from_data_lmp'], 1)
            self.assertEqual(summary['cells'], [2, 1, 1])
            self.assertTrue((tmp / 'repo_kmc' / 'generated_kmc.repo.in').exists())
            state_text = (tmp / 'repo_kmc' / 'state.repo.values.sites').read_text(encoding='utf-8')
            self.assertIn('1 0 1', state_text)
            self.assertIn('4 2 4', state_text)

    def test_run_moire_repo_kmc_can_generate_seed_event_from_data_lmp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_lmp = tmp / 'data.lmp'
            data_lmp.write_text(
                '\n'.join(
                    [
                        'LAMMPS data file',
                        '',
                        '2 atoms',
                        '2 atom types',
                        '',
                        '0.0 1.0 xlo xhi',
                        '0.0 1.0 ylo yhi',
                        '0.0 1.0 zlo zhi',
                        '',
                        'Atoms',
                        '',
                        '1 1 0.0 0.0 0.0',
                        '2 1 0.5 0.5 0.5',
                    ]
                )
                + '\n',
                encoding='utf-8',
            )
            fake_kmc = tmp / 'misa-kmc'
            fake_kmc.write_text(
                '#!/bin/bash\n'
                'cat <<\'EOF\'\n'
                '      Time    Naccept    Nreject    Nsweeps        CPU      Energy\tCu_alone\n'
                '         0          0          0          0          0 -10.0\t0\n'
                '     1e-10          5          0          0   0.000401 -11.0\t1\n'
                'Loop time of 0.000403 on 1 procs\n'
                'EOF\n',
                encoding='utf-8',
            )
            fake_kmc.chmod(0o755)
            fake_eam = tmp / 'MoRe.eam.fs'
            fake_eam.write_text('eam', encoding='utf-8')

            summary = run_moire_repo_kmc(
                barrier_eV=0.42,
                workdir=str(tmp / 'repo_kmc'),
                data_lmp=str(data_lmp),
                misa_kmc_binary=fake_kmc,
                eam_file=fake_eam,
            )

            self.assertEqual(summary['status'], 'completed')
            self.assertIsNotNone(summary['generated_event'])
            self.assertEqual(summary['generated_event']['source'], 'generated_from_data_lmp')
            self.assertTrue((tmp / 'repo_kmc' / 'generated_seed_event.json').exists())
            self.assertEqual(summary['state_generation']['source'], 'event_json')

    def test_run_moire_repo_kmc_flags_abnormal_output_markers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_lmp = tmp / 'data.lmp'
            data_lmp.write_text(
                '\n'.join(
                    [
                        'LAMMPS data file',
                        '',
                        '2 atoms',
                        '2 atom types',
                        '',
                        '0.0 1.0 xlo xhi',
                        '0.0 1.0 ylo yhi',
                        '0.0 1.0 zlo zhi',
                        '',
                        'Atoms',
                        '',
                        '1 1 0.0 0.0 0.0',
                        '2 1 0.5 0.5 0.5',
                    ]
                )
                + '\n',
                encoding='utf-8',
            )
            fake_kmc = tmp / 'misa-kmc'
            fake_kmc.write_text(
                '#!/bin/bash\n'
                'cat <<\'EOF\'\n'
                '      Time    Naccept    Nreject    Nsweeps        CPU      Energy\tCu_alone\n'
                '         0          0          0          0          0 -10.0\t0\n'
                '     1e-10          5          0          0   0.000401 -11.0\t1\n'
                'ERROR: simulated neighbor list failure\n'
                'Loop time of 0.000403 on 1 procs\n'
                'EOF\n',
                encoding='utf-8',
            )
            fake_kmc.chmod(0o755)
            fake_eam = tmp / 'MoRe.eam.fs'
            fake_eam.write_text('eam', encoding='utf-8')
            workdir = tmp / 'repo_kmc'

            with self.assertRaises(MoReWorkflowError):
                run_moire_repo_kmc(
                    barrier_eV=0.42,
                    workdir=str(workdir),
                    data_lmp=str(data_lmp),
                    misa_kmc_binary=fake_kmc,
                    eam_file=fake_eam,
                )

            summary = json.loads((workdir / 'kmc_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['runtime_health']['status'], 'failed')
            self.assertTrue(summary['runtime_health']['checks']['abnormal_output_detected'])
            self.assertIn('ERROR', summary['runtime_health']['checks']['abnormal_markers'])

    def test_run_moire_repo_kmc_can_auto_retry_with_next_seed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_lmp = tmp / 'data.lmp'
            data_lmp.write_text(
                '\n'.join(
                    [
                        'LAMMPS data file',
                        '',
                        '2 atoms',
                        '2 atom types',
                        '',
                        '0.0 1.0 xlo xhi',
                        '0.0 1.0 ylo yhi',
                        '0.0 1.0 zlo zhi',
                        '',
                        'Atoms',
                        '',
                        '1 1 0.0 0.0 0.0',
                        '2 1 0.5 0.5 0.5',
                    ]
                )
                + '\n',
                encoding='utf-8',
            )
            fake_kmc = tmp / 'misa-kmc'
            fake_kmc.write_text(
                '#!/bin/bash\n'
                'if grep -q "seed                 3401" generated_kmc.repo.in; then\n'
                '  cat <<\'EOF\'\n'
                '      Time    Naccept    Nreject    Nsweeps        CPU      Energy\tCu_alone\n'
                '     1e-10          1          0          0   0.000401 -11.0\t1\n'
                'ERROR: first seed failed\n'
                'Loop time of 0.000403 on 1 procs\n'
                'EOF\n'
                'else\n'
                '  cat <<\'EOF\'\n'
                '      Time    Naccept    Nreject    Nsweeps        CPU      Energy\tCu_alone\n'
                '     1e-10          7          0          0   0.000401 -11.0\t1\n'
                'Loop time of 0.000403 on 1 procs\n'
                'EOF\n'
                'fi\n',
                encoding='utf-8',
            )
            fake_kmc.chmod(0o755)
            fake_eam = tmp / 'MoRe.eam.fs'
            fake_eam.write_text('eam', encoding='utf-8')

            summary = run_moire_repo_kmc(
                barrier_eV=0.42,
                workdir=str(tmp / 'repo_kmc'),
                data_lmp=str(data_lmp),
                misa_kmc_binary=fake_kmc,
                eam_file=fake_eam,
                retry_attempts=1,
            )

            self.assertEqual(summary['status'], 'warning')
            self.assertEqual(summary['representative_seed'], 3402)
            self.assertEqual(summary['auto_retry']['added_retry_seeds'], [3402])
            self.assertEqual(summary['ensemble']['completed_count'], 1)
            self.assertEqual(summary['parsed_run']['accepted_events'], 7)

    def test_run_moire_repo_kmc_can_repeat_multiple_seeds_and_collect_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            event_json = tmp / 'event.json'
            event_json.write_text(
                json.dumps(
                    {
                        'pair_type': 'MoRe',
                        're_concentration': 0.07,
                        'pair_displacement': 0.3,
                        'box_lo': [0.0, 0.0, 0.0],
                        'box_hi': [2.0, 1.0, 1.0],
                        'initsite': {'site_id': 1, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                        'jumpsite': {'site_id': 2, 'atom_type': 1, 'x': 0.5, 'y': 0.5, 'z': 0.5},
                        'normal_sites': [
                            {'site_id': 2, 'atom_type': 1, 'x': 0.5, 'y': 0.5, 'z': 0.5},
                            {'site_id': 3, 'atom_type': 2, 'x': 1.0, 'y': 0.0, 'z': 0.0},
                        ],
                        'other_pair_sites': [
                            {'site_id': 4, 'pair_type': 'MoRe', 'x': 1.5, 'y': 0.5, 'z': 0.5},
                        ],
                    }
                ),
                encoding='utf-8',
            )
            data_lmp = tmp / 'data.lmp'
            data_lmp.write_text(
                '\n'.join(
                    [
                        'LAMMPS data file',
                        '',
                        '3 atoms',
                        '2 atom types',
                        '',
                        '0.0 2.0 xlo xhi',
                        '0.0 1.0 ylo yhi',
                        '0.0 1.0 zlo zhi',
                        '',
                        'Atoms',
                        '',
                        '1 1 0.5 0.5 0.5',
                        '2 2 1.0 0.0 0.0',
                        '3 2 1.5 0.5 0.5',
                    ]
                )
                + '\n',
                encoding='utf-8',
            )
            fake_kmc = tmp / 'misa-kmc'
            fake_kmc.write_text(
                '#!/bin/bash\n'
                'cat <<\'EOF\'\n'
                '      Time    Naccept    Nreject    Nsweeps        CPU      Energy\tCu_alone\n'
                '         0          0          0          0          0 -10.0\t0\n'
                '     1e-10          9          0          0   0.000601 -11.0\t1\n'
                'Loop time of 0.000603 on 1 procs\n'
                'EOF\n',
                encoding='utf-8',
            )
            fake_kmc.chmod(0o755)
            fake_eam = tmp / 'MoRe.eam.fs'
            fake_eam.write_text('eam', encoding='utf-8')

            def fake_ovito(*, run_dir, seed, enabled, python_cmd):
                image = Path(run_dir) / 'ovito_snapshot.png'
                image.write_text(f'ovito seed {seed}', encoding='utf-8')
                return {
                    'requested': True,
                    'status': 'completed',
                    'seed': seed,
                    'dump_file': str(Path(run_dir) / '1.dump'),
                    'output_png': str(image),
                }

            def fake_gif(image_paths, output_path, *, python_cmd, duration_ms=800):
                output_path.write_text('gif', encoding='utf-8')
                return {'returncode': 0, 'output': ''}

            with mock.patch('miet_claw.moire_runtime._resolve_ovito_python', return_value='python3'):
                with mock.patch('miet_claw.moire_runtime._resolve_pillow_python', return_value='python3'):
                    with mock.patch('miet_claw.moire_runtime._render_image_sequence_gif', side_effect=fake_gif):
                        with mock.patch('miet_claw.moire_runtime._maybe_render_seed_ovito', side_effect=fake_ovito):
                            summary = run_moire_repo_kmc(
                                barrier_eV=0.59798,
                                event_json=str(event_json),
                                workdir=str(tmp / 'repo_kmc'),
                                data_lmp=str(data_lmp),
                                misa_kmc_binary=fake_kmc,
                                eam_file=fake_eam,
                                kmc_seeds=[4101, 4102],
                                render_ovito=True,
                            )

            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['ensemble']['count'], 2)
            self.assertEqual(summary['ensemble']['completed_count'], 2)
            self.assertEqual(summary['ensemble']['seeds'], [4101, 4102])
            self.assertEqual(summary['representative_seed'], 4101)
            self.assertEqual(summary['visualization']['status'], 'completed')
            self.assertEqual(summary['visualization']['completed_count'], 2)
            self.assertEqual(summary['visualization']['gif_status'], 'completed')
            self.assertTrue(summary['visualization']['comparison_chart_svg'].endswith('kmc_seed_comparison.svg'))
            self.assertTrue(summary['visualization']['animated_gif'].endswith('kmc_seed_animation.gif'))
            self.assertTrue((tmp / 'repo_kmc' / 'seed_4101' / 'generated_kmc.repo.in').exists())
            self.assertTrue((tmp / 'repo_kmc' / 'seed_4102' / 'generated_kmc.repo.in').exists())
            self.assertIn('seed                 4101', (tmp / 'repo_kmc' / 'seed_4101' / 'generated_kmc.repo.in').read_text(encoding='utf-8'))
            self.assertIn('seed                 4102', (tmp / 'repo_kmc' / 'seed_4102' / 'generated_kmc.repo.in').read_text(encoding='utf-8'))
            self.assertTrue((tmp / 'repo_kmc' / 'seed_4101' / 'ovito_snapshot.png').exists())
            self.assertTrue((tmp / 'repo_kmc' / 'seed_4102' / 'ovito_snapshot.png').exists())
            self.assertTrue((tmp / 'repo_kmc' / 'kmc_seed_comparison.svg').exists())
            self.assertTrue((tmp / 'repo_kmc' / 'kmc_seed_animation.gif').exists())

    def test_run_moire_repo_kmc_reconstructs_vacancy_diffusion_from_dumps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            event_json = tmp / 'event.json'
            event_json.write_text(
                json.dumps(
                    {
                        'pair_type': 'MoRe',
                        're_concentration': 0.07,
                        'pair_displacement': 0.3,
                        'box_lo': [0.0, 0.0, 0.0],
                        'box_hi': [1.0, 1.0, 1.0],
                        'initsite': {'site_id': 1, 'x': 0.0, 'y': 0.0, 'z': 0.0},
                        'jumpsite': {'site_id': 2, 'atom_type': 1, 'x': 0.5, 'y': 0.5, 'z': 0.5},
                        'normal_sites': [
                            {'site_id': 2, 'atom_type': 1, 'x': 0.5, 'y': 0.5, 'z': 0.5},
                        ],
                        'other_pair_sites': [],
                    }
                ),
                encoding='utf-8',
            )
            data_lmp = tmp / 'data.lmp'
            data_lmp.write_text(
                '\n'.join(
                    [
                        'LAMMPS data file',
                        '',
                        '1 atoms',
                        '2 atom types',
                        '',
                        '0.0 1.0 xlo xhi',
                        '0.0 1.0 ylo yhi',
                        '0.0 1.0 zlo zhi',
                        '',
                        'Atoms',
                        '',
                        '1 1 0.5 0.5 0.5',
                    ]
                )
                + '\n',
                encoding='utf-8',
            )
            fake_kmc = tmp / 'misa-kmc'
            fake_kmc.write_text(
                "#!/bin/bash\n"
                "cat <<'EOF'\n"
                "      Time    Naccept    Nreject    Nsweeps        CPU      Energy\tCu_alone\n"
                "         0          0          0          0          0 -10.0\t0\n"
                "     1e-10          6          0          0   0.000601 -11.0\t1\n"
                "Loop time of 0.000603 on 1 procs\n"
                "EOF\n"
                "cat > 0.dump <<'EOF'\n"
                "ITEM: TIMESTEP\n"
                "0 0.0\n"
                "ITEM: NUMBER OF ATOMS\n"
                "2\n"
                "ITEM: BOX BOUNDS pp pp pp\n"
                "0 1\n"
                "0 1\n"
                "0 1\n"
                "ITEM: ATOMS type i2 x y z\n"
                "0 1 0.0 0.0 0.0\n"
                "1 2 0.5 0.5 0.5\n"
                "EOF\n"
                "cat > 1.dump <<'EOF'\n"
                "ITEM: TIMESTEP\n"
                "1 1e-10\n"
                "ITEM: NUMBER OF ATOMS\n"
                "2\n"
                "ITEM: BOX BOUNDS pp pp pp\n"
                "0 1\n"
                "0 1\n"
                "0 1\n"
                "ITEM: ATOMS type i2 x y z\n"
                "0 2 0.5 0.5 0.5\n"
                "1 1 0.0 0.0 0.0\n"
                "EOF\n",
                encoding='utf-8',
            )
            fake_kmc.chmod(0o755)
            fake_eam = tmp / 'MoRe.eam.fs'
            fake_eam.write_text('eam', encoding='utf-8')

            summary = run_moire_repo_kmc(
                barrier_eV=0.45,
                event_json=str(event_json),
                workdir=str(tmp / 'repo_kmc'),
                data_lmp=str(data_lmp),
                misa_kmc_binary=fake_kmc,
                eam_file=fake_eam,
            )

            self.assertEqual(summary['status'], 'completed')
            self.assertTrue((tmp / 'repo_kmc' / 'vacancy_diffusion.csv').exists())
            self.assertEqual(summary['diffusion_analysis']['status'], 'completed')
            self.assertEqual(summary['diffusion_analysis']['final_vacancy_site_id'], 2)
            self.assertAlmostEqual(summary['diffusion_analysis']['final_msd'], 0.75)
            self.assertAlmostEqual(summary['diffusion_analysis']['final_diffusion_coefficient'], 0.75 / (6.0e-10))
            self.assertAlmostEqual(summary['derived_metrics']['jump_frequency_hz'], 6.0e10)

    def test_run_moire_diffusion_sweep_writes_temperature_summary_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            case_dir = tmp / 'case'
            case_dir.mkdir()
            event_json = tmp / 'event.json'
            event_json.write_text('{}', encoding='utf-8')

            fake_lammps_summary = {
                'status': 'completed',
                'barrier_eV': 0.454657,
                'summary_json': str(tmp / 'sweep' / 'lammps_summary.json'),
                'kmc_data_lmp_assist': str(tmp / 'sweep' / 'lammps_case' / 'source_data.lmp'),
                'visualization': {
                    'status': 'completed',
                    'final_snapshot': str(tmp / 'sweep' / 'lammps_case' / 'ovito_final.png'),
                },
            }

            def fake_kmc(*, temperature, workdir, kmc_seeds, **kwargs):
                run_dir = Path(workdir)
                run_dir.mkdir(parents=True, exist_ok=True)
                coeff = float(temperature) * 1.0e-12
                summary = {
                    'status': 'completed',
                    'summary_json': str(run_dir / 'kmc_summary.json'),
                    'representative_seed': kmc_seeds[0],
                    'ensemble': {
                        'completed_count': len(kmc_seeds),
                        'metrics': {
                            'jump_frequency_hz': {'mean': float(temperature) * 1.0e6},
                            'accepted_events': {'mean': float(temperature) / 10.0},
                        },
                    },
                    'diffusion_ensemble': {'mean': coeff, 'std': coeff * 0.1},
                    'files': {
                        'ovito_snapshot': str(run_dir / 'seed_3401' / 'ovito_snapshot.png'),
                        'vacancy_diffusion_csv': str(run_dir / 'seed_3401' / 'vacancy_diffusion.csv'),
                    },
                }
                Path(summary['summary_json']).write_text(json.dumps(summary), encoding='utf-8')
                return summary

            with mock.patch('miet_claw.moire_runtime.run_moire_lammps_case', return_value=fake_lammps_summary):
                with mock.patch('miet_claw.moire_runtime.run_moire_repo_kmc', side_effect=fake_kmc):
                    summary = run_moire_diffusion_sweep(
                        event_json=str(event_json),
                        case_dir=str(case_dir),
                        workdir=str(tmp / 'sweep'),
                        temperatures_k=[800.0, 900.0, 1000.0],
                        kmc_seeds=[3401, 3402],
                        render_ovito=True,
                    )

            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['mode'], 'moire_diffusion_sweep')
            self.assertEqual(summary['completed_count'], 3)
            self.assertTrue(summary['temperature_trend']['monotonic_increasing'])
            self.assertTrue((tmp / 'sweep' / 'diffusion_vs_temperature.csv').exists())
            self.assertTrue((tmp / 'sweep' / 'diffusion_vs_temperature.svg').exists())
            self.assertTrue((tmp / 'sweep' / 'arrhenius.csv').exists())
            self.assertTrue((tmp / 'sweep' / 'arrhenius.svg').exists())
            self.assertTrue((tmp / 'sweep' / 'summary.json').exists())

    def test_chat_moire_workflow_uses_internal_mcp_chain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            event_json = tmp / 'event.json'
            event_json.write_text('{}', encoding='utf-8')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            fake_summary = {
                'status': 'completed',
                'event_json': str(event_json),
                'source_case_dir': '/tmp/case',
                'copied_case_dir': '/tmp/work/lammps_case',
                'generated_lammps_input': '/tmp/work/lammps_case/generated_in.neb.mietclaw',
                'generated_barrier_script': '/tmp/work/lammps_case/extract_barrier.mietclaw.sh',
                'neb_txt': '/tmp/work/lammps_case/neb.txt',
                'barrier_eV': 0.51,
                'runtime_health': {'status': 'ok', 'warnings': [], 'checks': {'returncode_ok': True}},
                'summary_json': str(tmp / 'workdir' / 'summary.json'),
                'kmc': {
                    'status': 'completed',
                    'barrier_eV': 0.51,
                    'files': {
                        'state_values_sites': '/tmp/work/kmc_bridge/state.repo.values.sites',
                        'input_kmc': '/tmp/work/kmc_bridge/generated_kmc.repo.in',
                        'run_out': '/tmp/work/kmc_bridge/run.out',
                    },
                    'state_transform': {'converted_pair_markers': 24, 'pair_marker_host_type': 1},
                    'barrier_assignment': {'Mo': 0.51, 'Re': 0.51},
                    'parsed_run': {'accepted_events': 9, 'final_time': 1e-10},
                    'runtime_health': {'status': 'ok', 'warnings': [], 'checks': {'returncode_ok': True}},
                },
            }

            with mock.patch.object(
                session,
                '_call_local_mcp_tool',
                return_value={'structuredContent': fake_summary},
            ):
                reply = session._run_moire_workflow(
                    event_json=str(event_json),
                    case_dir='/tmp/case',
                    workdir=str(tmp / 'workdir'),
                    validate=True,
                )

            self.assertIn('LAMMPS dispatch: local stdio MCP (miet_moire_run)', reply)
            self.assertIn('KMC dispatch: local stdio MCP (miet_moire_run)', reply)
            self.assertEqual(session.current_moire_summary['dispatch']['lammps']['tool'], 'miet_moire_run')
            self.assertEqual(session.current_moire_summary['dispatch']['kmc']['tool'], 'miet_moire_run')

    def test_chat_moire_compare_uses_internal_mcp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            event_a = tmp / 'event_a.json'
            event_b = tmp / 'event_b.json'
            event_a.write_text('{}', encoding='utf-8')
            event_b.write_text('{}', encoding='utf-8')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            fake_compare_summary = {
                'status': 'completed',
                'mode': 'moire_event_compare',
                'case_dir': '/tmp/case',
                'workdir': str(tmp / 'compare'),
                'event_count': 2,
                'completed_count': 2,
                'run_kmc': True,
                'ovito_requested': True,
                'event_runs': [
                    {'label': 'event_a', 'status': 'completed', 'barrier_eV': 0.31, 'summary_json': '/tmp/compare/event_a/summary.json'},
                    {'label': 'event_b', 'status': 'completed', 'barrier_eV': 0.58, 'summary_json': '/tmp/compare/event_b/summary.json'},
                ],
                'barrier_ranking': [
                    {'rank': 1, 'label': 'event_a', 'barrier_eV': 0.31, 'delta_vs_lowest_eV': 0.0},
                    {'rank': 2, 'label': 'event_b', 'barrier_eV': 0.58, 'delta_vs_lowest_eV': 0.27},
                ],
                'summary_json': str(tmp / 'compare' / 'summary.json'),
                'comparison_json': str(tmp / 'compare' / 'comparison.json'),
            }

            with mock.patch.object(
                session,
                '_call_local_mcp_tool',
                return_value={'structuredContent': fake_compare_summary},
            ) as mocked_mcp:
                reply = session._run_moire_compare_workflow(
                    case_dir='/tmp/case',
                    event_jsons=[str(event_a), str(event_b)],
                    workdir=str(tmp / 'compare'),
                    validate=True,
                    kmc_seeds=[3401, 3402],
                    ovito=True,
                )

            self.assertIn('MoRe event compare complete', reply)
            self.assertEqual(mocked_mcp.call_args.args[0], 'miet_moire_compare')
            self.assertEqual(session.current_moire_compare_summary['dispatch']['tool'], 'miet_moire_compare')

    def test_chat_moire_diffusion_sweep_uses_internal_mcp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            event_json = tmp / 'event.json'
            event_json.write_text('{}', encoding='utf-8')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            fake_summary = {
                'status': 'completed',
                'mode': 'moire_diffusion_sweep',
                'case_dir': '/tmp/case',
                'event_json': str(event_json),
                'workdir': str(tmp / 'diffusion'),
                'barrier_eV': 0.454657,
                'temperatures_k': [800.0, 900.0, 1000.0],
                'kmc_seeds': [3401, 3402],
                'run_time': '1e-6',
                'stats_step': '1e-7',
                'ovito_requested': True,
                'temperature_runs': [
                    {'label': '800 K', 'status': 'completed', 'diffusion_coefficient': 8.0e-10, 'kmc_summary_json': '/tmp/diffusion/T_800K/kmc_summary.json'},
                    {'label': '900 K', 'status': 'completed', 'diffusion_coefficient': 9.0e-10, 'kmc_summary_json': '/tmp/diffusion/T_900K/kmc_summary.json'},
                ],
                'completed_count': 2,
                'arrhenius_fit': {'activation_energy_eV': 0.45},
                'temperature_trend': {'monotonic_increasing': True},
                'files': {
                    'diffusion_vs_temperature_svg': '/tmp/diffusion/diffusion_vs_temperature.svg',
                    'arrhenius_svg': '/tmp/diffusion/arrhenius.svg',
                },
                'summary_json': str(tmp / 'diffusion' / 'summary.json'),
            }

            with mock.patch.object(
                session,
                '_call_local_mcp_tool',
                return_value={'structuredContent': fake_summary},
            ) as mocked_mcp:
                reply = session._run_moire_diffusion_workflow(
                    event_json=str(event_json),
                    case_dir='/tmp/case',
                    workdir=str(tmp / 'diffusion'),
                    validate=True,
                    temperatures_k=[800.0, 900.0, 1000.0],
                    kmc_seeds=[3401, 3402],
                    ovito=True,
                )

            self.assertIn('MoRe diffusion-vs-temperature sweep complete', reply)
            self.assertEqual(mocked_mcp.call_args.args[0], 'miet_moire_diffusion_sweep')
            self.assertEqual(session.current_moire_diffusion_summary['dispatch']['tool'], 'miet_moire_diffusion_sweep')

    def test_parser_rejects_unknown_tool_actions(self):
        self.assertIsNone(parse_tool_intent('{"action":"not_real","reply":"x"}'))
        self.assertIsNone(
            parse_agent_decision(
                '{"status":"continue","step":{"action":"not_real","params":{"run":"latest"}}}'
            )
        )

    def test_chat_session_reuses_one_local_mcp_client(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            events = []

            class FakeClient:
                def __init__(self, **kwargs):
                    events.append(('init', kwargs['provider']))
                    self.closed = False

                def connect(self):
                    events.append(('connect', None))
                    return self

                def call_tool(self, tool_name, arguments):
                    events.append(('call', tool_name, arguments))
                    return {'structuredContent': {'tool': tool_name, 'arguments': arguments}}

                def close(self):
                    self.closed = True
                    events.append(('close', None))

            with mock.patch('miet_claw.chat.LocalMCPClient', FakeClient):
                first = session._call_local_mcp_tool('miet_list_runs', {'limit': 1}, 'first')
                second = session._call_local_mcp_tool('miet_list_runs', {'limit': 2}, 'second')
                session.close()

            self.assertEqual(first['structuredContent']['arguments']['limit'], 1)
            self.assertEqual(second['structuredContent']['arguments']['limit'], 2)
            self.assertEqual([item[0] for item in events].count('init'), 1)
            self.assertEqual([item[0] for item in events].count('call'), 2)
            self.assertIn(('close', None), events)

    def test_read_only_tool_result_is_cached_within_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = self._make_synthetic_run(tmp, 'cached_inspect_case')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            fake_report = {
                'job_id': run_dir.name,
                'path': str(run_dir),
                'mode': 'md_to_kmc_chain',
                'material_name': 'cached_inspect_case',
                'temperature_k': 800.0,
                'step_statuses': {'kmc.run': 'completed'},
                'barrier_source_mode': 'lammps-neb',
                'workflow_kind': 'md_to_kmc_chain',
                'neb_images': None,
                'events': [],
                'latest_diffusion': None,
                'summary': 'done',
                'summary_path': None,
                'artifacts': [],
                'md_log_path': str(run_dir / 'artifacts' / 'md' / 'md_execution.log'),
                'kmc_log_path': str(run_dir / 'artifacts' / 'kmc' / 'log.spparks'),
            }

            with mock.patch('miet_claw.chat.inspect_run', return_value=fake_report) as inspect_mock:
                first = session._execute_tool_intent(ToolIntent(action='inspect', params={'run': 'latest'}), 'inspect latest')
                second = session._execute_tool_intent(ToolIntent(action='inspect', params={'run': 'latest'}), 'inspect latest')

            self.assertEqual(first, second)
            self.assertEqual(inspect_mock.call_count, 1)

    def test_mutating_tool_action_invalidates_read_only_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = self._make_synthetic_run(tmp, 'invalidate_cache_case')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            fake_report = {
                'job_id': run_dir.name,
                'path': str(run_dir),
                'mode': 'md_to_kmc_chain',
                'material_name': 'invalidate_cache_case',
                'temperature_k': 800.0,
                'step_statuses': {'kmc.run': 'completed'},
                'barrier_source_mode': 'lammps-neb',
                'workflow_kind': 'md_to_kmc_chain',
                'neb_images': None,
                'events': [],
                'latest_diffusion': None,
                'summary': 'done',
                'summary_path': None,
                'artifacts': [],
                'md_log_path': str(run_dir / 'artifacts' / 'md' / 'md_execution.log'),
                'kmc_log_path': str(run_dir / 'artifacts' / 'kmc' / 'log.spparks'),
            }

            with mock.patch('miet_claw.chat.inspect_run', return_value=fake_report) as inspect_mock:
                session._execute_tool_intent(ToolIntent(action='inspect', params={'run': 'latest'}), 'inspect latest')
                with mock.patch.object(session, '_run_draft', return_value='drafted'):
                    session._execute_tool_intent(ToolIntent(action='draft', params={'prompt': 'draft something'}), 'draft something')
                session._execute_tool_intent(ToolIntent(action='inspect', params={'run': 'latest'}), 'inspect latest')

            self.assertEqual(inspect_mock.call_count, 2)

    def test_tool_execution_failure_returns_safe_outcome(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            with mock.patch.object(session, '_run_draft', side_effect=RuntimeError('boom')):
                outcome = session._execute_tool_intent_outcome(
                    ToolIntent(action='draft', params={'prompt': 'draft something'}),
                    'draft something',
                )

            self.assertFalse(outcome.ok)
            self.assertIn('工具 draft 执行失败：boom', outcome.output)

    def test_tool_plan_stops_after_failure_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            plan = ToolPlan(
                steps=[
                    ToolIntent(action='inspect', params={'run': 'latest'}),
                    ToolIntent(action='logs', params={'run': 'latest'}),
                ],
                summarize=True,
            )
            with mock.patch.dict('os.environ', {'MIETCLAW_TOOL_MAX_FAILURES': '1'}, clear=False):
                with mock.patch.object(session, '_summarize_tool_outputs', return_value='stopped after failure') as summarize_mock:
                    reply = session._execute_tool_plan(plan, '检查最新 run')

            self.assertEqual(reply, 'stopped after failure')
            summarize_mock.assert_called_once()
            outputs = summarize_mock.call_args.args[1]
            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0][0].action, 'inspect')
            self.assertIn('已经出现 1 次失败或无效结果', summarize_mock.call_args.kwargs['note'])

    def test_tool_plan_reuses_existing_turn_evidence_and_skips_duplicate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            existing_intent = ToolIntent(action='inspect', params={'run': 'latest'})
            turn = session._new_tool_turn_state(default_steps=4)
            turn.outputs.append((existing_intent, 'existing inspect output'))
            turn.seen_signatures.add(session._intent_signature(existing_intent))
            plan = ToolPlan(
                steps=[
                    ToolIntent(action='inspect', params={'run': 'latest'}),
                    ToolIntent(action='logs', params={'run': 'latest'}),
                ],
                summarize=True,
            )
            with mock.patch.object(
                session,
                '_execute_tool_intent_outcome',
                return_value=ToolExecutionOutcome(output='logs output', ok=True),
            ) as execute_mock:
                with mock.patch.object(session, '_summarize_tool_outputs', return_value='summary with shared turn') as summarize_mock:
                    reply = session._execute_tool_plan(plan, '检查最新 run', state=turn)

            self.assertEqual(reply, 'summary with shared turn')
            self.assertEqual(execute_mock.call_count, 1)
            executed_intent = execute_mock.call_args.args[0]
            self.assertEqual(executed_intent.action, 'logs')
            self.assertIn('已跳过 1 个重复工具步骤', summarize_mock.call_args.kwargs['note'])

    def test_agent_loop_stops_when_model_repeats_same_tool_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            prompt = '请判断最新 run 是否正常，如果有必要再看日志。'
            with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': True}):
                with mock.patch.object(session, '_heuristic_agent_first_step', return_value=None):
                    with mock.patch.object(
                        session,
                        '_execute_tool_intent_outcome',
                        return_value=ToolExecutionOutcome(output='inspect output', ok=True),
                    ) as execute_mock:
                        with mock.patch.object(
                            session,
                            '_summarize_tool_outputs',
                            return_value='summary from existing evidence',
                        ) as summarize_mock:
                            with mock.patch(
                                'miet_claw.chat.chat_with_local_model',
                                side_effect=[
                                    {
                                        'content': '{"status":"continue","step":{"action":"inspect","params":{"run":"latest"}}}'
                                    },
                                    {
                                        'content': '{"status":"continue","step":{"action":"inspect","params":{"run":"latest"}}}'
                                    },
                                ],
                            ):
                                reply = session._run_agent_loop(prompt)

            self.assertEqual(reply, 'summary from existing evidence')
            self.assertEqual(execute_mock.call_count, 1)
            summarize_mock.assert_called_once()

    def test_agent_loop_stops_when_mutating_budget_is_exhausted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            prompt = '请判断这个 run 为什么失败，如果有必要的话先帮我起草一个修复 workflow，再直接运行。'
            with mock.patch.dict(
                'os.environ',
                {
                    'MIETCLAW_TOOL_MAX_MUTATIONS': '1',
                    'MIETCLAW_AGENT_MAX_STEPS': '4',
                },
                clear=False,
            ):
                with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': True}):
                    with mock.patch.object(session, '_heuristic_agent_first_step', return_value=None):
                        with mock.patch.object(
                            session,
                            '_execute_tool_intent_outcome',
                            return_value=ToolExecutionOutcome(output='draft ok', ok=True),
                        ) as execute_mock:
                            with mock.patch.object(
                                session,
                                '_summarize_tool_outputs',
                                return_value='stopped by mutating budget',
                            ) as summarize_mock:
                                    with mock.patch(
                                    'miet_claw.chat.chat_with_local_model',
                                    side_effect=[
                                        {
                                            'content': '{"status":"continue","step":{"action":"draft","params":{"prompt":"x"}}}'
                                        },
                                        {
                                            'content': '{"status":"continue","step":{"action":"run","params":{"prompt":"x"}}}'
                                        },
                                    ],
                                ):
                                        reply = session._run_agent_loop(prompt)

            self.assertEqual(reply, 'stopped by mutating budget')
            self.assertEqual(execute_mock.call_count, 1)
            summarize_mock.assert_called_once()
            self.assertIn('最多允许 1 个会改动状态的工具动作', summarize_mock.call_args.kwargs['note'])

    def test_confirm_mutations_policy_blocks_agent_mutation_but_allows_manual_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            prompt = '请直接起草一个 workflow。'
            with mock.patch.dict(
                'os.environ',
                {'MIETCLAW_TOOL_APPROVAL_POLICY': 'confirm_mutations'},
                clear=False,
            ):
                with mock.patch('miet_claw.chat.should_try_agent_loop', return_value=True):
                    with mock.patch('miet_claw.runtime.query_engine.should_try_agent_loop', return_value=True):
                        with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': True}):
                            with mock.patch.object(session, '_heuristic_agent_first_step', return_value=None):
                                with mock.patch.object(session, '_execute_tool_intent_outcome') as execute_mock:
                                    with mock.patch(
                                        'miet_claw.chat.chat_with_local_model',
                                        return_value={'content': '{"status":"continue","step":{"action":"draft","params":{"prompt":"x"}}}'},
                                    ):
                                        reply = session._run_agent_loop(prompt)

                self.assertIsNotNone(reply)
                self.assertIn('显式确认', reply)
                self.assertIn('/draft', reply)
                execute_mock.assert_not_called()

                with mock.patch.object(session, '_run_draft', return_value='draft ok') as draft_mock:
                    manual_reply = session._handle_command('/draft hello')

            self.assertTrue(manual_reply.startswith('draft ok'))
            self.assertIn('如果草案方向对', manual_reply)
            draft_mock.assert_called_once()

    def test_read_only_policy_denies_manual_mutation_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            with mock.patch.dict(
                'os.environ',
                {'MIETCLAW_TOOL_APPROVAL_POLICY': 'read_only'},
                clear=False,
            ):
                with mock.patch.object(session, '_run_draft', return_value='draft ok') as draft_mock:
                    reply = session._handle_command('/draft hello')

            self.assertIn('只读模式', reply)
            draft_mock.assert_not_called()

    def test_handle_line_uses_query_engine_line_handler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            with mock.patch.object(engine, 'handle_line', return_value='line handled') as handle_line_mock:
                reply = session.handle_line('你好')

            self.assertEqual(reply, 'line handled')
            handle_line_mock.assert_called_once_with('你好')

    def test_handle_prompt_uses_query_engine_prompt_turn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            result = QueryTurnResult(turn_id='turn-handle-prompt', reply='tool turn result', used_tools=True, status='tool')
            with mock.patch.object(engine, 'handle_prompt_turn', return_value=result) as handle_prompt_turn_mock:
                with mock.patch.object(session, '_refresh_local_model_status') as status_mock:
                    reply = session._handle_prompt('检查一下最新 run')

            self.assertEqual(reply, 'tool turn result')
            handle_prompt_turn_mock.assert_called_once_with('检查一下最新 run')
            status_mock.assert_not_called()

    def test_manual_inspect_command_uses_shared_cache_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = self._make_synthetic_run(tmp, 'manual_inspect_cached_case')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            fake_report = {
                'job_id': run_dir.name,
                'path': str(run_dir),
                'mode': 'md_to_kmc_chain',
                'material_name': 'manual_inspect_cached_case',
                'temperature_k': 800.0,
                'step_statuses': {'kmc.run': 'completed'},
                'barrier_source_mode': 'lammps-neb',
                'workflow_kind': 'md_to_kmc_chain',
                'neb_images': None,
                'events': [],
                'latest_diffusion': None,
                'summary': 'done',
                'summary_path': None,
                'artifacts': [],
                'md_log_path': str(run_dir / 'artifacts' / 'md' / 'md_execution.log'),
                'kmc_log_path': str(run_dir / 'artifacts' / 'kmc' / 'log.spparks'),
            }

            with mock.patch('miet_claw.chat.inspect_run', return_value=fake_report) as inspect_mock:
                first = session._handle_command('/inspect latest')
                second = session._handle_command('/inspect latest')

            self.assertEqual(first, second)
            self.assertEqual(inspect_mock.call_count, 1)

    def test_manual_tool_command_populates_context_for_followup_turns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_synthetic_run(tmp, 'manual_context_case')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )

            session._handle_command('/inspect latest')
            messages = session._build_tool_router_messages('再看下日志')
            context = json.loads(messages[1]['content'])

            self.assertIn('completed_tool_steps', context)
            self.assertEqual(context['completed_tool_steps'][-1]['action'], 'inspect')
            self.assertEqual(context['completed_tool_steps'][-1]['params']['run'], 'latest')

    def test_plain_chat_fallback_receives_recent_tool_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = self._make_synthetic_run(tmp, 'plain_chat_tool_context_case')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            fake_report = {
                'job_id': run_dir.name,
                'path': str(run_dir),
                'mode': 'md_to_kmc_chain',
                'material_name': 'plain_chat_tool_context_case',
                'temperature_k': 800.0,
                'step_statuses': {'kmc.run': 'completed'},
                'barrier_source_mode': 'lammps-neb',
                'workflow_kind': 'md_to_kmc_chain',
                'neb_images': None,
                'events': [],
                'latest_diffusion': None,
                'summary': 'done',
                'summary_path': None,
                'artifacts': [],
                'md_log_path': str(run_dir / 'artifacts' / 'md' / 'md_execution.log'),
                'kmc_log_path': str(run_dir / 'artifacts' / 'kmc' / 'log.spparks'),
            }
            healthy = {
                'healthy': True,
                'default_model': 'demo-model',
                'models': ['demo-model'],
                'base_url': 'http://127.0.0.1:8000',
            }

            with mock.patch('miet_claw.chat.inspect_run', return_value=fake_report):
                session._handle_command('/inspect latest')

            with mock.patch.object(session, '_run_tool_turn', return_value=None):
                with mock.patch.object(session, '_refresh_local_model_status', return_value=healthy):
                    with mock.patch('miet_claw.chat.chat_with_local_model', return_value={'content': 'plain chat answer'}) as chat_mock:
                        reply = session._handle_prompt('你总结一下刚才看到了什么')

            self.assertEqual(reply, 'plain chat answer')
            messages = chat_mock.call_args.args[0]
            evidence_messages = [
                item for item in messages
                if item['role'] == 'system' and 'authoritative session context' in item['content']
            ]
            self.assertTrue(evidence_messages)
            self.assertIn('action=inspect', evidence_messages[-1]['content'])
            self.assertIn('plain_chat_tool_context_case', evidence_messages[-1]['content'])

    def test_render_plan_outputs_uses_structured_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            rendered = session._render_plan_outputs(
                '帮我判断这个 run 是否正常',
                [(ToolIntent(action='inspect', params={'run': 'latest'}), 'inspect output')],
                note='已有足够证据',
            )

            self.assertIn('结论', rendered)
            self.assertIn('证据', rendered)
            self.assertIn('下一步', rendered)
            self.assertIn('已有足够证据', rendered)

    def test_render_plan_outputs_lists_answered_and_deferred_goals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_synthetic_run(tmp, 'render_optional_new')
            self._make_synthetic_run(tmp, 'render_optional_old')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            rendered = session._render_plan_outputs(
                '帮我找出为什么失败了，顺便比较最近两次 run 的差异',
                [
                    (ToolIntent(action='inspect', params={'run': 'latest'}), 'Run detail\n- steps:\n  - kmc.run: failed\n'),
                    (ToolIntent(action='logs', params={'run': 'current', 'target': 'kmc'}), 'Log excerpt (kmc)\nMPI_ABORT\n'),
                ],
            )

            self.assertIn('已回答：失败原因/异常判断', rendered)
            self.assertIn('暂未展开：run 差异', rendered)
            self.assertIn('可直接继续问', rendered)
            self.assertIn('那再帮我比较最近两次 run 的差异。', rendered)

    def test_tool_summary_fallback_is_structured_when_model_unavailable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': False}):
                reply = session._summarize_tool_outputs(
                    '帮我总结',
                    [(ToolIntent(action='logs', params={'run': 'latest'}), 'log output')],
                    note='本轮已停止继续扩展工具调用',
                )

            self.assertIn('结论', reply)
            self.assertIn('证据', reply)
            self.assertIn('下一步', reply)
            self.assertIn('本轮已停止继续扩展工具调用', reply)

    def test_response_strategy_requests_logs_for_diagnostic_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            outputs = [
                (
                    ToolIntent(action='inspect', params={'run': 'latest'}),
                    'Run detail\n- steps:\n  - kmc.run: failed\n',
                )
            ]
            strategy = session._response_strategy_for_prompt('帮我看看为什么失败了', outputs)

            self.assertEqual(strategy.status, 'needs_more_evidence')
            self.assertIsNotNone(strategy.followup_intent)
            self.assertEqual(strategy.followup_intent.action, 'logs')
            self.assertEqual(strategy.followup_intent.params['target'], 'kmc')

    def test_run_tool_turn_auto_collects_followup_logs_when_needed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            router_blocks = [
                AssistantActionBlock(
                    source='heuristic_router',
                    tool_requests=[
                        ToolRequestBlock(
                            request_id='inspect-latest',
                            intent=ToolIntent(action='inspect', params={'run': 'latest'}),
                            source='heuristic_router',
                        )
                    ],
                    metadata={'synthetic': True},
                )
            ]
            with mock.patch.object(engine, 'run_agent_loop', return_value=None):
                with mock.patch.object(engine, 'resolve_tool_plan_blocks', return_value=[]):
                    with mock.patch.object(engine, 'resolve_router_blocks', return_value=router_blocks):
                        with mock.patch.object(
                            session,
                            '_execute_tool_intent_outcome',
                            side_effect=[
                                ToolExecutionOutcome(output='Run detail\n- steps:\n  - kmc.run: failed\n', ok=True),
                                ToolExecutionOutcome(output='Log excerpt (kmc)\nMPI_ABORT', ok=True),
                            ],
                        ) as execute_mock:
                            with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': False}):
                                reply = session._run_tool_turn('帮我看看为什么失败了')

            self.assertEqual(execute_mock.call_count, 2)
            self.assertIn('step 2: logs', reply)
            self.assertIn('MPI_ABORT', reply)

    def test_response_strategy_requests_artifacts_for_output_file_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = self._make_synthetic_run(tmp, 'artifact_prompt_case')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            session.current_run_dir = run_dir
            outputs = [
                (
                    ToolIntent(action='inspect', params={'run': 'latest'}),
                    'Run detail\n- path: artifact_prompt_case\n',
                )
            ]
            strategy = session._response_strategy_for_prompt('这个 run 生成了什么文件？', outputs)

            self.assertEqual(strategy.status, 'needs_more_evidence')
            self.assertIsNotNone(strategy.followup_intent)
            self.assertEqual(strategy.followup_intent.action, 'artifacts')
            self.assertEqual(strategy.followup_intent.params['run'], 'current')

    def test_plan_evidence_followups_prefers_direct_compare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_synthetic_run(tmp, 'compare_path_new')
            self._make_synthetic_run(tmp, 'compare_path_old')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )

            path = session._plan_evidence_followups('帮我比较最近两次 run 的差异', [])

            self.assertEqual([intent.action for intent in path], ['compare_runs'])

    def test_plan_evidence_followups_prefers_direct_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = self._make_synthetic_run(tmp, 'artifact_path_case')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            session.current_run_dir = run_dir

            path = session._plan_evidence_followups('这个 run 生成了什么文件？', [])

            self.assertEqual([intent.action for intent in path], ['artifacts'])

    def test_plan_evidence_followups_uses_inspect_then_logs_for_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_synthetic_run(tmp, 'diagnostic_path_case')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )

            path = session._plan_evidence_followups('帮我看看为什么失败了', [])

            self.assertEqual([intent.action for intent in path], ['inspect', 'logs'])

    def test_plan_evidence_followups_prioritizes_diagnostic_before_compare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_synthetic_run(tmp, 'priority_compare_new')
            self._make_synthetic_run(tmp, 'priority_compare_old')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )

            path = session._plan_evidence_followups('帮我比较最近两次 run 的差异，并解释为什么最新这次失败了', [])

            self.assertEqual([intent.action for intent in path], ['inspect', 'logs', 'compare_runs'])

    def test_plan_evidence_followups_respects_explicit_compare_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_synthetic_run(tmp, 'explicit_compare_new')
            self._make_synthetic_run(tmp, 'explicit_compare_old')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )

            path = session._plan_evidence_followups('先比较最近两次 run 的差异，再解释为什么最新这次失败了', [])

            self.assertEqual([intent.action for intent in path], ['compare_runs', 'inspect', 'logs'])

    def test_plan_evidence_followups_skips_optional_compare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_synthetic_run(tmp, 'optional_compare_new')
            self._make_synthetic_run(tmp, 'optional_compare_old')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )

            path = session._plan_evidence_followups('帮我找出为什么失败了，顺便比较最近两次 run 的差异', [])

            self.assertEqual([intent.action for intent in path], ['inspect', 'logs'])

    def test_response_strategy_requests_compare_for_compare_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_synthetic_run(tmp, 'compare_case_new')
            self._make_synthetic_run(tmp, 'compare_case_old')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            outputs = [(ToolIntent(action='runs', params={}), 'Recent runs\n- compare_case_new\n- compare_case_old')]
            strategy = session._response_strategy_for_prompt('帮我比较最近两次 run 的差异', outputs)

            self.assertEqual(strategy.status, 'needs_more_evidence')
            self.assertIsNotNone(strategy.followup_intent)
            self.assertEqual(strategy.followup_intent.action, 'compare_runs')

    def test_response_strategy_stops_after_main_goal_when_compare_is_optional(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_synthetic_run(tmp, 'optional_strategy_new')
            self._make_synthetic_run(tmp, 'optional_strategy_old')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            outputs = [
                (
                    ToolIntent(action='inspect', params={'run': 'latest'}),
                    'Run detail\n- steps:\n  - kmc.run: failed\n- path: optional_strategy_new\n',
                ),
                (
                    ToolIntent(action='logs', params={'run': 'current', 'target': 'kmc'}),
                    'Log excerpt (kmc)\nMPI_ABORT\n',
                ),
            ]

            strategy = session._response_strategy_for_prompt('帮我找出为什么失败了，顺便比较最近两次 run 的差异', outputs)

            self.assertEqual(strategy.status, 'sufficient')
            self.assertIsNone(strategy.followup_intent)
            self.assertIn('主要问题已经回答', strategy.reason)
            self.assertIn('失败原因/异常判断', strategy.answered_goals)
            self.assertIn('run 差异', strategy.deferred_goals)
            self.assertIn('那再帮我比较最近两次 run 的差异。', strategy.followup_prompts)

    def test_response_strategy_uses_mode_aware_compare_followup_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_synthetic_run(tmp, 'mode_optional_new', mode='md_to_kmc_chain')
            self._make_synthetic_run(tmp, 'mode_optional_old', mode='md_to_kmc_chain')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            outputs = [
                (
                    ToolIntent(action='inspect', params={'run': 'latest', 'mode': 'md_to_kmc_chain'}),
                    'Run detail\n- job_id: mode_optional_new\n- mode: md_to_kmc_chain\n- steps:\n  - kmc.run: failed\n',
                ),
                (
                    ToolIntent(action='logs', params={'run': 'current', 'target': 'kmc', 'mode': 'md_to_kmc_chain'}),
                    'Log excerpt (kmc)\nMPI_ABORT\n',
                ),
            ]

            strategy = session._response_strategy_for_prompt('帮我找出这个 MD→KMC run 为什么失败了，顺便比较最近两次 run 的差异', outputs)

            self.assertIn('那再帮我比较最近两次 MD→KMC chain run 的差异。', strategy.followup_prompts)

    def test_response_strategy_uses_run_aware_artifact_followup_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = self._make_synthetic_run(tmp, 'artifact_context_case')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            session.current_run_dir = run_dir
            outputs = [
                (
                    ToolIntent(action='inspect', params={'run': 'latest'}),
                    'Run detail\n- job_id: artifact_context_case\n- steps:\n  - kmc.run: completed\n',
                ),
                (
                    ToolIntent(action='logs', params={'run': 'current', 'target': 'kmc'}),
                    'Log excerpt (kmc)\nall good\n',
                ),
            ]

            strategy = session._response_strategy_for_prompt('帮我判断这个 run 是否正常，顺便列一下它生成了什么文件', outputs)

            self.assertIn('那再帮我列一下 `artifact_context_case` 这个 run 生成的产物文件。', strategy.followup_prompts)

    def test_response_strategy_uses_target_aware_diagnostic_followup_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            outputs = [
                (
                    ToolIntent(action='compare_runs', params={}),
                    'Run comparison\n- newer run: compare_context_new\n- older run: compare_context_old\n',
                )
            ]

            strategy = session._response_strategy_for_prompt('先比较最近两次 run 的差异，如果方便再帮我看一下 KMC 日志找失败原因', outputs)

            self.assertIn('那再继续帮我看一下 `compare_context_new` 的 KMC 日志，找出失败根因。', strategy.followup_prompts)

    def test_run_tool_turn_auto_collects_artifacts_when_output_files_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = self._make_synthetic_run(tmp, 'artifact_followup_case')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            session.current_run_dir = run_dir
            engine = session._get_query_engine()
            router_blocks = [
                AssistantActionBlock(
                    source='heuristic_router',
                    tool_requests=[
                        ToolRequestBlock(
                            request_id='inspect-latest',
                            intent=ToolIntent(action='inspect', params={'run': 'latest'}),
                            source='heuristic_router',
                        )
                    ],
                    metadata={'synthetic': True},
                )
            ]
            with mock.patch.object(engine, 'run_agent_loop', return_value=None):
                with mock.patch.object(engine, 'resolve_tool_plan_blocks', return_value=[]):
                    with mock.patch.object(engine, 'resolve_router_blocks', return_value=router_blocks):
                        with mock.patch.object(
                            session,
                            '_execute_tool_intent_outcome',
                            side_effect=[
                                ToolExecutionOutcome(output='Run detail\n- path: artifact_followup_case\n', ok=True),
                                ToolExecutionOutcome(output='Artifacts\n- run: artifact_followup_case\n- artifacts/kmc/generated_kmc.in', ok=True),
                            ],
                        ) as execute_mock:
                            with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': False}):
                                reply = session._run_tool_turn('这个 run 生成了什么文件？')

            self.assertEqual(execute_mock.call_count, 2)
            self.assertIn('step 2: artifacts', reply)
            self.assertIn('generated_kmc.in', reply)

    def test_run_tool_turn_auto_collects_compare_when_compare_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_synthetic_run(tmp, 'compare_followup_new')
            self._make_synthetic_run(tmp, 'compare_followup_old')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            router_blocks = [
                AssistantActionBlock(
                    source='heuristic_router',
                    tool_requests=[
                        ToolRequestBlock(
                            request_id='runs',
                            intent=ToolIntent(action='runs', params={}),
                            source='heuristic_router',
                        )
                    ],
                    metadata={'synthetic': True},
                )
            ]
            with mock.patch.object(engine, 'run_agent_loop', return_value=None):
                with mock.patch.object(engine, 'resolve_tool_plan_blocks', return_value=[]):
                    with mock.patch.object(engine, 'resolve_router_blocks', return_value=router_blocks):
                        with mock.patch.object(
                            session,
                            '_execute_tool_intent_outcome',
                            side_effect=[
                                ToolExecutionOutcome(output='Recent runs\n- compare_followup_new\n- compare_followup_old', ok=True),
                                ToolExecutionOutcome(output='Run comparison\n- newer run: compare_followup_new', ok=True),
                            ],
                        ) as execute_mock:
                            with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': False}):
                                reply = session._run_tool_turn('帮我比较最近两次 run 的差异')

            self.assertEqual(execute_mock.call_count, 2)
            self.assertIn('step 2: compare_runs', reply)
            self.assertIn('compare_followup_new', reply)

    def test_run_tool_turn_auto_prioritizes_diagnostic_path_before_compare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_synthetic_run(tmp, 'combo_followup_new')
            self._make_synthetic_run(tmp, 'combo_followup_old')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            router_blocks = [
                AssistantActionBlock(
                    source='heuristic_router',
                    tool_requests=[
                        ToolRequestBlock(
                            request_id='inspect-latest',
                            intent=ToolIntent(action='inspect', params={'run': 'latest'}),
                            source='heuristic_router',
                        )
                    ],
                    metadata={'synthetic': True},
                )
            ]
            with mock.patch.object(engine, 'run_agent_loop', return_value=None):
                with mock.patch.object(engine, 'resolve_tool_plan_blocks', return_value=[]):
                    with mock.patch.object(engine, 'resolve_router_blocks', return_value=router_blocks):
                        with mock.patch.object(
                            session,
                            '_execute_tool_intent_outcome',
                            side_effect=[
                                ToolExecutionOutcome(
                                    output='Run detail\n- steps:\n  - kmc.run: failed\n- path: combo_followup_new\n',
                                    ok=True,
                                ),
                                ToolExecutionOutcome(output='Log excerpt (kmc)\nMPI_ABORT', ok=True),
                                ToolExecutionOutcome(
                                    output='Run comparison\n- newer run: combo_followup_new\n- older run: combo_followup_old',
                                    ok=True,
                                ),
                            ],
                        ) as execute_mock:
                            with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': False}):
                                reply = session._run_tool_turn('帮我比较最近两次 run 的差异，并解释为什么最新这次失败了')

            self.assertEqual(execute_mock.call_count, 3)
            self.assertIn('step 1: inspect', reply)
            self.assertIn('step 2: logs', reply)
            self.assertIn('step 3: compare_runs', reply)
            self.assertIn('MPI_ABORT', reply)

    def test_run_tool_turn_stops_before_optional_compare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._make_synthetic_run(tmp, 'optional_turn_new')
            self._make_synthetic_run(tmp, 'optional_turn_old')
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            router_blocks = [
                AssistantActionBlock(
                    source='heuristic_router',
                    tool_requests=[
                        ToolRequestBlock(
                            request_id='inspect-latest',
                            intent=ToolIntent(action='inspect', params={'run': 'latest'}),
                            source='heuristic_router',
                        )
                    ],
                    metadata={'synthetic': True},
                )
            ]
            with mock.patch.object(engine, 'run_agent_loop', return_value=None):
                with mock.patch.object(engine, 'resolve_tool_plan_blocks', return_value=[]):
                    with mock.patch.object(engine, 'resolve_router_blocks', return_value=router_blocks):
                        with mock.patch.object(
                            session,
                            '_execute_tool_intent_outcome',
                            side_effect=[
                                ToolExecutionOutcome(
                                    output='Run detail\n- steps:\n  - kmc.run: failed\n- path: optional_turn_new\n',
                                    ok=True,
                                ),
                                ToolExecutionOutcome(output='Log excerpt (kmc)\nMPI_ABORT', ok=True),
                            ],
                        ) as execute_mock:
                            with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': False}):
                                reply = session._run_tool_turn('帮我找出为什么失败了，顺便比较最近两次 run 的差异')

            self.assertEqual(execute_mock.call_count, 2)
            self.assertIn('step 1: inspect', reply)
            self.assertIn('step 2: logs', reply)
            self.assertNotIn('compare_runs', reply)
            self.assertIn('已回答：失败原因/异常判断', reply)
            self.assertIn('暂未展开：run 差异', reply)
            self.assertIn('可直接继续问', reply)
            self.assertIn('那再帮我比较最近两次 run 的差异。', reply)


    def test_chat_tool_registry_exposes_richer_tool_metadata(self):
        run_tool = get_chat_tool_definition('run')
        self.assertIsNotNone(run_tool)
        self.assertEqual(run_tool.name, 'miet_autonomy_run')
        self.assertTrue(run_tool.mutating)
        self.assertFalse(run_tool.read_only)
        self.assertEqual(run_tool.permission_scope, 'run')

        draft_tool = get_chat_tool_definition('draft')
        self.assertIsNotNone(draft_tool)
        self.assertEqual(draft_tool.permission_scope, 'plan')

        compare_tool = get_chat_tool_definition('compare_runs')
        self.assertIsNotNone(compare_tool)
        self.assertEqual(compare_tool.name, 'miet_compare_runs')
        self.assertTrue(compare_tool.read_only)
        self.assertFalse(compare_tool.expose_mcp)
        self.assertEqual(compare_tool.permission_scope, 'read')

    def test_permission_profile_is_driven_by_tool_registry(self):
        logs_profile = permission_profile_for_intent(ToolIntent(action='logs', params={'run': 'latest'}))
        self.assertTrue(logs_profile.read_only)
        self.assertFalse(logs_profile.mutating)
        self.assertEqual(logs_profile.scope, 'read')

        draft_profile = permission_profile_for_intent(ToolIntent(action='draft', params={'prompt': 'x'}))
        self.assertFalse(draft_profile.read_only)
        self.assertTrue(draft_profile.mutating)
        self.assertEqual(draft_profile.scope, 'plan')

        run_profile = permission_profile_for_intent(ToolIntent(action='run', params={'prompt': 'x'}))
        self.assertFalse(run_profile.read_only)
        self.assertTrue(run_profile.mutating)
        self.assertEqual(run_profile.scope, 'run')
        self.assertEqual(permission_scope_for_intent(ToolIntent(action='run', params={'prompt': 'x'})), 'run')

    def test_chat_tool_dispatch_delegates_through_tool_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            tool = get_chat_tool_definition('run')
            self.assertIsNotNone(tool)
            with mock.patch('miet_claw.runtime.tool_registry.ToolDefinition.execute_chat', autospec=True, return_value=ToolExecutionOutcome(output='delegated run')) as execute_mock:
                outcome = execute_chat_tool_intent_outcome(
                    session,
                    ToolIntent(action='run', params={'prompt': 'hello'}),
                    'hello',
                    api=__import__('miet_claw.chat', fromlist=['format_draft_report']).__dict__,
                )

        self.assertEqual(outcome.output, 'delegated run')
        execute_mock.assert_called_once()
        self.assertIs(execute_mock.call_args.args[0], tool)

    def test_mcp_tool_dispatch_delegates_through_tool_object(self):
        class FakeServer:
            project_root = str(ROOT)
            workspace_root = str(ROOT)
            output_dir = str(ROOT / 'runs')
            provider = 'local'

        tool = get_tool_definition('miet_list_runs')
        self.assertIsNotNone(tool)
        with mock.patch('miet_claw.runtime.tool_registry.ToolDefinition.execute_mcp', autospec=True, return_value={'content': [{'type': 'text', 'text': 'delegated'}], 'isError': False}) as execute_mock:
            result = dispatch_mcp_tool(
                FakeServer(),
                'miet_list_runs',
                {'limit': 1},
                api={'MCPServerError': RuntimeError},
            )

        self.assertEqual(result['content'][0]['text'], 'delegated')
        execute_mock.assert_called_once()
        self.assertIs(execute_mock.call_args.args[0], tool)

    def test_tool_definition_validates_required_chat_params(self):
        tool = get_chat_tool_definition('draft')
        self.assertIsNotNone(tool)
        outcome = tool.execute_chat(
            None,
            ToolIntent(action='draft', params={}),
            'hello',
            handlers={},
            api={},
        )
        self.assertFalse(outcome.ok)
        self.assertIn('缺少必需参数 `prompt`', outcome.output)

    def test_tool_definition_renders_empty_chat_output(self):
        tool = get_chat_tool_definition('runs')
        self.assertIsNotNone(tool)
        rendered = tool.render_chat_result(
            ToolIntent(action='runs', params={}),
            ToolExecutionOutcome(output='', ok=True),
        )
        self.assertTrue(rendered.ok)
        self.assertIn('没有返回可展示文本', rendered.output)

    def test_execution_tool_result_strategy_appends_resume_followup_hint(self):
        tool = get_chat_tool_definition('run')
        self.assertIsNotNone(tool)
        rendered = tool.render_chat_result(
            ToolIntent(action='run', params={'prompt': 'hello', 'resume_existing': True}),
            ToolExecutionOutcome(
                output='Run complete',
                ok=True,
                metadata={
                    'execution': {
                        'resume_existing': True,
                        'final_recovery': {
                            'resume_summary': {'completed_steps': ['md.run']},
                        },
                    }
                },
            ),
        )
        self.assertIn('如果你想确认恢复后的状态', rendered.output)

    def test_execution_tool_result_strategy_mentions_restarted_steps(self):
        tool = get_chat_tool_definition('run')
        self.assertIsNotNone(tool)
        rendered = tool.render_chat_result(
            ToolIntent(action='run', params={'prompt': 'hello', 'resume_existing': True}),
            ToolExecutionOutcome(
                output='Run complete',
                ok=True,
                metadata={
                    'execution': {
                        'resume_existing': True,
                        'final_recovery': {
                            'recovery_plan': {
                                'steps': [
                                    {'step_id': 'md.run', 'action': 'reuse_completed'},
                                    {'step_id': 'kmc.run', 'action': 'restart_resumable_step'},
                                ]
                            }
                        },
                    }
                },
            ),
        )
        self.assertIn('kmc.run', rendered.output)

    def test_execution_tool_result_strategy_mentions_missing_outputs(self):
        tool = get_chat_tool_definition('run')
        self.assertIsNotNone(tool)
        rendered = tool.render_chat_result(
            ToolIntent(action='run', params={'prompt': 'hello', 'resume_existing': True}),
            ToolExecutionOutcome(
                output='Run complete',
                ok=True,
                metadata={
                    'execution': {
                        'resume_existing': True,
                        'final_recovery': {
                            'recovery_plan': {
                                'steps': [
                                    {
                                        'step_id': 'kmc.prepare_input',
                                        'action': 'rebuild_from_checkpoint',
                                        'missing_outputs': ['artifacts/kmc/generated_kmc.in'],
                                    }
                                ]
                            }
                        },
                    }
                },
            ),
        )
        self.assertIn('缺失产物并自动调整了恢复策略', rendered.output)

    def test_execution_tool_result_strategy_mentions_cascaded_steps(self):
        tool = get_chat_tool_definition('run')
        self.assertIsNotNone(tool)
        rendered = tool.render_chat_result(
            ToolIntent(action='run', params={'prompt': 'hello', 'resume_existing': True}),
            ToolExecutionOutcome(
                output='Run complete',
                ok=True,
                metadata={
                    'execution': {
                        'resume_existing': True,
                        'final_recovery': {
                            'recovery_plan': {
                                'steps': [
                                    {
                                        'step_id': 'kmc.prepare_input',
                                        'action': 'rebuild_from_checkpoint',
                                    },
                                    {
                                        'step_id': 'kmc.run',
                                        'action': 'restart_resumable_step',
                                        'invalidated_by': 'kmc.prepare_input',
                                    },
                                ]
                            }
                        },
                    }
                },
            ),
        )
        self.assertIn('下游步骤也会跟着重新处理', rendered.output)

    def test_execution_tool_result_strategy_mentions_drifted_outputs(self):
        tool = get_chat_tool_definition('run')
        self.assertIsNotNone(tool)
        rendered = tool.render_chat_result(
            ToolIntent(action='run', params={'prompt': 'hello', 'resume_existing': True}),
            ToolExecutionOutcome(
                output='Run complete',
                ok=True,
                metadata={
                    'execution': {
                        'resume_existing': True,
                        'final_recovery': {
                            'recovery_plan': {
                                'steps': [
                                    {
                                        'step_id': 'kmc.prepare_input',
                                        'action': 'rebuild_from_checkpoint',
                                        'drifted_outputs': ['artifacts/kmc/generated_kmc.in'],
                                    }
                                ]
                            }
                        },
                    }
                },
            ),
        )
        self.assertIn('产物内容已经变化', rendered.output)

    def test_dispatch_mcp_tool_validates_required_arguments(self):
        class FakeServer:
            project_root = str(ROOT)
            workspace_root = str(ROOT)
            output_dir = str(ROOT / 'runs')
            provider = 'local'

        with self.assertRaises(RuntimeError) as ctx:
            dispatch_mcp_tool(
                FakeServer(),
                'miet_plan_job',
                {},
                api={'MCPServerError': RuntimeError, '_tool_result': lambda text, structured=None: {'content': [{'type': 'text', 'text': text}], 'structuredContent': structured}},
            )
        self.assertIn('缺少必需参数 `job_spec_path`', str(ctx.exception))

    def test_run_scope_approval_asks_when_existing_run_context_would_be_replaced(self):
        decision = decide_tool_approval(
            ToolIntent(action='run', params={'prompt': 'x'}),
            context=ToolApprovalContext(active_run_dir='/tmp/existing-run'),
        )
        self.assertEqual(decision.action, 'ask')
        self.assertIn('当前已经有一个 run 上下文', decision.reason)

    def test_preview_run_is_allowed_under_confirm_mutations_policy(self):
        with mock.patch.dict('os.environ', {'MIETCLAW_TOOL_APPROVAL_POLICY': 'confirm_mutations'}, clear=False):
            decision = decide_tool_approval(
                ToolIntent(action='run', params={'prompt': 'x'}),
                context=ToolApprovalContext(preview_before_run=True),
            )
        self.assertEqual(decision.action, 'allow')

    def test_overwrite_run_requires_explicit_manual_confirmation(self):
        decision = decide_tool_approval(
            ToolIntent(action='run', params={'prompt': 'x', 'overwrite_existing': True}),
            context=ToolApprovalContext(active_run_dir='/tmp/existing-run'),
        )
        self.assertEqual(decision.action, 'ask')
        self.assertIn('overwrite_existing', decision.reason)


    def test_remember_tool_turn_appends_trace_replay_to_transcript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            turn = ToolTurnState(budget=ToolBudget(max_steps=2, max_mutating_steps=1, max_failures=1))
            turn.trace.add(
                AssistantActionBlockEvent(
                    block=AssistantActionBlock(
                        source='legacy_router_model',
                        raw_content='{"action":"runs"}',
                        tool_requests=[
                            ToolRequestBlock(
                                request_id='runs-latest',
                                intent=ToolIntent(action='runs', params={'limit': 5}),
                                source='legacy_router_model',
                            )
                        ],
                    )
                )
            )
            turn.trace.add(
                ToolResultBlockEvent(
                    block=ToolResultBlock(
                        request_id='runs-latest',
                        intent=ToolIntent(action='runs', params={'limit': 5}),
                        output='Recent runs\n- demo',
                        ok=True,
                        source='legacy_router',
                    )
                )
            )
            turn.trace.add(TurnFinishEvent(status='finish', reason='tool event loop produced a final answer', reply='Recent runs'))

            session._remember_tool_turn(turn)
            session._append_tool_trace(turn)
            transcript_text = session.transcript_path.read_text(encoding='utf-8')
            self.assertIn('### tool trace', transcript_text)
            self.assertIn('"traceId": "trace-', transcript_text)
            self.assertIn('"kind": "assistant_action_block"', transcript_text)
            self.assertIn('"kind": "turn_finish"', transcript_text)


    def test_tool_context_memory_prefers_block_backed_evidence_across_turns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            turn = ToolTurnState(budget=ToolBudget(max_steps=2, max_mutating_steps=1, max_failures=1))
            turn.outputs.append((ToolIntent(action='inspect', params={'run': 'latest'}), 'tuple fallback output'))
            turn.trace.add(
                ToolResultBlockEvent(
                    block=ToolResultBlock(
                        request_id='inspect-latest',
                        intent=ToolIntent(action='inspect', params={'run': 'latest'}),
                        output='authoritative block output',
                        ok=True,
                        source='legacy_router',
                    )
                )
            )

            session._remember_tool_turn(turn)

            history = session._tool_context_history(output_limit=200)
            self.assertEqual(history[0]['action'], 'inspect')
            self.assertIn('authoritative block output', history[0]['output'])
            self.assertNotIn('tuple fallback output', history[0]['output'])

            local_messages = session._build_local_model_messages()
            evidence_messages = [message for message in local_messages if 'authoritative session context' in message.get('content', '')]
            self.assertEqual(len(evidence_messages), 1)
            self.assertIn('source=legacy_router ok=True', evidence_messages[0]['content'])
            self.assertIn('authoritative block output', evidence_messages[0]['content'])


    def test_run_tool_event_loop_records_typed_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            turn = session._new_tool_turn_state(default_steps=4, step_env='MIETCLAW_AGENT_MAX_STEPS')
            engine = session._get_query_engine()
            router_blocks = [
                AssistantActionBlock(
                    source='heuristic_router',
                    tool_requests=[
                        ToolRequestBlock(
                            request_id='inspect-latest',
                            intent=ToolIntent(action='inspect', params={'run': 'latest'}),
                            source='heuristic_router',
                        )
                    ],
                    metadata={'synthetic': True},
                )
            ]
            with mock.patch.object(engine, 'resolve_tool_plan_blocks', return_value=[]):
                with mock.patch.object(engine, 'resolve_router_blocks', return_value=router_blocks):
                    with mock.patch.object(
                        session,
                        '_execute_tool_intent_outcome',
                        side_effect=[
                            ToolExecutionOutcome(output='Run detail\n- steps:\n  - kmc.run: failed\n', ok=True),
                            ToolExecutionOutcome(output='Log excerpt (kmc)\nMPI_ABORT', ok=True),
                        ],
                    ):
                        with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': False}):
                            reply = run_tool_event_loop(session, '帮我看看为什么失败了', state=turn)

            self.assertIn('MPI_ABORT', reply)
            event_kinds = [event.kind for event in turn.trace.events]
            self.assertIn('assistant_action_block', event_kinds)
            self.assertIn('tool_use', event_kinds)
            self.assertIn('permission_decision', event_kinds)
            self.assertIn('tool_result', event_kinds)
            self.assertIn('tool_result_block', event_kinds)
            self.assertEqual(event_kinds[-1], 'turn_finish')
            tool_actions = [event.intent.action for event in turn.trace.events if getattr(event, 'kind', '') == 'tool_use']
            self.assertEqual(tool_actions, ['inspect', 'logs'])

    def test_agent_loop_records_final_answer_block_from_legacy_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            with mock.patch('miet_claw.runtime.query_engine.should_try_agent_loop', return_value=True):
                with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': True}):
                    with mock.patch.object(session, '_heuristic_agent_first_step', return_value=None):
                        with mock.patch(
                            'miet_claw.chat.chat_with_local_model',
                            return_value={'content': '{"status":"finish","reply":"最终结论"}'},
                        ):
                            turn = session._new_tool_turn_state(default_steps=2, step_env='MIETCLAW_AGENT_MAX_STEPS')
                            reply = session._run_agent_loop('直接给结论', state=turn)

            self.assertEqual(reply, '最终结论')
            action_blocks = [event.block for event in turn.trace.events if getattr(event, 'kind', '') == 'assistant_action_block']
            self.assertEqual(len(action_blocks), 1)
            self.assertEqual(action_blocks[0].final_answer.reply, '最终结论')

    def test_confirm_mutations_policy_records_permission_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            intent = ToolIntent(action='draft', params={'prompt': 'hello'})
            with mock.patch.dict('os.environ', {'MIETCLAW_TOOL_APPROVAL_POLICY': 'confirm_mutations'}, clear=False):
                turn = session._new_tool_turn_state(default_steps=1)
                executed, should_stop = session._apply_tool_intent_to_turn(turn, intent, 'hello', source='legacy_router')
                self.assertFalse(executed)
                self.assertTrue(should_stop)
                decisions = [event for event in turn.trace.events if getattr(event, 'kind', '') == 'permission_decision']
                self.assertEqual(len(decisions), 1)
                self.assertEqual(decisions[0].decision, 'ask')

                manual_turn = session._new_tool_turn_state(default_steps=1)
                with mock.patch.object(session, '_execute_tool_intent_outcome', return_value=ToolExecutionOutcome(output='draft ok', ok=True)):
                    executed, should_stop = session._apply_tool_intent_to_turn(
                        manual_turn,
                        intent,
                        'hello',
                        manual=True,
                        source='manual_command',
                    )
                self.assertTrue(executed)
                self.assertFalse(should_stop)
                manual_decisions = [event for event in manual_turn.trace.events if getattr(event, 'kind', '') == 'permission_decision']
                self.assertEqual(len(manual_decisions), 1)
                self.assertEqual(manual_decisions[0].decision, 'allow')

    def test_query_engine_runs_plain_chat_turn_and_records_session_turn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = MietQueryEngine(session)
            healthy = {
                'healthy': True,
                'default_model': 'demo-model',
                'models': ['demo-model'],
                'base_url': 'http://127.0.0.1:8000',
            }

            with mock.patch('miet_claw.runtime.query_engine.should_skip_tool_router', return_value=True):
                with mock.patch.object(session, '_refresh_local_model_status', return_value=healthy):
                    with mock.patch('miet_claw.chat.chat_with_local_model', return_value={'content': 'plain chat answer'}):
                        result = engine.run_turn('你好')

            self.assertEqual(result.reply, 'plain chat answer')
            self.assertEqual(result.status, 'chat')
            self.assertFalse(result.used_tools)
            self.assertEqual(session.turns[-1]['status'], 'chat')
            self.assertEqual(session.turns[-1]['reply'], 'plain chat answer')
            self.assertIsNone(session.active_turn_id)
            self.assertEqual(session.usage_stats['chat_turns'], 1)

    def test_query_engine_tool_turn_commits_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = MietQueryEngine(session)
            turn = ToolTurnState(budget=ToolBudget(max_steps=1, max_mutating_steps=1, max_failures=1))
            turn.trace.add(
                AssistantActionBlockEvent(
                    block=AssistantActionBlock(
                        source='legacy_router_model',
                        tool_requests=[
                            ToolRequestBlock(
                                request_id='draft-1',
                                intent=ToolIntent(action='draft', params={'prompt': 'x'}),
                                source='legacy_router_model',
                            )
                        ],
                    )
                )
            )
            turn.trace.add(TurnFinishEvent(status='finish', reason='done', reply='draft blocked'))

            with mock.patch.object(engine, 'execute_engine_turn', return_value=('draft blocked', turn, True)):
                result = engine.run_turn('请直接起草一个 workflow')

            self.assertEqual(result.reply, 'draft blocked')
            self.assertTrue(result.used_tools)
            self.assertEqual(session.turns[-1]['used_tools'], True)
            self.assertIs(session.last_tool_turn_state, turn)
            self.assertEqual(session.pending_tool_requests[0]['action'], 'draft')
            self.assertGreaterEqual(len(session._runtime_state.turn_blocks[result.turn_id]), 1)
            self.assertGreaterEqual(len(session._runtime_state.turn_events[result.turn_id]), 1)

    def test_query_engine_execute_engine_turn_prefers_router_blocks_before_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = MietQueryEngine(session)
            turn = session._new_tool_turn_state(default_steps=2, step_env='MIETCLAW_AGENT_MAX_STEPS')
            router_blocks = [
                AssistantActionBlock(
                    source='router_model',
                    tool_requests=[
                        ToolRequestBlock(
                            request_id='inspect-latest',
                            intent=ToolIntent(action='inspect', params={'run': 'latest'}),
                            source='router_model',
                        )
                    ],
                    metadata={'native': True},
                )
            ]

            with mock.patch('miet_claw.runtime.query_engine.should_skip_tool_router', return_value=False):
                with mock.patch.object(engine, 'run_agent_loop', return_value=None) as run_agent_loop_mock:
                    with mock.patch.object(engine, 'handle_plan_branch', return_value=None) as handle_plan_branch_mock:
                        with mock.patch.object(engine, 'resolve_router_blocks', return_value=router_blocks) as resolve_router_blocks_mock:
                            with mock.patch.object(
                                engine,
                                'execute_router_blocks',
                                return_value=('router branch reply', True),
                            ) as execute_router_blocks_mock:
                                with mock.patch.object(engine, 'resolve_tool_intent') as resolve_tool_intent_mock:
                                    with mock.patch.object(engine, 'execute_legacy_router_step') as execute_legacy_router_step_mock:
                                        with mock.patch.object(engine, 'finalize_engine_turn') as finalize_engine_turn_mock:
                                            reply, returned_turn, used_tools = engine.execute_engine_turn('帮我看看最近的 run', state=turn)

            self.assertEqual(reply, 'router branch reply')
            self.assertIs(returned_turn, turn)
            self.assertTrue(used_tools)
            run_agent_loop_mock.assert_called_once_with('帮我看看最近的 run', state=turn)
            handle_plan_branch_mock.assert_called_once_with('帮我看看最近的 run', turn)
            resolve_router_blocks_mock.assert_called_once_with('帮我看看最近的 run', state=turn)
            execute_router_blocks_mock.assert_called_once_with('帮我看看最近的 run', turn, router_blocks)
            resolve_tool_intent_mock.assert_not_called()
            execute_legacy_router_step_mock.assert_not_called()
            finalize_engine_turn_mock.assert_not_called()

    def test_query_engine_execute_engine_turn_finalizes_when_router_blocks_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = MietQueryEngine(session)
            turn = session._new_tool_turn_state(default_steps=2, step_env='MIETCLAW_AGENT_MAX_STEPS')

            with mock.patch('miet_claw.runtime.query_engine.should_skip_tool_router', return_value=False):
                with mock.patch.object(engine, 'run_agent_loop', return_value=None):
                    with mock.patch.object(engine, 'handle_plan_branch', return_value=None):
                        with mock.patch.object(engine, 'resolve_router_blocks', return_value=[]):
                            with mock.patch.object(
                                engine,
                                'finalize_engine_turn',
                                return_value=('engine finalized reply', False),
                            ) as finalize_engine_turn_mock:
                                with mock.patch.object(engine, 'resolve_tool_intent') as resolve_tool_intent_mock:
                                    with mock.patch.object(engine, 'execute_legacy_router_step') as execute_legacy_router_step_mock:
                                        reply, returned_turn, used_tools = engine.execute_engine_turn('帮我看看最近的 run', state=turn)

            self.assertEqual(reply, 'engine finalized reply')
            self.assertIs(returned_turn, turn)
            self.assertFalse(used_tools)
            finalize_engine_turn_mock.assert_called_once_with('帮我看看最近的 run', turn)
            resolve_tool_intent_mock.assert_not_called()
            execute_legacy_router_step_mock.assert_not_called()

    def test_query_engine_handle_plan_branch_does_not_fall_back_to_compat_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = MietQueryEngine(session)
            turn = session._new_tool_turn_state(default_steps=2, step_env='MIETCLAW_AGENT_MAX_STEPS')

            with mock.patch.object(engine, 'resolve_tool_plan_blocks', return_value=[]):
                with mock.patch.object(engine, 'resolve_tool_plan') as resolve_tool_plan_mock:
                    with mock.patch.object(engine, 'execute_tool_plan') as execute_tool_plan_mock:
                        reply = engine.handle_plan_branch('帮我规划一下', turn)

            self.assertIsNone(reply)
            resolve_tool_plan_mock.assert_not_called()
            execute_tool_plan_mock.assert_not_called()

    def test_query_engine_run_once_payload_builds_snapshot_from_turn_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            trace_state = ToolTurnState(budget=ToolBudget(max_steps=1, max_mutating_steps=0, max_failures=1))
            trace_state.trace.add(
                AssistantActionBlockEvent(
                    block=AssistantActionBlock(
                        source='legacy_router_model',
                        tool_requests=[
                            ToolRequestBlock(
                                request_id='inspect-latest',
                                intent=ToolIntent(action='inspect', params={'run': 'latest'}),
                                source='legacy_router_model',
                            )
                        ],
                    )
                )
            )
            trace_state.trace.add(
                ToolResultBlockEvent(
                    block=ToolResultBlock(
                        request_id='inspect-latest',
                        intent=ToolIntent(action='inspect', params={'run': 'latest'}),
                        output='Run detail\\n- status: completed',
                        ok=True,
                        source='legacy_router',
                    )
                )
            )
            trace_state.trace.add(TurnFinishEvent(status='finish', reason='done', reply='engine payload reply'))
            turn_result = QueryTurnResult(
                turn_id='turn-payload',
                reply='engine payload reply',
                used_tools=True,
                status='tool',
                tool_turn_state=trace_state,
            )

            def fake_handle_line(message: str) -> str:
                session._last_query_turn_result = turn_result
                return turn_result.reply

            with mock.patch.object(engine, 'handle_line', side_effect=fake_handle_line) as handle_line_mock:
                payload = engine.run_once_payload('帮我看看最近有哪些 run', [])

            self.assertEqual(payload['reply'], 'engine payload reply')
            self.assertTrue(payload['used_tools'])
            self.assertEqual(payload['tool_trace_summary']['toolStepCount'], 1)
            self.assertEqual(payload['tool_timeline'][0]['stage'], 'assistant')
            self.assertEqual(payload['message']['toolTraceSummary']['toolStepCount'], 1)
            handle_line_mock.assert_called_once_with('帮我看看最近有哪些 run')

    def test_run_chat_once_payload_includes_engine_session_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            healthy = {
                'healthy': True,
                'default_model': 'demo-model',
                'models': ['demo-model'],
                'base_url': 'http://127.0.0.1:8000',
            }
            with mock.patch('miet_claw.runtime.query_engine.should_skip_tool_router', return_value=True):
                with mock.patch('miet_claw.chat.get_local_model_status', return_value=healthy):
                    with mock.patch('miet_claw.chat.chat_with_local_model', return_value={'content': 'plain chat answer'}):
                        payload = run_chat_once_payload(
                            project_root=str(ROOT),
                            workspace_root=tmpdir,
                            output_dir=tmpdir,
                            provider='local',
                            prompt='你好',
                        )

            self.assertEqual(payload['reply'], 'plain chat answer')
            self.assertEqual(payload['session']['turn_count'], 1)
            self.assertEqual(payload['message']['session']['turnCount'], 1)
            self.assertEqual(payload['session']['usage_stats']['chat_turns'], 1)
            self.assertEqual(payload['turn_state']['messageCount'], 2)

    def test_run_chat_once_payload_delegates_to_query_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = mock.Mock()
            fake_payload = {'reply': 'delegated payload'}
            engine.run_once_payload.return_value = fake_payload

            with mock.patch.object(MietClawChatSession, '_get_query_engine', return_value=engine):
                payload = run_chat_once_payload(
                    project_root=str(ROOT),
                    workspace_root=tmpdir,
                    output_dir=tmpdir,
                    provider='local',
                    prompt='你好',
                )

            self.assertIs(payload, fake_payload)
            engine.run_once_payload.assert_called_once()
            args, _ = engine.run_once_payload.call_args
            self.assertEqual(args[0], '你好')
            self.assertIsInstance(args[1], list)

    def test_chat_tool_wrappers_delegate_to_query_engine(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = mock.Mock()
            engine.build_tool_router_messages.return_value = [{'role': 'system', 'content': 'router'}]
            engine.build_tool_plan_messages.return_value = [{'role': 'system', 'content': 'plan'}]
            engine.build_agent_loop_messages.return_value = [{'role': 'system', 'content': 'agent'}]
            engine.build_tool_summary_messages.return_value = [{'role': 'system', 'content': 'summary'}]
            engine.render_plan_outputs.return_value = 'rendered plan'
            engine.summarize_tool_outputs.return_value = 'summarized output'
            engine.finalize_tool_turn.return_value = 'finalized output'
            engine.execute_tool_plan.return_value = 'plan execution output'
            engine.handle_plan_branch.return_value = 'plan branch reply'
            engine.resolve_tool_plan.return_value = ToolPlan(steps=[ToolIntent(action='inspect', params={'run': 'latest'})])
            engine.resolve_tool_intent.return_value = ToolIntent(action='logs', params={'run': 'latest'})
            engine.execute_legacy_router_step.return_value = 'router stop reply'
            engine.run_agent_loop.return_value = 'agent loop reply'
            engine.parse_legacy_agent_reply.return_value = AssistantActionBlock(
                source='legacy_agent_model',
                final_answer=FinalAnswerBlock(reply='legacy final', source='legacy_agent_model'),
            )
            engine.execute_legacy_agent_block.return_value = ('final_answer', 'legacy final')
            engine.finalize_engine_turn.return_value = ('engine final reply', True)
            engine.execute_engine_turn.return_value = ('engine turn reply', session._new_tool_turn_state(default_steps=1), True)
            engine.run_turn.return_value = QueryTurnResult(
                turn_id='turn-wrapper',
                reply='engine run-turn reply',
                used_tools=False,
                status='chat',
            )
            engine.handle_prompt_turn.return_value = QueryTurnResult(
                turn_id='turn-prompt-wrapper',
                reply='engine prompt-turn reply',
                used_tools=False,
                status='chat',
            )
            engine.handle_line.return_value = 'engine line reply'
            session._query_engine = engine

            self.assertEqual(session._build_tool_router_messages('看下日志')[0]['content'], 'router')
            self.assertEqual(session._build_tool_plan_messages('帮我规划')[0]['content'], 'plan')
            self.assertEqual(
                session._build_agent_loop_messages('继续', [], ToolBudget(max_steps=1, max_mutating_steps=0, max_failures=1))[0]['content'],
                'agent',
            )
            self.assertEqual(session._build_tool_summary_messages('总结', [], note='note')[0]['content'], 'summary')
            self.assertEqual(session._render_plan_outputs('总结', []), 'rendered plan')
            self.assertEqual(session._summarize_tool_outputs('总结', []), 'summarized output')
            self.assertEqual(session._finalize_tool_turn('总结', session._new_tool_turn_state(default_steps=1)), 'finalized output')
            self.assertEqual(session._execute_tool_plan(ToolPlan(), '总结'), 'plan execution output')
            self.assertEqual(session._handle_engine_plan_branch('帮我规划', session._new_tool_turn_state(default_steps=1)), 'plan branch reply')
            self.assertEqual(session._resolve_tool_plan('帮我规划').steps[0].action, 'inspect')
            self.assertEqual(session._resolve_tool_intent('看下日志').action, 'logs')
            self.assertEqual(
                session._execute_legacy_router_step(
                    '看下日志',
                    session._new_tool_turn_state(default_steps=1),
                    ToolIntent(action='logs', params={'run': 'latest'}),
                    raw_router_content='{}',
                ),
                'router stop reply',
            )
            self.assertEqual(session._run_agent_loop('继续分析'), 'agent loop reply')
            self.assertEqual(session._parse_legacy_agent_reply(session._new_tool_turn_state(default_steps=1), '{"status":"finish"}').final_answer.reply, 'legacy final')
            self.assertEqual(
                session._execute_legacy_agent_block(
                    '继续分析',
                    session._new_tool_turn_state(default_steps=1),
                    AssistantActionBlock(source='legacy_agent_model'),
                ),
                ('final_answer', 'legacy final'),
            )
            self.assertEqual(
                session._finalize_engine_turn('继续分析', session._new_tool_turn_state(default_steps=1)),
                ('engine final reply', True),
            )
            self.assertEqual(
                session._run_engine_turn('继续分析', state=session._new_tool_turn_state(default_steps=1))[0],
                'engine turn reply',
            )
            self.assertEqual(session._run_turn('继续分析').reply, 'engine run-turn reply')
            self.assertEqual(session._handle_prompt('继续分析'), 'engine prompt-turn reply')
            self.assertEqual(session.handle_line('继续分析'), 'engine line reply')

            engine.build_tool_router_messages.assert_called_once()
            engine.build_tool_plan_messages.assert_called_once()
            engine.build_agent_loop_messages.assert_called_once()
            engine.build_tool_summary_messages.assert_called_once()
            engine.render_plan_outputs.assert_called_once()
            engine.summarize_tool_outputs.assert_called_once()
            engine.finalize_tool_turn.assert_called_once()
            engine.execute_tool_plan.assert_called_once()
            engine.handle_plan_branch.assert_called_once()
            engine.resolve_tool_plan.assert_called_once()
            engine.resolve_tool_intent.assert_called_once()
            engine.execute_legacy_router_step.assert_called_once()
            engine.run_agent_loop.assert_called_once()
            engine.parse_legacy_agent_reply.assert_called_once()
            engine.execute_legacy_agent_block.assert_called_once()
            engine.finalize_engine_turn.assert_called_once()
            engine.execute_engine_turn.assert_called_once()
            engine.run_turn.assert_called_once()
            engine.handle_prompt_turn.assert_called_once()
            engine.handle_line.assert_called_once()

    def test_query_engine_handle_line_records_command_history_and_transcript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            with mock.patch.object(session, '_handle_command', return_value='status output') as handle_command_mock:
                handled = engine.handle_line('/status')

            self.assertEqual(handled, 'status output')
            handle_command_mock.assert_called_once_with('/status')
            self.assertEqual(session.history[-2:], [('user', '/status'), ('assistant', 'status output')])
            transcript_text = session.transcript_path.read_text(encoding='utf-8')
            self.assertIn('## user', transcript_text)
            self.assertIn('/status', transcript_text)
            self.assertIn('## mietclaw', transcript_text)
            self.assertIn('status output', transcript_text)

    def test_query_engine_handle_prompt_turn_records_history_and_transcript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            result = QueryTurnResult(turn_id='turn-handle', reply='engine handled reply', used_tools=False, status='chat')
            with mock.patch.object(engine, 'run_turn', return_value=result) as run_turn_mock:
                handled = engine.handle_prompt_turn('你好')

            self.assertEqual(handled.reply, 'engine handled reply')
            run_turn_mock.assert_called_once_with('你好')
            self.assertEqual(session.history[-2:], [('user', '你好'), ('assistant', 'engine handled reply')])
            transcript_text = session.transcript_path.read_text(encoding='utf-8')
            self.assertIn('## user', transcript_text)
            self.assertIn('你好', transcript_text)
            self.assertIn('## mietclaw', transcript_text)
            self.assertIn('engine handled reply', transcript_text)

    def test_query_engine_parse_reply_to_blocks_prefers_native_block_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()

            blocks = engine.parse_reply_to_blocks(
                json.dumps(
                    {
                        "blocks": [
                            {
                                "toolRequests": [
                                    {
                                        "action": "inspect",
                                        "params": {"run": "latest"},
                                    }
                                ]
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                source='native_test',
                mode='router',
            )

            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0].tool_requests[0].intent.action, 'inspect')
            self.assertTrue(blocks[0].metadata.get('native'))
            self.assertNotIn('legacyMode', blocks[0].metadata)

    def test_query_engine_resolve_tool_intent_prefers_router_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            blocks = [
                AssistantActionBlock(
                    source='router_model',
                    tool_requests=[
                        ToolRequestBlock(
                            request_id='inspect-latest',
                            intent=ToolIntent(action='inspect', params={'run': 'latest'}),
                            source='router_model',
                        )
                    ],
                    metadata={'native': True},
                )
            ]

            with mock.patch.object(engine, 'resolve_router_blocks', return_value=blocks) as resolve_router_blocks_mock:
                intent = engine.resolve_tool_intent('帮我看看最近的 run')

            self.assertEqual(intent.action, 'inspect')
            self.assertEqual(intent.params['run'], 'latest')
            resolve_router_blocks_mock.assert_called_once()

    def test_query_engine_resolve_tool_plan_prefers_plan_blocks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            blocks = [
                AssistantActionBlock(
                    source='planner_model',
                    tool_requests=[
                        ToolRequestBlock(
                            request_id='inspect-latest',
                            intent=ToolIntent(action='inspect', params={'run': 'latest'}),
                            source='planner_model',
                        )
                    ],
                    metadata={'native': True, 'plan_summarize': True},
                ),
                AssistantActionBlock(
                    source='planner_model',
                    tool_requests=[
                        ToolRequestBlock(
                            request_id='logs-current',
                            intent=ToolIntent(action='logs', params={'run': 'current', 'target': 'kmc'}),
                            source='planner_model',
                        )
                    ],
                    metadata={'native': True, 'plan_summarize': True},
                ),
            ]

            with mock.patch.object(engine, 'resolve_tool_plan_blocks', return_value=blocks) as resolve_tool_plan_blocks_mock:
                plan = engine.resolve_tool_plan('帮我分析最近一次 run 为什么失败')

            self.assertEqual([step.action for step in plan.steps], ['inspect', 'logs'])
            self.assertTrue(plan.summarize)
            resolve_tool_plan_blocks_mock.assert_called_once()

    def test_build_engine_context_includes_turn_state_and_followups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            turn_id = session._runtime_state.start_turn('帮我检查最近的 run')
            session._runtime_state.record_turn_message(turn_id, 'user', '帮我检查最近的 run')
            session._runtime_state.record_turn_followup(turn_id, {'kind': 'next_step', 'text': '- 可以继续 inspect。'})
            session._runtime_state.record_turn_denial(turn_id, {'action': 'run', 'decision': 'ask', 'reason': 'need confirmation'})
            session._runtime_state.set_turn_finish_reason(turn_id, status='tool', reason='need confirmation')
            session._runtime_state.set_turn_resume_source(turn_id, {'mode': 'resume', 'source_turn_id': 'turn-source'})
            session._runtime_state.record_turn_child(turn_id, 'turn-child', mode='retry', payload={'prompt': '继续分析'})
            session.pending_tool_requests = [{'action': 'inspect', 'params': {'run': 'latest'}}]
            session.permission_denials = [{'action': 'run', 'decision': 'ask', 'reason': 'need confirmation'}]
            session.queued_followups = [{'action': 'logs', 'target': 'kmc', 'text': '继续自动展开 kmc 日志', 'kind': 'followup_intent', 'runnable': True, 'auto_continue': True}]
            session.usage_stats = {'tool_turns': 2}
            session._runtime_state.aborted_turns.append({'turn_id': 'turn-old', 'reason': 'host stop'})
            current_state = session._new_tool_turn_state(default_steps=2)
            current_state.outputs.append((ToolIntent(action='inspect', params={'run': 'latest'}), 'done'))

            context = build_runtime_engine_context(
                current_run_dir=session.current_run_dir,
                current_report=session.current_report,
                current_bridge_summary=session.current_bridge_summary,
                current_moire_summary=session.current_moire_summary,
                current_moire_compare_summary=session.current_moire_compare_summary,
                current_moire_diffusion_summary=session.current_moire_diffusion_summary,
                existing_outputs=session._tool_context_outputs,
                existing_blocks=session._tool_context_blocks,
                current_outputs=current_state.outputs,
                current_blocks=[],
                output_limit=1600,
                intent_signature=session._intent_signature,
                truncate_output=lambda text, limit: text[:limit],
                api=__import__('miet_claw.chat', fromlist=['format_draft_report']).__dict__,
                active_turn_id=turn_id,
                current_turn=session._runtime_state.current_turn(),
                pending_tool_requests=session.pending_tool_requests,
                permission_denials=session.permission_denials,
                queued_followups=session.queued_followups,
                usage_stats=session.usage_stats,
                current_state=current_state,
                session_state=session._runtime_state,
            )

            self.assertEqual(context['turn_context']['active_turn_id'], turn_id)
            self.assertEqual(context['turn_context']['tool_step_count'], 1)
            self.assertEqual(context['turn_context']['message_count'], 1)
            self.assertEqual(context['turn_context']['turn_finish_reason'], 'need confirmation')
            self.assertEqual(context['turn_context']['turn_resume_source']['mode'], 'resume')
            self.assertEqual(context['turn_context']['turn_child_count'], 1)
            self.assertEqual(context['turn_context']['queued_followup_count'], 1)
            self.assertEqual(context['turn_context']['queued_runnable_followup_count'], 1)
            self.assertEqual(context['turn_context']['queued_auto_followup_count'], 1)
            self.assertEqual(context['turn_context']['aborted_turn_count'], 1)
            self.assertEqual(context['followup_context']['pending_tool_requests'][0]['action'], 'inspect')
            self.assertEqual(context['followup_context']['turn_followups'][0]['kind'], 'next_step')
            self.assertEqual(context['work_context']['active_kind'], None)

    def test_snapshot_from_turn_result_includes_turn_state_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            turn_id = session._runtime_state.start_turn('你好')
            session._runtime_state.record_turn_message(turn_id, 'user', '你好')
            session._runtime_state.record_turn_message(turn_id, 'assistant', 'hello')
            session._runtime_state.record_turn_block(turn_id, {'type': 'assistant_action', 'source': 'native'})
            session._runtime_state.record_turn_event(turn_id, {'index': 1, 'kind': 'assistant_action_block'})
            session._runtime_state.set_turn_status_detail(turn_id, {'status': 'chat', 'usedTools': False})

            result = QueryTurnResult(
                turn_id=turn_id,
                reply='hello',
                used_tools=False,
                status='chat',
            )

            payload = build_snapshot_from_turn_result(
                turn_result=result,
                session_state=session._runtime_state,
                transcript_path=session.transcript_path,
                selected_model=session.selected_model,
                history_length=len(session.history),
                current_run_dir=session.current_run_dir,
                current_report=session.current_report,
                current_bridge_summary=session.current_bridge_summary,
                current_moire_summary=session.current_moire_summary,
                current_moire_compare_summary=session.current_moire_compare_summary,
                current_moire_diffusion_summary=session.current_moire_diffusion_summary,
                turn_count=len(session.turns),
                active_turn_id=session.active_turn_id,
                permission_denial_count=len(session.permission_denials),
                usage_stats=session.usage_stats,
            )

            self.assertEqual(payload['turn_state']['turnId'], turn_id)
            self.assertEqual(payload['turn_state']['messageCount'], 2)
            self.assertEqual(payload['message']['turnState']['blockCount'], 1)
            self.assertEqual(payload['message']['session']['turnEventCount'], 1)

    def test_query_engine_commit_turn_records_turn_followups_and_finish_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            turn_id = session._runtime_state.start_turn('继续分析最近一次 run')
            result = QueryTurnResult(
                turn_id=turn_id,
                reply='可以继续 inspect 或者展开日志。',
                used_tools=False,
                status='chat',
                followups=[
                    {'kind': 'next_step', 'text': '- 我可以继续 inspect 最近一次 run。'},
                    {'kind': 'followup_prompt', 'text': '继续展开 kmc 日志'},
                ],
                turn_usage={'chat_turns': 1, 'model_calls': 1},
                finish_reason='chat',
            )

            engine.commit_turn(result)
            snapshot = session._runtime_state.resume_turn_state(turn_id)

            self.assertEqual(snapshot['followups'][0]['kind'], 'next_step')
            self.assertEqual(snapshot['usage']['chat_turns'], 1)
            self.assertEqual(snapshot['finish_reason']['reason'], 'chat')
            self.assertEqual(session.usage_stats['chat_turns'], 1)
            self.assertEqual(session._runtime_state.queued_followup_items(limit=4)[0]['kind'], 'next_step')
            self.assertEqual(session._runtime_state.queued_followup_items(limit=4)[0]['source_finish_reason'], 'chat')

    def test_query_engine_commit_turn_records_turn_denials_from_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            turn_id = session._runtime_state.start_turn('帮我直接启动新的 run')
            turn = ToolTurnState(budget=ToolBudget(max_steps=1, max_mutating_steps=1, max_failures=1))
            turn.trace.add(
                PermissionDecisionEvent(
                    intent=ToolIntent(action='run', params={'prompt': 'hello'}),
                    source='legacy_router_model',
                    decision='ask',
                    reason='need confirmation',
                )
            )
            turn.trace.add(TurnFinishEvent(status='stopped', reason='permission confirmation required', reply='先确认'))
            result = QueryTurnResult(
                turn_id=turn_id,
                reply='先确认',
                used_tools=True,
                status='tool',
                tool_turn_state=turn,
                finish_reason='permission confirmation required',
            )

            engine.commit_turn(result)
            snapshot = session._runtime_state.resume_turn_state(turn_id)

            self.assertEqual(snapshot['denials'][0]['action'], 'run')
            self.assertEqual(snapshot['finish_reason']['reason'], 'permission confirmation required')
            self.assertEqual(session.permission_denials[-1]['decision'], 'ask')

    def test_snapshot_from_turn_result_exposes_turn_followups_and_finish_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            turn_id = session._runtime_state.start_turn('继续分析')
            session._runtime_state.record_turn_message(turn_id, 'user', '继续分析')
            session._runtime_state.record_turn_message(turn_id, 'assistant', '结论')
            session._runtime_state.record_turn_followup(turn_id, {'kind': 'next_step', 'text': '- 可以继续 inspect。'})
            session._runtime_state.record_turn_usage(turn_id, {'chat_turns': 1})
            session._runtime_state.set_turn_finish_reason(turn_id, status='chat', reason='chat', reply='结论')
            session._runtime_state.set_turn_resume_source(turn_id, {'mode': 'retry', 'source_turn_id': 'turn-prev'})
            session._runtime_state.record_turn_child(turn_id, 'turn-child', mode='resume', payload={'prompt': '继续分析'})
            session._runtime_state.queue_followup(
                turn_id,
                {
                    'kind': 'followup_intent',
                    'text': '继续自动检查当前 run 的 `kmc` 日志。',
                    'action': 'logs',
                    'params': {'run': 'current', 'target': 'kmc'},
                    'auto_continue': True,
                },
            )
            session._runtime_state.aborted_turns.append({'turn_id': 'turn-aborted', 'reason': 'user stop'})

            result = QueryTurnResult(turn_id=turn_id, reply='结论', used_tools=False, status='chat')
            payload = build_snapshot_from_turn_result(
                turn_result=result,
                session_state=session._runtime_state,
                transcript_path=session.transcript_path,
                selected_model=session.selected_model,
                history_length=len(session.history),
                current_run_dir=session.current_run_dir,
                current_report=session.current_report,
                current_bridge_summary=session.current_bridge_summary,
                current_moire_summary=session.current_moire_summary,
                current_moire_compare_summary=session.current_moire_compare_summary,
                current_moire_diffusion_summary=session.current_moire_diffusion_summary,
                turn_count=len(session.turns),
                active_turn_id=session.active_turn_id,
                permission_denial_count=len(session.permission_denials),
                usage_stats=session.usage_stats,
            )

            self.assertEqual(payload['turn_state']['followupCount'], 1)
            self.assertEqual(payload['message']['turnState']['finishReason'], 'chat')
            self.assertEqual(payload['message']['session']['turnFollowupCount'], 1)
            self.assertEqual(payload['message']['turnState']['resumeSource']['mode'], 'retry')
            self.assertEqual(payload['turn_state']['childTurnCount'], 1)
            self.assertEqual(payload['turn_state']['queuedRunnableCount'], 1)
            self.assertEqual(payload['turn_state']['queuedAutoFollowupCount'], 1)
            self.assertEqual(payload['message']['session']['turnChildCount'], 1)
            self.assertEqual(payload['session']['queued_followup_count'], 1)
            self.assertEqual(payload['message']['session']['runnableFollowupCount'], 1)
            self.assertEqual(payload['message']['session']['autoFollowupCount'], 1)
            self.assertEqual(payload['message']['session']['abortedTurnCount'], 1)

    def test_snapshot_from_turn_result_exposes_memory_summary_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            archived_turn_id = session._runtime_state.start_turn('先总结最近一次 run')
            session._runtime_state.finish_turn(archived_turn_id, reply='旧总结', used_tools=False, status='chat')
            session._runtime_state.set_turn_finish_reason(archived_turn_id, status='chat', reason='chat')
            session._runtime_state.rebuild_memory_summary(live_turn_window=0)

            current_turn_id = session._runtime_state.start_turn('继续分析')
            result = QueryTurnResult(turn_id=current_turn_id, reply='结论', used_tools=False, status='chat')

            payload = build_snapshot_from_turn_result(
                turn_result=result,
                session_state=session._runtime_state,
                transcript_path=session.transcript_path,
                selected_model=session.selected_model,
                history_length=len(session.history),
                current_run_dir=session.current_run_dir,
                current_report=session.current_report,
                current_bridge_summary=session.current_bridge_summary,
                current_moire_summary=session.current_moire_summary,
                current_moire_compare_summary=session.current_moire_compare_summary,
                current_moire_diffusion_summary=session.current_moire_diffusion_summary,
                turn_count=len(session.turns),
                active_turn_id=session.active_turn_id,
                permission_denial_count=len(session.permission_denials),
                usage_stats=session.usage_stats,
            )

            self.assertEqual(payload['session']['memory_record_count'], 1)
            self.assertEqual(payload['message']['session']['memoryRecordCount'], 1)
            self.assertEqual(payload['message']['session']['freshMemoryCount'], 1)
            self.assertIn('Earlier working memory', payload['message']['session']['memorySummary'])

    def test_query_engine_ensure_turn_consumes_matching_queued_followup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            source_turn_id = session._runtime_state.start_turn('先看最近一次 run')
            session._runtime_state.finish_turn(source_turn_id, reply='先看完了', used_tools=False, status='chat')
            queued = session._runtime_state.queue_followup(
                source_turn_id,
                {'kind': 'followup_prompt', 'text': '继续展开 kmc 日志'},
            )

            turn_id = engine.ensure_turn('继续展开 kmc 日志')
            snapshot = session._runtime_state.resume_turn_state(turn_id)
            source_snapshot = session._runtime_state.resume_turn_state(source_turn_id)
            queued_items = session._runtime_state.queued_followup_items(include_consumed=True)

            self.assertEqual(snapshot['resume_source']['mode'], 'followup')
            self.assertEqual(snapshot['resume_source']['source_turn_id'], source_turn_id)
            self.assertEqual(snapshot['resume_source']['followup_id'], queued['followup_id'])
            self.assertEqual(source_snapshot['child_turns'][-1]['turn_id'], turn_id)
            self.assertEqual(source_snapshot['child_turns'][-1]['mode'], 'followup')
            consumed = next(item for item in queued_items if item['followup_id'] == queued['followup_id'])
            self.assertTrue(consumed['consumed'])
            self.assertEqual(consumed['consumed_by_turn_id'], turn_id)

    def test_query_engine_continue_queued_followup_marks_queue_item_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            source_turn_id = session._runtime_state.start_turn('先看最近一次 run')
            session._runtime_state.finish_turn(source_turn_id, reply='done', used_tools=False, status='chat')
            queued = session._runtime_state.queue_followup(
                source_turn_id,
                {'kind': 'followup_prompt', 'text': '继续展开 kmc 日志'},
            )

            def fake_handle_prompt_turn(prompt: str, *, auto_background: bool = True) -> QueryTurnResult:
                return QueryTurnResult(
                    turn_id=session._runtime_state.active_turn_id or 'turn-fallback',
                    reply=f'processed: {prompt}',
                    used_tools=False,
                    status='chat',
                    finish_reason='chat',
                )

            with mock.patch.object(engine, 'handle_prompt_turn', side_effect=fake_handle_prompt_turn) as handle_prompt_turn_mock:
                result = engine.run_queued_followup()

            self.assertIsNotNone(result)
            handle_prompt_turn_mock.assert_called_once_with('继续展开 kmc 日志', auto_background=True)
            queued_items = session._runtime_state.queued_followup_items(include_consumed=True)
            completed = next(item for item in queued_items if item['followup_id'] == queued['followup_id'])
            self.assertEqual(completed['status'], 'completed')
            self.assertEqual(completed['attempt_count'], 1)
            self.assertEqual(completed['completed_turn_id'], result.turn_id)
            self.assertEqual(completed['completed_finish_reason'], 'chat')

    def test_query_engine_followups_include_auto_followup_intent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            turn = session._new_tool_turn_state(default_steps=2)
            turn.outputs.append((ToolIntent(action='runs', params={}), '最近有 2 个 run'))
            strategy = mock.Mock(
                status='needs_more_evidence',
                reason='还缺日志证据',
                followup_intent=ToolIntent(action='logs', params={'run': 'current', 'target': 'kmc'}),
                next_steps=['- 继续查看 KMC 日志。'],
                followup_prompts=[],
            )

            with mock.patch.object(session, '_response_strategy_for_prompt', return_value=strategy):
                followups = engine._followups_for_turn('帮我看看为什么失败了', turn)

            auto_item = next(item for item in followups if item['kind'] == 'followup_intent')
            self.assertEqual(auto_item['action'], 'logs')
            self.assertTrue(auto_item['runnable'])
            self.assertTrue(auto_item['auto_continue'])

    def test_query_engine_run_background_followups_processes_auto_intent_item(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            source_turn_id = session._runtime_state.start_turn('帮我看看为什么失败了')
            session._runtime_state.finish_turn(source_turn_id, reply='先拿到初步证据', used_tools=True, status='tool')
            queued = session._runtime_state.queue_followup(
                source_turn_id,
                {
                    'kind': 'followup_intent',
                    'text': '继续自动检查当前 run 的 `kmc` 日志。',
                    'action': 'logs',
                    'params': {'run': 'current', 'target': 'kmc'},
                    'auto_continue': True,
                },
            )

            with mock.patch.object(
                session,
                '_execute_tool_intent_outcome',
                return_value=ToolExecutionOutcome(output='kmc 日志片段', ok=True),
            ):
                with mock.patch.object(session, '_response_strategy_for_prompt', return_value=mock.Mock(status='sufficient', reason='日志已经足够', followup_intent=None, next_steps=[], followup_prompts=[])):
                    results = engine.run_background_followups(source_turn_id=source_turn_id, limit=1)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, 'tool')
            queued_items = session._runtime_state.queued_followup_items(include_consumed=True)
            completed = next(item for item in queued_items if item['followup_id'] == queued['followup_id'])
            self.assertEqual(completed['status'], 'completed')
            self.assertEqual(completed['started_via'], 'background_auto')
            source_snapshot = session._runtime_state.resume_turn_state(source_turn_id)
            self.assertEqual(source_snapshot['child_turns'][-1]['mode'], 'followup')

    def test_query_engine_handle_prompt_turn_runs_background_followups_and_preserves_primary_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            background_result = QueryTurnResult(
                turn_id='turn-bg',
                reply='background reply',
                used_tools=True,
                status='tool',
                finish_reason='tool',
            )

            def fake_run_turn(prompt: str) -> QueryTurnResult:
                return QueryTurnResult(
                    turn_id=session._runtime_state.active_turn_id or 'turn-main',
                    reply='primary reply',
                    used_tools=False,
                    status='chat',
                    finish_reason='chat',
                )

            with mock.patch.object(engine, 'run_turn', side_effect=fake_run_turn):
                with mock.patch.object(engine, 'run_background_followups', return_value=[background_result]) as background_mock:
                    result = engine.handle_prompt_turn('继续分析')

            self.assertEqual(result.reply, 'primary reply')
            self.assertIs(session._last_query_turn_result, result)
            background_mock.assert_called_once_with(source_turn_id=result.turn_id)
            detail = session._runtime_state.turn_status_detail[result.turn_id]
            self.assertEqual(detail['autoFollowupCount'], 1)
            self.assertEqual(detail['autoFollowupTurns'], ['turn-bg'])

    def test_query_engine_abort_retry_and_resume_turns_record_runtime_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            active_turn_id = session._runtime_state.start_turn('执行中的回合')
            session.pending_tool_requests = [{'action': 'run', 'params': {'prompt': 'hello'}}]

            aborted = engine.abort_turn(active_turn_id, reason='user requested stop')

            self.assertEqual(aborted['finish_reason']['reason'], 'user requested stop')
            self.assertEqual(aborted['status_detail']['status'], 'aborted')
            self.assertEqual(session.pending_tool_requests, [])

            source_turn_id = session._runtime_state.start_turn('帮我检查最近一次 run')
            session._runtime_state.set_turn_finish_reason(source_turn_id, status='tool', reason='tool event loop produced a final answer', reply='done')
            session._runtime_state.finish_turn(source_turn_id, reply='done', used_tools=True, status='tool')

            with mock.patch('miet_claw.runtime.query_engine.should_skip_tool_router', return_value=True):
                with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': False, 'error': 'offline'}):
                    retry_result = engine.retry_turn(source_turn_id)
                    retry_snapshot = session._runtime_state.resume_turn_state(retry_result.turn_id)
                    resume_result = engine.resume_turn(source_turn_id)
                    resume_snapshot = session._runtime_state.resume_turn_state(resume_result.turn_id)
            source_snapshot = session._runtime_state.resume_turn_state(source_turn_id)

            self.assertEqual(retry_snapshot['resume_source']['mode'], 'retry')
            self.assertEqual(retry_snapshot['resume_source']['source_turn_id'], source_turn_id)
            self.assertEqual(resume_snapshot['resume_source']['mode'], 'resume')
            self.assertEqual(resume_snapshot['resume_source']['source_turn_id'], source_turn_id)
            self.assertEqual(len(source_snapshot['child_turns']), 2)
            self.assertEqual(source_snapshot['child_turns'][0]['mode'], 'retry')
            self.assertEqual(source_snapshot['child_turns'][1]['mode'], 'resume')
            self.assertIn('本地模型当前不可用', retry_result.reply)
            self.assertIn('本地模型当前不可用', resume_result.reply)

    def test_run_tool_event_loop_uses_run_engine_turn_wrapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            turn = session._new_tool_turn_state(default_steps=4, step_env='MIETCLAW_AGENT_MAX_STEPS')
            with mock.patch.object(session, '_run_engine_turn', return_value=('planner direct reply', turn, False)) as run_engine_turn_mock:
                reply = run_tool_event_loop(session, '请直接给我规划结论', state=turn)

            self.assertEqual(reply, 'planner direct reply')
            run_engine_turn_mock.assert_called_once()

    def test_run_agent_query_loop_uses_run_agent_loop_wrapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            turn = session._new_tool_turn_state(default_steps=4, step_env='MIETCLAW_AGENT_MAX_STEPS')
            with mock.patch.object(session, '_run_agent_loop', return_value='agent wrapper reply') as run_agent_loop_mock:
                reply = run_agent_query_loop(session, '继续分析', state=turn)

            self.assertEqual(reply, 'agent wrapper reply')
            run_agent_loop_mock.assert_called_once()

    def test_run_agent_query_loop_records_finish_from_legacy_final_answer_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            turn = session._new_tool_turn_state(default_steps=2, step_env='MIETCLAW_AGENT_MAX_STEPS')
            with mock.patch('miet_claw.runtime.query_engine.should_skip_tool_router', return_value=False):
                with mock.patch('miet_claw.runtime.query_engine.should_try_agent_loop', return_value=True):
                    with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': True}):
                        with mock.patch.object(session, '_heuristic_agent_first_step', return_value=None):
                            with mock.patch.object(session, '_forced_log_target', return_value=None):
                                with mock.patch.object(
                                    session,
                                    '_chat_with_local_model',
                                    return_value={'content': '{"status":"finish","reply":"agent final answer"}'},
                                ):
                                    reply = session._run_agent_loop('请直接总结现状', state=turn)

            self.assertEqual(reply, 'agent final answer')
            self.assertEqual(turn.trace.events[-1].kind, 'turn_finish')
            self.assertEqual(turn.trace.events[-1].reason, 'agent model chose to finish')
            self.assertEqual(turn.trace.events[-1].reply, 'agent final answer')

    def test_run_agent_query_loop_routes_forced_log_followup_through_helper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            turn = session._new_tool_turn_state(default_steps=2, step_env='MIETCLAW_AGENT_MAX_STEPS')
            with mock.patch('miet_claw.runtime.query_engine.should_skip_tool_router', return_value=False):
                with mock.patch('miet_claw.runtime.query_engine.should_try_agent_loop', return_value=True):
                    with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': True}):
                        with mock.patch.object(session, '_heuristic_agent_first_step', return_value=None):
                            with mock.patch.object(session, '_forced_log_target', side_effect=[None, 'kmc']):
                                with mock.patch.object(
                                    session,
                                    '_chat_with_local_model',
                                    return_value={
                                        'content': '{"status":"continue","step":{"action":"inspect","params":{"run":"latest"}}}'
                                    },
                                ):
                                    with mock.patch.object(
                                        session,
                                        '_apply_tool_intent_to_turn',
                                        return_value=(True, True),
                                    ) as apply_mock:
                                        with mock.patch.object(
                                            session,
                                            '_finalize_tool_turn',
                                            return_value='forced followup reply',
                                        ):
                                            reply = session._run_agent_loop('帮我看看为什么这个 run 异常', state=turn)

            self.assertEqual(reply, 'forced followup reply')
            apply_mock.assert_called_once()
            forced_intent = apply_mock.call_args.args[1]
            self.assertEqual(forced_intent.action, 'logs')
            self.assertEqual(forced_intent.params['target'], 'kmc')
            self.assertEqual(turn.trace.events[-1].reason, 'forced log followup requested a stop')

    def test_run_agent_query_loop_stops_on_repeated_request_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            turn = session._new_tool_turn_state(default_steps=2, step_env='MIETCLAW_AGENT_MAX_STEPS')
            repeated_intent = ToolIntent(action='inspect', params={'run': 'latest'})
            turn.seen_signatures.add(session._intent_signature(repeated_intent))
            turn.outputs.append((repeated_intent, '已有 inspect 证据'))
            with mock.patch('miet_claw.runtime.query_engine.should_skip_tool_router', return_value=False):
                with mock.patch('miet_claw.runtime.query_engine.should_try_agent_loop', return_value=True):
                    with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': True}):
                        with mock.patch.object(session, '_heuristic_agent_first_step', return_value=None):
                            with mock.patch.object(session, '_forced_log_target', return_value=None):
                                with mock.patch.object(
                                    session,
                                    '_chat_with_local_model',
                                    return_value={
                                        'content': '{"status":"continue","step":{"action":"inspect","params":{"run":"latest"}}}'
                                    },
                                ):
                                    reply = session._run_agent_loop('继续分析这个 run', state=turn)

            self.assertIn('重复请求同一步工具', reply)
            self.assertEqual(turn.trace.events[-1].reason, 'agent loop repeated the same tool step')
            self.assertTrue(any('重复请求同一步工具' in note for note in turn.notes))

    def test_run_agent_query_loop_uses_session_legacy_agent_wrappers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = MietClawChatSession(
                project_root=str(ROOT),
                workspace_root=tmpdir,
                output_dir=tmpdir,
                provider='local',
            )
            engine = session._get_query_engine()
            turn = session._new_tool_turn_state(default_steps=2, step_env='MIETCLAW_AGENT_MAX_STEPS')
            block = AssistantActionBlock(
                source='legacy_agent_model',
                final_answer=FinalAnswerBlock(reply='wrapped final answer', source='legacy_agent_model'),
            )
            with mock.patch('miet_claw.runtime.query_engine.should_skip_tool_router', return_value=False):
                with mock.patch('miet_claw.runtime.query_engine.should_try_agent_loop', return_value=True):
                    with mock.patch.object(session, '_refresh_local_model_status', return_value={'healthy': True}):
                        with mock.patch.object(session, '_heuristic_agent_first_step', return_value=None):
                            with mock.patch.object(session, '_forced_log_target', return_value=None):
                                with mock.patch.object(
                                    session,
                                    '_chat_with_local_model',
                                    return_value={'content': '{"status":"finish","reply":"wrapped final answer"}'},
                                ):
                                    with mock.patch.object(engine, 'parse_legacy_agent_reply', return_value=block) as parse_mock:
                                        with mock.patch.object(
                                            engine,
                                            'execute_legacy_agent_block',
                                            return_value=('final_answer', 'wrapped final answer'),
                                        ) as execute_mock:
                                            reply = session._run_agent_loop('直接给我结论', state=turn)

            self.assertEqual(reply, 'wrapped final answer')
            parse_mock.assert_called_once()
            execute_mock.assert_called_once()


if __name__ == '__main__':
    unittest.main()
