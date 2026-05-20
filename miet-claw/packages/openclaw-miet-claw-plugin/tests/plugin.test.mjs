import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import {
  buildAssistantMessage,
  buildToolTimelineFromReplay,
  buildMietRunCommandActions,
  buildRuntimeMessageMeta,
  buildMietWebBootstrap,
  createMietClawToolCatalog,
  createMietClawTools,
  detectProjectRoot,
  listMietClawToolNames,
  listMietCommandDetails,
  listMietCommandHints,
  loadMietTranscriptEvidence,
  loadRunSnapshot,
  MIET_PYTHON_MCP_TOOL_DEFINITIONS,
  renderMietSlashCommandHelpText,
  registerMietClawTools,
  renderRunSnapshot,
  runCommand,
  runMietAutonomyCli,
  runMietCli,
  runMietJsonCli,
} from '../runtime.js';

const projectRoot = path.resolve(path.join(import.meta.dirname, '..', '..', '..'));

const api = {
  rootDir: path.join(projectRoot, 'packages', 'openclaw-miet-claw-plugin'),
  pluginConfig: {
    projectRoot,
    pythonBin: 'python3',
    defaultOutputDir: path.join(projectRoot, 'runs'),
  },
};

const expectedPythonMcpToolNames = [
  'miet_runtime_doctor',
  'miet_list_runs',
  'miet_inspect_run',
  'miet_get_logs',
  'miet_list_artifacts',
  'miet_autonomy_draft',
  'miet_autonomy_run',
  'miet_plan_job',
  'miet_run_job',
  'miet_kmc_bridge',
  'miet_moire_run',
  'miet_moire_compare',
  'miet_moire_diffusion_sweep',
  'miet_moire_lammps',
  'miet_moire_kmc',
];

const expectedOpenClawToolNames = [
  ...expectedPythonMcpToolNames,
  'miet_resume_job',
];

async function makeTempOutputDir(prefix) {
  return await fs.promises.mkdtemp(path.join(os.tmpdir(), prefix));
}

test('detectProjectRoot finds the repo root', () => {
  const detected = detectProjectRoot([path.join(projectRoot, 'packages', 'openclaw-miet-claw-plugin')]);
  assert.equal(detected, projectRoot);
});

test('createMietClawToolCatalog exposes the expected tool registry', () => {
  const catalog = createMietClawToolCatalog(api);
  assert.equal(catalog.length, expectedOpenClawToolNames.length);
  assert.deepEqual(listMietClawToolNames(), catalog.map((tool) => tool.name));
  assert.deepEqual(catalog.map((tool) => tool.name), expectedOpenClawToolNames);
  assert.equal(catalog.every((tool) => typeof tool.execute === 'function'), true);
});

test('OpenClaw registry stays in sync with the Python MCP registry', async () => {
  assert.deepEqual(
    MIET_PYTHON_MCP_TOOL_DEFINITIONS.map((tool) => tool.name),
    expectedPythonMcpToolNames,
  );

  const result = await runCommand([
    'python3',
    '-c',
    [
      'import json',
      'from miet_claw.runtime.tool_registry import mcp_tool_definitions',
      'print(json.dumps(mcp_tool_definitions(), ensure_ascii=False))',
    ].join('; '),
  ], {
    cwd: projectRoot,
    env: {
      ...process.env,
      PYTHONPATH: path.join(projectRoot, 'src'),
    },
  });

  assert.equal(result.code, 0, result.stderr);
  const pythonDefinitions = JSON.parse(result.stdout);
  const pythonNames = pythonDefinitions.map((tool) => tool.name);
  assert.deepEqual(pythonNames, expectedPythonMcpToolNames);
  assert.deepEqual(MIET_PYTHON_MCP_TOOL_DEFINITIONS, pythonDefinitions);
  for (const name of pythonNames) {
    assert.ok(listMietClawToolNames().includes(name), `${name} missing from OpenClaw registry`);
  }
});

test('slash command helpers expose shared hints and help text', () => {
  const hints = listMietCommandHints();
  const details = listMietCommandDetails();
  assert.ok(hints.includes('/status'));
  assert.ok(hints.includes('/run <natural language task>'));
  assert.equal(details.length, hints.length);
  assert.equal(details[0].command, hints[0]);
  const helpText = renderMietSlashCommandHelpText();
  assert.match(helpText, /Available slash commands:/);
  assert.match(helpText, /\/runs — List recent workflow runs\./);
});

test('buildMietWebBootstrap exposes command details for the web shell', async () => {
  const payload = await buildMietWebBootstrap(api, {
    includeSystem: false,
    includeLocalModel: false,
    includeCommandHints: true,
  });
  assert.ok(Array.isArray(payload.commandHints));
  assert.ok(Array.isArray(payload.commandDetails));
  assert.equal(payload.commandDetails.length, payload.commandHints.length);
  assert.equal(payload.commandDetails[0].command, payload.commandHints[0]);
  assert.ok(payload.commandDetails[0].summary);
});

test('buildMietRunCommandActions exposes reusable prepared commands', () => {
  const actions = buildMietRunCommandActions('demo-run', {
    includeInspect: true,
    includeSummaryLog: true,
    includeKmcLog: false,
  });
  assert.deepEqual(actions, [
    { command: '/inspect demo-run', label: 'Prepare inspect command' },
    { command: '/logs demo-run summary', label: 'Prepare summary log command' },
  ]);
});

test('buildAssistantMessage carries tool trace evidence into shared cards', () => {
  const meta = buildRuntimeMessageMeta({
    session: { transcriptPath: '/tmp/chat.md', approvalPolicy: 'allow_all' },
    current: { activeKind: 'report', hasAny: true, report: { job_id: 'demo', generated_files: {} } },
    currentReport: { job_id: 'demo', generated_files: {} },
    toolTraceSummary: { toolStepCount: 1, finishStatus: 'finish' },
    toolEvidence: [{ step: 1, action: 'inspect', source: 'legacy_router', outputPreview: 'ok' }],
    toolTraceReplay: [{ index: 1, kind: 'tool_result_block', action: 'inspect', ok: true }],
  });
  const message = buildAssistantMessage({
    content: 'done',
    kind: 'tool',
    meta,
  });
  assert.equal(message.toolTraceSummary.toolStepCount, 1);
  assert.equal(message.toolEvidence[0].action, 'inspect');
  assert.equal(message.toolTraceReplay[0].kind, 'tool_result_block');
  assert.equal(message.toolTimeline[0].stage, 'tool_result');
  assert.equal(message.toolTraceId, null);
  assert.equal(message.cards[0].toolTraceSummary.finishStatus, 'finish');
  assert.equal(message.cards[0].toolTraceReplay[0].kind, 'tool_result_block');
  assert.equal(message.cards[1].type, 'tool_timeline');
  assert.equal(message.cards[1].timeline[0].stage, 'tool_result');
  assert.equal(message.cards[2].toolEvidence[0].action, 'inspect');
});

test('buildToolTimelineFromReplay turns replay rows into readable timeline items', () => {
  const timeline = buildToolTimelineFromReplay([
    { index: 1, kind: 'assistant_action_block', toolActions: ['runs'], source: 'legacy_router_model' },
    { index: 2, kind: 'permission_decision', action: 'runs', decision: 'allow', reason: 'readonly', source: 'legacy_router' },
    { index: 3, kind: 'turn_finish', status: 'finish', reason: 'done' },
  ]);
  assert.equal(timeline.length, 3);
  assert.equal(timeline[0].stage, 'assistant');
  assert.equal(timeline[1].status, 'completed');
  assert.equal(timeline[2].stage, 'finish');
});

test('loadMietTranscriptEvidence finds a specific trace event in transcript markdown', async () => {
  const tmpPath = path.join(projectRoot, '.tmp-transcript-evidence.md');
  await fs.promises.writeFile(
    tmpPath,
    [
      '## mietclaw',
      '',
      '### tool trace',
      '',
      '```json',
      JSON.stringify({
        traceId: 'trace-demo123',
        summary: { finishStatus: 'finish' },
        replay: [
          { index: 1, kind: 'assistant_action_block', toolActions: ['runs'] },
          { index: 2, kind: 'turn_finish', status: 'finish', reason: 'done' },
        ],
      }, null, 2),
      '```',
      '',
    ].join('\n'),
    'utf-8',
  );
  try {
    const evidence = await loadMietTranscriptEvidence({
      transcriptPath: tmpPath,
      traceId: 'trace-demo123',
      eventIndex: 2,
    });
    assert.equal(evidence.traceId, 'trace-demo123');
    assert.equal(evidence.event.kind, 'turn_finish');
    assert.match(evidence.excerpt, /"index": 2/);
  } finally {
    await fs.promises.rm(tmpPath, { force: true });
  }
});

test('createMietClawTools keeps registry metadata and optional flags', () => {
  const tools = createMietClawTools(api);
  assert.equal(tools.length, expectedOpenClawToolNames.length);
  assert.equal(tools[0].definition.name, 'miet_runtime_doctor');
  const byName = Object.fromEntries(tools.map((tool) => [tool.definition.name, tool]));
  assert.deepEqual(byName.miet_run_job.options, { optional: true });
  assert.deepEqual(byName.miet_moire_run.options, { optional: true });
  assert.deepEqual(byName.miet_moire_compare.options, { optional: true });
  assert.deepEqual(byName.miet_moire_diffusion_sweep.options, { optional: true });
  assert.deepEqual(byName.miet_moire_lammps.options, { optional: true });
  assert.deepEqual(byName.miet_moire_kmc.options, { optional: true });
  assert.deepEqual(byName.miet_resume_job.options, { optional: true });
});

test('registerMietClawTools registers every tool exactly once', () => {
  const calls = [];
  registerMietClawTools({
    ...api,
    registerTool(definition, options) {
      calls.push({ definition, options });
    },
  });
  assert.equal(calls.length, expectedOpenClawToolNames.length);
  assert.deepEqual(
    calls.map((call) => call.definition.name),
    createMietClawToolCatalog(api).map((tool) => tool.name),
  );
});

test('runMietJsonCli can read the runtime doctor payload', async () => {
  const output = await runMietJsonCli(api, {
    subcommand: 'doctor',
    projectRoot,
  });

  assert.equal(output.payload.agent_name, 'mietclaw');
  assert.ok(output.payload.paths);
  assert.equal(output.payload.project_root, projectRoot);
});

test('runMietCli plans the sample chain job', async () => {
  const output = await runMietCli(api, {
    subcommand: 'plan',
    jobSpecPath: 'examples/jobs/md_to_kmc_chain.json',
  });

  assert.equal(output.plan.job_id, 'fe_cu_ni_vacancy_chain');
  assert.equal(output.plan.plan[0].id, 'md.run');
});

test('runMietCli can execute dry-run chain job and inspect the result', async () => {
  const outputDir = await makeTempOutputDir('mietclaw-plugin-run-');
  const output = await runMietCli(api, {
    subcommand: 'run',
    jobSpecPath: 'examples/jobs/md_to_kmc_chain.json',
    outputDir,
    dryRun: true,
  });

  const snapshot = await loadRunSnapshot(output.runDir);
  const rendered = renderRunSnapshot(snapshot);

  assert.match(rendered, /Run directory:/);
  assert.match(rendered, /Archived files:/);
  assert.equal(snapshot.state.steps['md.run'].status, 'completed');
});

test('loadRunSnapshot can read summary.json based runs', async () => {
  const workdir = path.join(projectRoot, '.runs-plugin-summary-test');
  await fs.promises.mkdir(workdir, { recursive: true });
  await fs.promises.writeFile(
    path.join(workdir, 'summary.json'),
    JSON.stringify({
      status: 'completed',
      source_case_dir: '/tmp/MoRe/Re_0.07/model_4',
      barrier_eV: 0.59798,
      kmc: {
        barrier_eV: 0.59798,
        parsed_run: { accepted_events: 6, final_time: 1.0e-10 },
      },
    }),
    'utf-8',
  );

  const snapshot = await loadRunSnapshot(workdir);
  const rendered = renderRunSnapshot(snapshot);

  assert.equal(snapshot.state, null);
  assert.equal(snapshot.summaryJson.status, 'completed');
  assert.match(rendered, /Barrier: 0.59798 eV/);
});

test('runMietAutonomyCli drafts a workspace from a natural-language prompt', async () => {
  const output = await runMietAutonomyCli(api, {
    subcommand: 'autonomy-draft',
    prompt: 'Create a native MD to KMC vacancy diffusion job for plugin autonomy test at 845 K.',
    provider: 'local',
    workspaceRoot: path.join(projectRoot, '.autonomy-plugin-test'),
  });

  assert.equal(output.payload.mode, 'md_to_kmc_chain');
  assert.equal(output.payload.provider_used, 'local-heuristic');
  assert.match(output.payload.generated_files.job_spec, /job_spec\.generated\.json$/);
});

test('runMietAutonomyCli can validate an autonomy job with dry-run only', async () => {
  const output = await runMietAutonomyCli(api, {
    subcommand: 'autonomy-run',
    prompt: 'Create a KMC only diffusion job for plugin autonomy run test at 830 K with Fe=0.62 eV, Cu=0.54 eV, Ni=0.53 eV.',
    provider: 'local',
    workspaceRoot: path.join(projectRoot, '.autonomy-plugin-test'),
    outputDir: path.join(projectRoot, '.runs-plugin-test'),
    dryRun: true,
  });

  assert.equal(output.payload.mode, 'kmc_only');
  assert.equal(output.payload.execution.dry_run_only, true);
  assert.equal(output.payload.execution.final_run_dir, null);
  assert.match(output.payload.execution.validation_run_dir, /validation_runs/);
});

test('runMietJsonCli lists runs and returns summary logs', async () => {
  const outputDir = await makeTempOutputDir('mietclaw-plugin-json-');
  const output = await runMietCli(api, {
    subcommand: 'run',
    jobSpecPath: 'examples/jobs/md_to_kmc_chain.json',
    outputDir,
    dryRun: true,
  });

  const runs = await runMietJsonCli(api, {
    subcommand: 'runs',
    outputDir,
    limit: 3,
  });
  assert.ok(Array.isArray(runs.payload.runs));
  assert.ok(runs.payload.runs.length >= 1);

  const inspected = await runMietJsonCli(api, {
    subcommand: 'inspect',
    runName: path.basename(output.runDir),
    outputDir,
  });
  assert.equal(inspected.payload.path, output.runDir);

  const logs = await runMietJsonCli(api, {
    subcommand: 'logs',
    runDir: output.runDir,
    outputDir,
    target: 'summary',
    maxLines: 20,
  });
  assert.equal(logs.payload.target, 'summary');
  assert.equal(logs.payload.available, true);
  assert.match(logs.payload.content, /KMC 结果摘要|barrier \/ rate/);
});

test('runMietJsonCli can bridge event + neb into a validated lookup summary', async () => {
  const workdir = path.join(projectRoot, '.runs-plugin-json-bridge');
  const helper = path.join(workdir, 'fake_bridge.py');
  await fs.promises.mkdir(workdir, { recursive: true });
  await fs.promises.writeFile(
    helper,
    [
      'import argparse',
      'import json',
      'from pathlib import Path',
      '',
      'parser = argparse.ArgumentParser()',
      "parser.add_argument('--event-json')",
      "parser.add_argument('--workdir')",
      "parser.add_argument('--neb-txt')",
      "parser.add_argument('--barrier')",
      "parser.add_argument('--validate', action='store_true')",
      'args = parser.parse_args()',
      'workdir = Path(args.workdir)',
      'workdir.mkdir(parents=True, exist_ok=True)',
      "(workdir / 'summary.json').write_text(json.dumps({",
      "  'barrier_eV': 0.42,",
      "  'files': {",
      "    'barriers_tsv': str(workdir / 'barriers.tsv'),",
      "    'state_values_sites': str(workdir / 'state.values.sites'),",
      "    'input_ml': str(workdir / 'input.ml'),",
      "    'run_out': str(workdir / 'run.out'),",
      '  },',
      "  'validation': {'lookup_hits': 1, 'live_ml_misses': 0},",
      "  'validation_passed': True,",
      '}), encoding=\'utf-8\')',
    ].join('\n'),
    'utf-8',
  );

  const eventJson = path.join(workdir, 'event.json');
  const nebTxt = path.join(workdir, 'neb.txt');
  await fs.promises.writeFile(eventJson, '{}', 'utf-8');
  await fs.promises.writeFile(nebTxt, '#reaction_coordinate de\n0 0\n0.5 0.42\n', 'utf-8');

  const prevBridgeScript = process.env.MIETCLAW_KMC_BRIDGE_SCRIPT;
  const prevBridgePython = process.env.MIETCLAW_KMC_BRIDGE_PYTHON;
  process.env.MIETCLAW_KMC_BRIDGE_SCRIPT = helper;
  process.env.MIETCLAW_KMC_BRIDGE_PYTHON = 'python3';

  try {
    const output = await runMietJsonCli(api, {
      subcommand: 'bridge',
      eventJson,
      nebTxt,
      outputDir: workdir,
      workdir: path.join(workdir, 'bridge-out'),
      validate: true,
    });
    assert.equal(output.payload.validation_passed, true);
    assert.equal(output.payload.validation.lookup_hits, 1);
  } finally {
    if (prevBridgeScript === undefined) delete process.env.MIETCLAW_KMC_BRIDGE_SCRIPT;
    else process.env.MIETCLAW_KMC_BRIDGE_SCRIPT = prevBridgeScript;

    if (prevBridgePython === undefined) delete process.env.MIETCLAW_KMC_BRIDGE_PYTHON;
    else process.env.MIETCLAW_KMC_BRIDGE_PYTHON = prevBridgePython;
  }
});
