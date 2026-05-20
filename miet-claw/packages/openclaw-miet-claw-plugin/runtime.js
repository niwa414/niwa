import fs from 'node:fs';
import fsp from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { spawn } from 'node:child_process';

const TOOL_PARAM_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    prompt: {
      type: 'string',
      description: 'Natural-language autonomy prompt describing the MD/KMC task.',
    },
    provider: {
      type: 'string',
      description: 'Autonomy provider: auto, local, or claude.',
    },
    job_spec_path: {
      type: 'string',
      description: 'Path to an existing job spec JSON file.',
    },
    job_spec: {
      type: 'object',
      description: 'Inline job spec JSON payload.',
      additionalProperties: true,
    },
    job_name_hint: {
      type: 'string',
      description: 'Helpful name used when storing an inline spec file.',
    },
    output_dir: {
      type: 'string',
      description: 'Where the run directory should be created.',
    },
    workspace_root: {
      type: 'string',
      description: 'Where autonomy draft workspaces should be stored.',
    },
    mode: {
      type: 'string',
      description: 'Optional mode hint: md_only, kmc_only, or md_to_kmc_chain.',
    },
    template_path: {
      type: 'string',
      description: 'Optional template JSON path used by the autonomy layer.',
    },
    material_name: {
      type: 'string',
      description: 'Optional material system name override for the autonomy layer.',
    },
    run_dir: {
      type: 'string',
      description: 'Existing run directory to resume or inspect.',
    },
    event_json: {
      type: 'string',
      description: 'Absolute path to the KMC event.json input for lookup bridge generation.',
    },
    neb_txt: {
      type: 'string',
      description: 'Absolute path to the LAMMPS NEB output text file.',
    },
    barrier: {
      type: 'number',
      description: 'Optional barrier value in eV when no neb.txt is available.',
    },
    workdir: {
      type: 'string',
      description: 'Working directory used by the lookup bridge.',
    },
    target: {
      type: 'string',
      description: 'Log target: auto, md, kmc, or summary.',
    },
    max_lines: {
      type: 'integer',
      description: 'Maximum number of log lines to return.',
    },
    limit: {
      type: 'integer',
      description: 'Maximum number of entries to return.',
    },
    dry_run: {
      type: 'boolean',
      description: 'Run without launching external solvers.',
    },
    overwrite_existing: {
      type: 'boolean',
      description: 'Delete and recreate an existing run directory for the same job id.',
    },
    validate: {
      type: 'boolean',
      description: 'Validate the generated KMC lookup with misa-kmc.',
    },
  },
};

const MIET_INSPECT_RUN_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    run_dir: {
      type: 'string',
      description: 'Run directory to inspect.',
    },
  },
  required: ['run_dir'],
};

export const MIET_SLASH_COMMAND_DEFINITIONS = [
  { command: '/status', summary: 'Show the current shell session, model, and latest run context.' },
  { command: '/tools', summary: 'List the built-in materials tools that this shell can call directly.' },
  { command: '/runs', summary: 'List recent workflow runs.' },
  { command: '/inspect <run-id>', summary: 'Inspect a specific run and summarize its current state.' },
  { command: '/draft <natural language task>', summary: 'Draft a workflow from natural language without launching it.' },
  { command: '/run <natural language task>', summary: 'Prepare or launch a workflow from natural language.' },
  { command: '/moire-run <MoRe-case-dir> [workdir]', summary: 'Run a MoRe LAMMPS NEB case and bridge it into KMC.' },
  { command: '/moire-compare <MoRe-case-dir> <event-a.json> <event-b.json> [event-c.json ...] [workdir]', summary: 'Compare multiple MoRe events against one case.' },
  { command: '/moire-diffusion-sweep <event.json> <MoRe-case-dir> [workdir]', summary: 'Sweep a MoRe event over diffusion conditions.' },
  { command: '/artifacts <run-id>', summary: 'List archived artifacts for a run.' },
  { command: '/logs <run-id> [md|kmc|summary]', summary: 'Read recent log excerpts for a run.' },
];

export function listMietCommandHints() {
  return MIET_SLASH_COMMAND_DEFINITIONS.map((item) => item.command);
}

export function listMietCommandDetails() {
  return MIET_SLASH_COMMAND_DEFINITIONS.map((item) => ({
    command: item.command,
    summary: item.summary,
  }));
}

export function buildMietRunCommandActions(runId, {
  includeInspect = true,
  includeSummaryLog = true,
  includeKmcLog = true,
} = {}) {
  if (!runId) return [];
  return [
    includeInspect
      ? { command: `/inspect ${runId}`, label: 'Prepare inspect command' }
      : null,
    includeSummaryLog
      ? { command: `/logs ${runId} summary`, label: 'Prepare summary log command' }
      : null,
    includeKmcLog
      ? { command: `/logs ${runId} kmc`, label: 'Prepare KMC log command' }
      : null,
  ].filter(Boolean);
}

export function renderMietSlashCommandHelpText() {
  return [
    'Available slash commands:',
    ...listMietCommandDetails().map((item) => `${item.command} — ${item.summary}`),
  ].join('\n');
}

export const MIET_COMMAND_HINTS = listMietCommandHints();

export const MIET_WEB_HINT_CHIPS = [
  '你能做什么？',
  '/runs',
  '我希望你把 MoRe 的迁移能垒计算出来并把迁移能垒作为 KMC 软件的输入进行 KMC 模拟',
  '/draft Create a native MD to KMC vacancy diffusion job for "Chat Draft Demo" at 810 K owned by miet.',
  '/run Create a native MD to KMC vacancy diffusion job for "Chat Run Demo" at 812 K owned by miet with 7 images.',
];

export const MIET_PLUGIN_INFO = {
  id: 'miet-claw-sim',
  name: 'mietclaw agent',
  description: 'OpenClaw tools for multiscale materials simulation orchestration on top of the mietclaw control plane.',
};

export const MIET_WEB_MIME_TYPES = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.txt': 'text/plain; charset=utf-8',
};

export function parseMietWebServerArgs(argv) {
  const config = { host: '127.0.0.1', port: 4173 };
  for (let index = 0; index < argv.length; index += 1) {
    const current = argv[index];
    if (current === '--host' && argv[index + 1]) {
      config.host = argv[index + 1];
      index += 1;
    } else if (current === '--port' && argv[index + 1]) {
      config.port = Number(argv[index + 1]);
      index += 1;
    }
  }
  return config;
}

export async function readMietWebRequestBody(request, {
  maxBytes = 1024 * 1024,
} = {}) {
  const chunks = [];
  let total = 0;

  for await (const chunk of request) {
    total += chunk.length;
    if (total > maxBytes) {
      throw new Error('Request body too large');
    }
    chunks.push(chunk);
  }

  if (chunks.length === 0) return {};
  const raw = Buffer.concat(chunks).toString('utf-8');
  return raw ? JSON.parse(raw) : {};
}

export function tokenizeMietCommand(text) {
  const tokens = [];
  const pattern = /"([^"]*)"|'([^']*)'|(\S+)/g;
  for (const match of String(text ?? '').matchAll(pattern)) {
    tokens.push(match[1] ?? match[2] ?? match[3]);
  }
  return tokens;
}

function slugify(value = 'job') {
  return String(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'job';
}

export function joinPythonPath(projectRoot, currentValue = '') {
  const parts = [path.join(projectRoot, 'src')];
  if (currentValue) parts.push(currentValue);
  return parts.join(path.delimiter);
}

export function resolveMaybePath(baseDir, value) {
  if (!value || typeof value !== 'string') return value;
  return path.isAbsolute(value) ? value : path.resolve(baseDir, value);
}

export function detectProjectRoot(startDirs = []) {
  const tried = new Set();
  for (const start of startDirs.filter(Boolean)) {
    let cursor = path.resolve(start);
    while (!tried.has(cursor)) {
      tried.add(cursor);
      if (fs.existsSync(path.join(cursor, 'src', 'miet_claw', 'cli.py'))) {
        return cursor;
      }
      const parent = path.dirname(cursor);
      if (parent === cursor) break;
      cursor = parent;
    }
  }
  throw new Error('Could not locate mietclaw project root from plugin context. Set plugin config projectRoot.');
}

export function resolvePluginContext(api) {
  const pluginConfig = api?.pluginConfig ?? {};
  const projectRoot = resolveMaybePath(
    process.cwd(),
    pluginConfig.projectRoot,
  ) || detectProjectRoot([
    api?.rootDir,
    process.cwd(),
  ]);

  return {
    projectRoot,
    pythonBin: typeof pluginConfig.pythonBin === 'string' && pluginConfig.pythonBin.trim() ? pluginConfig.pythonBin.trim() : 'python3',
    defaultOutputDir: resolveMaybePath(projectRoot, pluginConfig.defaultOutputDir || 'runs'),
    defaultWorkspaceRoot: resolveMaybePath(projectRoot, pluginConfig.defaultWorkspaceRoot || '.autonomy-web'),
  };
}

export function createMietWebApiContext({
  projectRoot = null,
  pythonBin = 'python3',
  outputDir = 'runs',
  workspaceRoot = '.autonomy-web',
} = {}) {
  const resolvedProjectRoot = resolveMaybePath(process.cwd(), projectRoot) || detectProjectRoot([process.cwd()]);
  const rootDir = path.join(resolvedProjectRoot, 'packages', 'openclaw-miet-claw-plugin');

  return {
    rootDir,
    pluginConfig: {
      projectRoot: resolvedProjectRoot,
      pythonBin,
      defaultOutputDir: resolveMaybePath(resolvedProjectRoot, outputDir) || path.join(resolvedProjectRoot, 'runs'),
      defaultWorkspaceRoot: resolveMaybePath(resolvedProjectRoot, workspaceRoot) || path.join(resolvedProjectRoot, '.autonomy-web'),
    },
  };
}

function resolveTemplateCatalogRoot(context) {
  return path.join(context.projectRoot, 'examples', 'jobs');
}

function resolveJobSpecPathValue(baseDir, value) {
  if (typeof value !== 'string') return value;
  if (value.startsWith('.') || value.includes('/')) {
    return path.resolve(baseDir, value);
  }
  return value;
}

export function resolveMietJobSpecPaths(api, jobSpec, { templatePath = null } = {}) {
  if (!jobSpec || typeof jobSpec !== 'object') return jobSpec;
  const context = resolvePluginContext(api);
  const nextSpec = JSON.parse(JSON.stringify(jobSpec));
  const templateBase = templatePath
    ? path.dirname(resolveMaybePath(context.projectRoot, templatePath))
    : context.projectRoot;

  if (nextSpec.md) {
    if (nextSpec.md.barriers_source) {
      nextSpec.md.barriers_source = resolveJobSpecPathValue(templateBase, nextSpec.md.barriers_source);
    }
    if (nextSpec.md.working_dir) {
      nextSpec.md.working_dir = resolveJobSpecPathValue(templateBase, nextSpec.md.working_dir);
    }
    if (Array.isArray(nextSpec.md.command)) {
      nextSpec.md.command = nextSpec.md.command.map((item) => resolveJobSpecPathValue(templateBase, item));
    }
  }

  if (nextSpec.kmc) {
    if (Array.isArray(nextSpec.kmc.command)) {
      nextSpec.kmc.command = nextSpec.kmc.command.map((item) => resolveJobSpecPathValue(templateBase, item));
    }
    const template = nextSpec.kmc.template ?? {};
    if (template.cluster_xyz) {
      template.cluster_xyz = resolveJobSpecPathValue(templateBase, template.cluster_xyz);
    }
    if (Array.isArray(template.potential_assets)) {
      template.potential_assets = template.potential_assets.map((item) => resolveJobSpecPathValue(templateBase, item));
    }
  }

  return nextSpec;
}

export async function listMietTemplates(api) {
  const context = resolvePluginContext(api);
  const templatesRoot = resolveTemplateCatalogRoot(context);
  const entries = await fsp.readdir(templatesRoot, { withFileTypes: true });
  const jsonFiles = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
    .map((entry) => entry.name);

  const templates = await Promise.all(
    jsonFiles.map(async (fileName) => {
      const absolutePath = path.join(templatesRoot, fileName);
      const spec = JSON.parse(await fsp.readFile(absolutePath, 'utf-8'));
      return {
        id: fileName,
        path: path.posix.join('examples', 'jobs', fileName),
        fileName,
        jobId: spec.job_id,
        mode: spec.mode,
        materialName: spec.material_system?.name ?? 'Unnamed material system',
        isNative: fileName.includes('_native'),
        spec,
      };
    }),
  );

  return templates.sort((left, right) => {
    const leftScore = left.fileName === 'md_to_kmc_chain_native.json' ? 0 : left.isNative ? 1 : 2;
    const rightScore = right.fileName === 'md_to_kmc_chain_native.json' ? 0 : right.isNative ? 1 : 2;
    if (leftScore !== rightScore) return leftScore - rightScore;
    return left.fileName.localeCompare(right.fileName);
  });
}

export async function materializeJobSpec({ projectRoot, outputDir, jobSpecPath, jobSpec, jobNameHint }) {
  if (jobSpecPath) {
    const resolved = resolveMaybePath(projectRoot, jobSpecPath);
    if (!fs.existsSync(resolved)) {
      throw new Error(`job_spec_path not found: ${resolved}`);
    }
    return resolved;
  }

  if (!jobSpec || typeof jobSpec !== 'object' || Array.isArray(jobSpec)) {
    throw new Error('Provide either job_spec_path or inline job_spec.');
  }

  const requestDir = path.join(outputDir, '_requests');
  await fsp.mkdir(requestDir, { recursive: true });
  const fileName = `${slugify(jobNameHint || jobSpec.job_id || 'job')}-${Date.now()}.json`;
  const storedPath = path.join(requestDir, fileName);
  await fsp.writeFile(storedPath, JSON.stringify(jobSpec, null, 2), 'utf-8');
  return storedPath;
}

export function runCommand(command, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command[0], command.slice(1), {
      cwd: options.cwd,
      env: options.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      stdout += String(chunk);
    });

    child.stderr.on('data', (chunk) => {
      stderr += String(chunk);
    });

    child.on('error', reject);
    child.on('close', (code) => {
      resolve({ code, stdout, stderr });
    });
  });
}

export async function runJsonCommand(command, args, {
  cwd = process.cwd(),
  env = process.env,
} = {}) {
  const result = await runCommand([command, ...args], { cwd, env });
  if (result.code !== 0) {
    throw new Error(result.stderr || result.stdout || `Command failed: ${command} ${args.join(' ')}`);
  }
  return JSON.parse(result.stdout);
}

export async function getMietSystemStatus(api, {
  openclawCommand = 'openclaw',
} = {}) {
  const context = resolvePluginContext(api);
  const expectedToolNames = listMietClawToolNames();
  try {
    const [gateway, plugin] = await Promise.all([
      runJsonCommand(openclawCommand, ['gateway', 'status', '--json'], { cwd: context.projectRoot }),
      runJsonCommand(openclawCommand, ['plugins', 'inspect', MIET_PLUGIN_INFO.id, '--json'], { cwd: context.projectRoot }),
    ]);

    return {
      gateway: {
        healthy: gateway.health?.healthy ?? false,
        runtimeStatus: gateway.service?.runtime?.status ?? 'unknown',
        rpcOk: gateway.rpc?.ok ?? false,
        port: gateway.gateway?.port ?? null,
      },
      plugin: {
        id: plugin.plugin?.id ?? MIET_PLUGIN_INFO.id,
        name: plugin.plugin?.name ?? MIET_PLUGIN_INFO.name,
        status: plugin.plugin?.status ?? 'unknown',
        toolNames: plugin.plugin?.toolNames?.length ? plugin.plugin.toolNames : expectedToolNames,
      },
    };
  } catch (error) {
    return {
      gateway: {
        healthy: false,
        runtimeStatus: 'unavailable',
        rpcOk: false,
        port: null,
        error: String(error.message || error),
      },
      plugin: {
        id: MIET_PLUGIN_INFO.id,
        name: MIET_PLUGIN_INFO.name,
        status: 'unavailable',
        toolNames: expectedToolNames,
      },
    };
  }
}

export async function fetchMietLocalModelJson(resourcePath, options = {}) {
  const localModelBaseUrl = process.env.MIETCLAW_LOCAL_MODEL_BASE_URL ?? 'http://127.0.0.1:8000';
  const localModelApiKey = process.env.MIETCLAW_LOCAL_MODEL_API_KEY ?? 'omlx-local';
  const response = await fetch(`${localModelBaseUrl}${resourcePath}`, {
    ...options,
    headers: {
      authorization: `Bearer ${localModelApiKey}`,
      'content-type': 'application/json',
      ...(options.headers ?? {}),
    },
  });

  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(data?.error?.message || data?.detail || `Local model request failed: ${response.status}`);
  }
  return data;
}

export async function getMietLocalModelStatus() {
  const localModelBaseUrl = process.env.MIETCLAW_LOCAL_MODEL_BASE_URL ?? 'http://127.0.0.1:8000';
  try {
    const [health, models] = await Promise.all([
      fetch(`${localModelBaseUrl}/health`).then(async (response) => {
        const text = await response.text();
        return text ? JSON.parse(text) : {};
      }),
      fetchMietLocalModelJson('/v1/models'),
    ]);

    return {
      healthy: health.status === 'healthy',
      defaultModel: health.default_model ?? models.data?.[0]?.id ?? null,
      models: (models.data ?? []).map((item) => item.id),
      baseUrl: localModelBaseUrl,
    };
  } catch (error) {
    return {
      healthy: false,
      defaultModel: null,
      models: [],
      baseUrl: localModelBaseUrl,
      error: String(error.message || error),
    };
  }
}

export async function runMietCli(api, {
  subcommand,
  jobSpecPath,
  jobSpec,
  jobNameHint,
  outputDir,
  dryRun,
  runDir,
  resume = false,
  overwriteExisting = false,
}) {
  const context = resolvePluginContext(api);
  const finalOutputDir = resolveMaybePath(context.projectRoot, outputDir) || context.defaultOutputDir;

  let resolvedJobSpecPath = jobSpecPath;
  let effectiveOutputDir = finalOutputDir;

  if (!resolvedJobSpecPath && runDir) {
    const resolvedRunDir = resolveMaybePath(context.projectRoot, runDir);
    resolvedJobSpecPath = path.join(resolvedRunDir, 'job_spec.resolved.json');
    effectiveOutputDir = path.dirname(resolvedRunDir);
  }

  const materializedSpecPath = await materializeJobSpec({
    projectRoot: context.projectRoot,
    outputDir: effectiveOutputDir,
    jobSpecPath: resolvedJobSpecPath,
    jobSpec,
    jobNameHint,
  });

  const command = [
    context.pythonBin,
    '-m',
    'miet_claw.cli',
    subcommand,
  ];

  if (subcommand === 'plan' || subcommand === 'run') {
    command.push(materializedSpecPath);
  }

  if (subcommand === 'run') {
    command.push('--output-dir', effectiveOutputDir);
    if (dryRun) command.push('--dry-run');
    if (resume) command.push('--resume');
    if (overwriteExisting) command.push('--overwrite-existing');
  }

  const env = {
    ...process.env,
    PYTHONPATH: joinPythonPath(context.projectRoot, process.env.PYTHONPATH),
  };

  const result = await runCommand(command, {
    cwd: context.projectRoot,
    env,
  });

  if (result.code !== 0) {
    throw new Error(`mietclaw CLI failed (${result.code}): ${result.stderr || result.stdout}`.trim());
  }

  const stdout = result.stdout.trim();
  const output = {
    command,
    cwd: context.projectRoot,
    stdout,
    stderr: result.stderr.trim(),
    jobSpecPath: materializedSpecPath,
    outputDir: effectiveOutputDir,
  };

  if (subcommand === 'run') {
    output.runDir = stdout.split(/\r?\n/).filter(Boolean).at(-1);
  } else if (subcommand === 'plan') {
    output.plan = JSON.parse(stdout);
  }

  return output;
}

export async function runMietAutonomyCli(api, {
  subcommand,
  prompt,
  provider,
  outputDir,
  workspaceRoot,
  mode,
  templatePath,
  jobNameHint,
  materialName,
  dryRun,
  resumeExisting,
  overwriteExisting,
}) {
  const context = resolvePluginContext(api);
  const finalOutputDir = resolveMaybePath(context.projectRoot, outputDir) || context.defaultOutputDir;
  const command = [
    context.pythonBin,
    '-m',
    'miet_claw.cli',
    subcommand,
    prompt,
    '--provider',
    provider || 'auto',
    '--project-root',
    context.projectRoot,
    '--workspace-root',
    resolveMaybePath(context.projectRoot, workspaceRoot || path.join(context.projectRoot, '.autonomy')),
  ];

  if (mode) command.push('--mode', mode);
  if (templatePath) command.push('--template-path', resolveMaybePath(context.projectRoot, templatePath));
  if (jobNameHint) command.push('--job-id', jobNameHint);
  if (materialName) command.push('--material-name', materialName);
  if (subcommand === 'autonomy-run') {
    command.push('--output-dir', finalOutputDir);
    if (dryRun) command.push('--dry-run-only');
    if (resumeExisting) command.push('--resume-existing');
    if (overwriteExisting) command.push('--overwrite-existing');
  }

  const env = {
    ...process.env,
    PYTHONPATH: joinPythonPath(context.projectRoot, process.env.PYTHONPATH),
  };
  const result = await runCommand(command, {
    cwd: context.projectRoot,
    env,
  });

  if (result.code !== 0) {
    throw new Error(`mietclaw autonomy CLI failed (${result.code}): ${result.stderr || result.stdout}`.trim());
  }

  const payload = JSON.parse(result.stdout.trim());
  return {
    command,
    cwd: context.projectRoot,
    stdout: result.stdout.trim(),
    stderr: result.stderr.trim(),
    payload,
    workspaceDir: payload.workspace_dir,
    runDir: payload.execution?.final_run_dir || null,
  };
}

async function materializeChatHistory(outputDir, historyMessages) {
  if (!Array.isArray(historyMessages) || historyMessages.length === 0) {
    return null;
  }
  const requestDir = path.join(outputDir, '_requests');
  await fsp.mkdir(requestDir, { recursive: true });
  const historyPath = path.join(requestDir, `chat-history-${Date.now()}.json`);
  await fsp.writeFile(historyPath, JSON.stringify(historyMessages, null, 2), 'utf-8');
  return historyPath;
}

export function normalizeChatPayload(payload = {}) {
  const rawMessage = payload.message && typeof payload.message === 'object' ? payload.message : null;
  const progress = Array.isArray(rawMessage?.progress)
    ? rawMessage.progress
    : (Array.isArray(payload.progress) ? payload.progress : []);
  const rawSession = rawMessage?.session && typeof rawMessage.session === 'object'
    ? rawMessage.session
    : (payload.session && typeof payload.session === 'object' ? payload.session : {});
  const rawCurrent = rawMessage?.current && typeof rawMessage.current === 'object'
    ? rawMessage.current
    : (payload.current && typeof payload.current === 'object' ? payload.current : {});

  const session = {
    transcriptPath: rawSession.transcriptPath ?? rawSession.transcript_path ?? rawMessage?.transcriptPath ?? payload.transcript_path ?? null,
    selectedModel: rawSession.selectedModel ?? rawSession.selected_model ?? payload.selected_model ?? rawMessage?.model ?? null,
    approvalPolicy: rawSession.approvalPolicy ?? rawSession.approval_policy ?? payload.approval_policy ?? 'allow_all',
    historyLength: rawSession.historyLength ?? rawSession.history_length ?? payload.history_length ?? null,
  };

  const current = {
    activeKind: rawCurrent.activeKind ?? rawCurrent.active_kind ?? payload.current_context_kind ?? null,
    hasAny: Boolean(
      rawCurrent.hasAny
      ?? rawCurrent.has_any
      ?? rawCurrent.activeKind
      ?? rawCurrent.active_kind
      ?? payload.current_context_kind
      ?? rawCurrent.runDir
      ?? rawCurrent.run_dir
      ?? payload.current_run_dir
      ?? rawCurrent.report
      ?? payload.current_report
      ?? rawMessage?.currentReport
      ?? rawCurrent.bridgeSummary
      ?? rawCurrent.bridge_summary
      ?? payload.current_bridge_summary
      ?? rawMessage?.currentBridgeSummary
      ?? rawCurrent.moireSummary
      ?? rawCurrent.moire_summary
      ?? payload.current_moire_summary
      ?? rawMessage?.currentMoireSummary
      ?? rawCurrent.moireCompareSummary
      ?? rawCurrent.moire_compare_summary
      ?? payload.current_moire_compare_summary
      ?? rawMessage?.currentMoireCompareSummary
      ?? rawCurrent.moireDiffusionSummary
      ?? rawCurrent.moire_diffusion_summary
      ?? payload.current_moire_diffusion_summary
      ?? rawMessage?.currentMoireDiffusionSummary
      ?? rawMessage?.currentRunDetail?.runDir
    ),
    runDir: rawCurrent.runDir ?? rawCurrent.run_dir ?? rawMessage?.currentRunDetail?.runDir ?? payload.current_run_dir ?? null,
    report: rawCurrent.report ?? rawMessage?.currentReport ?? payload.current_report ?? null,
    bridgeSummary: rawCurrent.bridgeSummary ?? rawCurrent.bridge_summary ?? rawMessage?.currentBridgeSummary ?? payload.current_bridge_summary ?? null,
    moireSummary: rawCurrent.moireSummary ?? rawCurrent.moire_summary ?? rawMessage?.currentMoireSummary ?? payload.current_moire_summary ?? null,
    moireCompareSummary: rawCurrent.moireCompareSummary ?? rawCurrent.moire_compare_summary ?? rawMessage?.currentMoireCompareSummary ?? payload.current_moire_compare_summary ?? null,
    moireDiffusionSummary: rawCurrent.moireDiffusionSummary ?? rawCurrent.moire_diffusion_summary ?? rawMessage?.currentMoireDiffusionSummary ?? payload.current_moire_diffusion_summary ?? null,
  };

  const usedTools = payload.used_tools ?? (rawMessage?.kind ? rawMessage.kind === 'tool' : (progress.length > 0 || current.hasAny));
  const kind = rawMessage?.kind ?? payload.kind ?? (usedTools ? 'tool' : 'chat');
  const reply = rawMessage?.content ?? payload.reply ?? '';
  const toolTraceSummary = rawMessage?.toolTraceSummary ?? payload.tool_trace_summary ?? payload.toolTraceSummary ?? null;
  const toolEvidence = Array.isArray(rawMessage?.toolEvidence)
    ? rawMessage.toolEvidence
    : (Array.isArray(payload.tool_evidence) ? payload.tool_evidence : (Array.isArray(payload.toolEvidence) ? payload.toolEvidence : []));
  const toolTraceReplay = Array.isArray(rawMessage?.toolTraceReplay)
    ? rawMessage.toolTraceReplay
    : (Array.isArray(payload.tool_trace_replay) ? payload.tool_trace_replay : (Array.isArray(payload.toolTraceReplay) ? payload.toolTraceReplay : []));
  const toolTimeline = Array.isArray(rawMessage?.toolTimeline)
    ? rawMessage.toolTimeline
    : (Array.isArray(payload.tool_timeline) ? payload.tool_timeline : (Array.isArray(payload.toolTimeline) ? payload.toolTimeline : buildToolTimelineFromReplay(toolTraceReplay)));
  const toolTraceId = rawMessage?.toolTraceId ?? payload.tool_trace_id ?? payload.toolTraceId ?? null;
  const message = rawMessage ?? buildAssistantMessage({
    content: reply,
    kind,
    model: kind === 'chat' ? session.selectedModel ?? null : null,
    progress,
    meta: buildRuntimeMessageMeta({
      session,
      current,
      currentReport: current.report ?? null,
      currentBridgeSummary: current.bridgeSummary ?? null,
      currentMoireSummary: current.moireSummary ?? null,
      currentMoireCompareSummary: current.moireCompareSummary ?? null,
      currentMoireDiffusionSummary: current.moireDiffusionSummary ?? null,
      transcriptPath: session.transcriptPath ?? null,
      toolTraceSummary,
      toolEvidence,
      toolTraceReplay,
      toolTimeline,
      toolTraceId,
    }),
  });

  return {
    reply,
    kind,
    usedTools: Boolean(usedTools),
    progress,
    session,
    current,
    message,
    raw: payload,
  };
}

function deriveCurrentActiveKind({
  current = null,
  currentReport = null,
  currentRunDetail = null,
  currentBridgeSummary = null,
  currentMoireSummary = null,
  currentMoireCompareSummary = null,
  currentMoireDiffusionSummary = null,
}) {
  if (current?.activeKind) return current.activeKind;
  if (currentRunDetail?.runDir) return 'run';
  if (currentMoireDiffusionSummary) return 'moire_diffusion_summary';
  if (currentMoireCompareSummary) return 'moire_compare_summary';
  if (currentMoireSummary) return 'moire_summary';
  if (currentBridgeSummary) return 'bridge_summary';
  if (currentReport) return 'report';
  return null;
}

export function buildToolTimelineFromReplay(replay = []) {
  if (!Array.isArray(replay)) return [];
  return replay.map((item) => {
    const kind = item?.kind ?? 'event';
    if (kind === 'assistant_action_block') {
      const toolActions = Array.isArray(item.toolActions) ? item.toolActions.filter(Boolean) : [];
      return {
        index: item.index ?? null,
        kind,
        stage: 'assistant',
        status: 'info',
        title: 'Assistant normalized its next action',
        detail: toolActions.length
          ? `Planned tools: ${toolActions.join(', ')}`
          : 'No tool actions were proposed in this assistant block.',
        source: item.source ?? null,
        toolActions,
        finalAnswer: item.finalAnswer ?? null,
      };
    }
    if (kind === 'tool_use') {
      return {
        index: item.index ?? null,
        kind,
        stage: 'tool_request',
        status: 'running',
        title: `Requested tool: ${item.action ?? 'unknown'}`,
        detail: item.manual ? 'This step was issued manually.' : 'This step was issued automatically by the agent.',
        source: item.source ?? null,
        action: item.action ?? null,
        params: item.params ?? {},
      };
    }
    if (kind === 'permission_decision') {
      return {
        index: item.index ?? null,
        kind,
        stage: 'permission',
        status: item.decision === 'allow' ? 'completed' : 'blocked',
        title: `Permission check: ${item.action ?? 'unknown'}`,
        detail: item.reason ?? `Decision: ${item.decision ?? 'unknown'}`,
        source: item.source ?? null,
        action: item.action ?? null,
        decision: item.decision ?? null,
      };
    }
    if (kind === 'tool_result_block') {
      return {
        index: item.index ?? null,
        kind,
        stage: 'tool_result',
        status: item.ok === false ? 'failed' : 'completed',
        title: `Normalized tool result: ${item.action ?? 'unknown'}`,
        detail: item.outputPreview ?? 'No structured tool output preview was recorded.',
        source: item.source ?? null,
        action: item.action ?? null,
        params: item.params ?? {},
        requestId: item.requestId ?? null,
      };
    }
    if (kind === 'turn_finish') {
      return {
        index: item.index ?? null,
        kind,
        stage: 'finish',
        status: item.status === 'finish' ? 'completed' : (item.status === 'error' ? 'failed' : 'info'),
        title: 'Assistant finished this tool-backed turn',
        detail: item.reason ?? item.reply ?? 'The turn finished without an explicit reason.',
        source: item.source ?? null,
        finishStatus: item.status ?? null,
        reply: item.reply ?? null,
      };
    }
    return {
      index: item.index ?? null,
      kind,
      stage: 'event',
      status: 'info',
      title: kind,
      detail: 'A trace event was recorded.',
      source: item.source ?? null,
    };
  });
}

export async function loadMietTranscriptEvidence({
  transcriptPath,
  traceId = null,
  eventIndex = null,
} = {}) {
  if (!transcriptPath) {
    throw new Error('transcriptPath is required');
  }
  const transcriptText = await readTextIfExists(transcriptPath);
  if (!transcriptText) {
    throw new Error(`Transcript not found: ${transcriptPath}`);
  }

  const regex = /### tool trace\s*\n\s*```json\n([\s\S]*?)\n```/g;
  const matches = [];
  let match;
  while ((match = regex.exec(transcriptText)) !== null) {
    const jsonText = match[1];
    try {
      const parsed = JSON.parse(jsonText);
      const jsonOffset = match.index + match[0].indexOf(jsonText);
      matches.push({ parsed, jsonText, jsonOffset });
    } catch {
      // ignore malformed blocks and keep scanning
    }
  }

  if (!matches.length) {
    throw new Error('No tool trace sections were found in the transcript');
  }

  const selected = traceId
    ? matches.find((item) => item.parsed?.traceId === traceId)
    : matches[matches.length - 1];
  if (!selected) {
    throw new Error(`Tool trace not found in transcript: ${traceId}`);
  }

  const replay = Array.isArray(selected.parsed?.replay) ? selected.parsed.replay : [];
  const event = eventIndex == null
    ? null
    : replay.find((item) => Number(item?.index) === Number(eventIndex));
  if (eventIndex != null && !event) {
    throw new Error(`Trace event ${eventIndex} not found in transcript`);
  }

  const blockLines = String(selected.jsonText ?? '').split('\n');
  const marker = eventIndex == null
    ? (selected.parsed?.traceId ? `"traceId": "${selected.parsed.traceId}"` : null)
    : `"index": ${Number(eventIndex)}`;
  const markerLine = marker
    ? blockLines.findIndex((line) => line.includes(marker))
    : -1;
  const excerptStart = markerLine >= 0 ? Math.max(0, markerLine - 3) : 0;
  const excerptEnd = markerLine >= 0 ? Math.min(blockLines.length, markerLine + 9) : Math.min(blockLines.length, 12);
  const excerpt = blockLines.slice(excerptStart, excerptEnd).join('\n');
  const lineOffset = transcriptText.slice(0, selected.jsonOffset).split('\n').length;

  return {
    transcriptPath,
    traceId: selected.parsed?.traceId ?? null,
    eventIndex: eventIndex == null ? null : Number(eventIndex),
    summary: selected.parsed?.summary ?? null,
    event: event ?? null,
    excerpt,
    lineStart: lineOffset + excerptStart,
    lineEnd: lineOffset + excerptEnd - 1,
  };
}

export function buildRuntimeMessageMeta({
  session = null,
  current = null,
  currentReport = null,
  currentPlan = null,
  currentReportNotes = null,
  currentRunDetail = null,
  currentBridgeSummary = null,
  currentMoireSummary = null,
  currentMoireCompareSummary = null,
  currentMoireDiffusionSummary = null,
  transcriptPath = null,
  toolTraceSummary = null,
  toolEvidence = [],
  toolTraceReplay = [],
  toolTimeline = [],
  toolTraceId = null,
} = {}) {
  const normalizedSession = session
    ? {
        transcriptPath: session.transcriptPath ?? session.transcript_path ?? transcriptPath ?? null,
        selectedModel: session.selectedModel ?? session.selected_model ?? null,
        approvalPolicy: session.approvalPolicy ?? session.approval_policy ?? 'allow_all',
        historyLength: session.historyLength ?? session.history_length ?? null,
      }
    : (
        transcriptPath
          ? {
              transcriptPath,
              selectedModel: null,
              approvalPolicy: 'allow_all',
              historyLength: null,
            }
          : null
      );

  const activeKind = deriveCurrentActiveKind({
    current,
    currentReport,
    currentRunDetail,
    currentBridgeSummary,
    currentMoireSummary,
    currentMoireCompareSummary,
    currentMoireDiffusionSummary,
  });

  const normalizedCurrent = {
    activeKind,
    hasAny: Boolean(
      current?.hasAny
      ?? current?.has_any
      ?? activeKind
      ?? current?.runDir
      ?? current?.run_dir
      ?? currentRunDetail?.runDir
      ?? currentReport
      ?? currentBridgeSummary
      ?? currentMoireSummary
      ?? currentMoireCompareSummary
      ?? currentMoireDiffusionSummary
    ),
    runDir: current?.runDir ?? current?.run_dir ?? currentRunDetail?.runDir ?? null,
    report: current?.report ?? currentReport ?? null,
    bridgeSummary: current?.bridgeSummary ?? current?.bridge_summary ?? currentBridgeSummary ?? null,
    moireSummary: current?.moireSummary ?? current?.moire_summary ?? currentMoireSummary ?? null,
    moireCompareSummary: current?.moireCompareSummary ?? current?.moire_compare_summary ?? currentMoireCompareSummary ?? null,
    moireDiffusionSummary: current?.moireDiffusionSummary ?? current?.moire_diffusion_summary ?? currentMoireDiffusionSummary ?? null,
  };

  return {
    session: normalizedSession,
    current: normalizedCurrent,
    currentReport: normalizedCurrent.report,
    currentPlan: currentPlan ?? null,
    currentReportNotes: currentReportNotes ?? null,
    currentRunDetail: currentRunDetail ?? null,
    currentBridgeSummary: normalizedCurrent.bridgeSummary,
    currentMoireSummary: normalizedCurrent.moireSummary,
    currentMoireCompareSummary: normalizedCurrent.moireCompareSummary,
    currentMoireDiffusionSummary: normalizedCurrent.moireDiffusionSummary,
    transcriptPath: normalizedSession?.transcriptPath ?? null,
    toolTraceSummary: toolTraceSummary ?? null,
    toolEvidence: Array.isArray(toolEvidence) ? toolEvidence : [],
    toolTraceReplay: Array.isArray(toolTraceReplay) ? toolTraceReplay : [],
    toolTimeline: Array.isArray(toolTimeline) && toolTimeline.length ? toolTimeline : buildToolTimelineFromReplay(toolTraceReplay),
    toolTraceId: toolTraceId ?? null,
  };
}

export function buildMessageCards({
  meta = {},
  previewConfirmable = false,
  previewExecuted = false,
  jobSpecPath = null,
} = {}) {
  const cards = [];
  const resolvedJobSpecPath = jobSpecPath ?? meta.currentReport?.generated_files?.job_spec ?? null;
  const currentPlanSteps = Array.isArray(meta.currentPlan)
    ? meta.currentPlan
    : (Array.isArray(meta.currentPlan?.plan) ? meta.currentPlan.plan : []);

  if (meta.currentReport) {
    cards.push({
      type: 'transparency',
      report: meta.currentReport,
      plan: meta.currentPlan ?? null,
      notes: meta.currentReportNotes ?? null,
      previewConfirmable: Boolean(previewConfirmable && resolvedJobSpecPath && !previewExecuted),
      previewExecuted: Boolean(previewExecuted),
      jobSpecPath: resolvedJobSpecPath,
      transcriptPath: meta.transcriptPath ?? null,
      toolTraceSummary: meta.toolTraceSummary ?? null,
      toolEvidence: meta.toolEvidence ?? [],
      toolTraceReplay: meta.toolTraceReplay ?? [],
      toolTraceId: meta.toolTraceId ?? null,
    });
  }

  if (!meta.currentReport && currentPlanSteps.length) {
    cards.push({
      type: 'plan_result',
      plan: meta.currentPlan,
    });
  }

  if (Array.isArray(meta.toolTimeline) && meta.toolTimeline.length) {
    cards.push({
      type: 'tool_timeline',
      timeline: meta.toolTimeline,
      transcriptPath: meta.transcriptPath ?? null,
      toolTraceId: meta.toolTraceId ?? null,
    });
  }

  if (meta.session || meta.current || meta.transcriptPath) {
    cards.push({
      type: 'runtime_snapshot',
      session: meta.session ?? null,
      current: meta.current ?? null,
      transcriptPath: meta.transcriptPath ?? null,
      toolTraceSummary: meta.toolTraceSummary ?? null,
      toolEvidence: meta.toolEvidence ?? [],
      toolTraceReplay: meta.toolTraceReplay ?? [],
      toolTimeline: meta.toolTimeline ?? [],
      toolTraceId: meta.toolTraceId ?? null,
    });
  }

  if (meta.currentRunDetail) {
    cards.push({
      type: 'run_result',
      detail: meta.currentRunDetail,
      actions: meta.currentRunDetail.commandActions ?? [],
    });
  }

  return cards;
}

export function buildAssistantMessage({
  content,
  kind = 'chat',
  model = null,
  progress = [],
  meta = {},
  previewConfirmable = false,
  previewExecuted = false,
  jobSpecPath = null,
} = {}) {
  const resolvedJobSpecPath = jobSpecPath ?? meta.currentReport?.generated_files?.job_spec ?? null;
  return {
    role: 'assistant',
    content: content ?? '',
    kind,
    model,
    progress: Array.isArray(progress) ? progress : [],
    session: meta.session ?? null,
    current: meta.current ?? null,
    currentReport: meta.currentReport ?? null,
    currentPlan: meta.currentPlan ?? null,
    currentReportNotes: meta.currentReportNotes ?? null,
    currentRunDetail: meta.currentRunDetail ?? null,
    currentBridgeSummary: meta.currentBridgeSummary ?? null,
    currentMoireSummary: meta.currentMoireSummary ?? null,
    currentMoireCompareSummary: meta.currentMoireCompareSummary ?? null,
    currentMoireDiffusionSummary: meta.currentMoireDiffusionSummary ?? null,
    transcriptPath: meta.transcriptPath ?? null,
    toolTraceSummary: meta.toolTraceSummary ?? null,
    toolEvidence: meta.toolEvidence ?? [],
    toolTraceReplay: meta.toolTraceReplay ?? [],
    toolTimeline: Array.isArray(meta.toolTimeline) ? meta.toolTimeline : buildToolTimelineFromReplay(meta.toolTraceReplay ?? []),
    toolTraceId: meta.toolTraceId ?? null,
    previewConfirmable: Boolean(previewConfirmable && resolvedJobSpecPath && !previewExecuted),
    previewExecuted: Boolean(previewExecuted),
    jobSpecPath: resolvedJobSpecPath,
    cards: buildMessageCards({
      meta,
      previewConfirmable,
      previewExecuted,
      jobSpecPath: resolvedJobSpecPath,
    }),
  };
}

export function buildToolResponsePayload({
  content,
  kind = 'tool',
  model = null,
  progress = [],
  meta = null,
  session = null,
  current = null,
  currentReport = null,
  currentPlan = null,
  currentReportNotes = null,
  currentRunDetail = null,
  currentBridgeSummary = null,
  currentMoireSummary = null,
  currentMoireCompareSummary = null,
  currentMoireDiffusionSummary = null,
  transcriptPath = null,
  toolTraceSummary = null,
  toolEvidence = [],
  toolTraceReplay = [],
  toolTimeline = [],
  toolTraceId = null,
  previewConfirmable = false,
  previewExecuted = false,
  jobSpecPath = null,
} = {}) {
  const runtimeMeta = meta ?? buildRuntimeMessageMeta({
    session,
    current,
    currentReport,
    currentPlan,
    currentReportNotes,
    currentRunDetail,
    currentBridgeSummary,
    currentMoireSummary,
    currentMoireCompareSummary,
    currentMoireDiffusionSummary,
    transcriptPath,
    toolTraceSummary,
    toolEvidence,
    toolTraceReplay,
    toolTimeline,
    toolTraceId,
  });

  return {
    kind,
    content: content ?? '',
    meta: runtimeMeta,
    message: buildAssistantMessage({
      content,
      kind,
      model,
      progress,
      meta: runtimeMeta,
      previewConfirmable,
      previewExecuted,
      jobSpecPath,
    }),
  };
}

async function readTextIfExists(filePath) {
  if (!filePath || !fs.existsSync(filePath)) return null;
  return await fsp.readFile(filePath, 'utf-8');
}

async function buildPlanFromJobSpec(api, { jobSpecPath = null, jobSpec = null, jobNameHint = null } = {}) {
  if (!api || (!jobSpecPath && !jobSpec)) return null;
  const output = await runMietCli(api, {
    subcommand: 'plan',
    jobSpecPath,
    jobSpec,
    jobNameHint,
  });
  return output.plan ?? null;
}

function buildRuntimeMetaFromNormalizedChatPayload(normalizedPayload, {
  currentReport = null,
  currentPlan = null,
  currentReportNotes = null,
  currentRunDetail = null,
} = {}) {
  const existing = normalizedPayload.message && typeof normalizedPayload.message === 'object'
    ? normalizedPayload.message
    : null;

  return buildRuntimeMessageMeta({
    session: normalizedPayload.session ?? null,
    current: normalizedPayload.current ?? null,
    currentReport: currentReport ?? existing?.currentReport ?? normalizedPayload.current?.report ?? null,
    currentPlan: currentPlan ?? existing?.currentPlan ?? null,
    currentReportNotes: currentReportNotes ?? existing?.currentReportNotes ?? null,
    currentRunDetail: currentRunDetail ?? existing?.currentRunDetail ?? null,
    currentBridgeSummary: existing?.currentBridgeSummary ?? normalizedPayload.current?.bridgeSummary ?? null,
    currentMoireSummary: existing?.currentMoireSummary ?? normalizedPayload.current?.moireSummary ?? null,
    currentMoireCompareSummary: existing?.currentMoireCompareSummary ?? normalizedPayload.current?.moireCompareSummary ?? null,
    currentMoireDiffusionSummary: existing?.currentMoireDiffusionSummary ?? normalizedPayload.current?.moireDiffusionSummary ?? null,
    transcriptPath: existing?.transcriptPath ?? normalizedPayload.session?.transcriptPath ?? null,
  });
}

export async function buildChatResponsePayload(api, {
  chatOutput = null,
  payload = null,
  modelFallback = null,
  loadRunDetail = null,
} = {}) {
  const context = api ? resolvePluginContext(api) : null;
  const effectiveLoadRunDetail = typeof loadRunDetail === 'function'
    ? loadRunDetail
    : (context ? (runDir) => readMietRunDetail(runDir, { projectRoot: context.projectRoot }) : null);
  const normalizedPayload = normalizeChatPayload(payload ?? chatOutput?.payload ?? {});
  const existing = normalizedPayload.message && typeof normalizedPayload.message === 'object'
    ? normalizedPayload.message
    : null;
  const currentReport = existing?.currentReport ?? normalizedPayload.current?.report ?? null;
  const shouldLoadRunDetail = !existing?.currentRunDetail
    && normalizedPayload.current?.runDir
    && typeof effectiveLoadRunDetail === 'function';
  const shouldLoadNotes = !existing?.currentReportNotes && currentReport?.generated_files?.notes;
  const shouldLoadPlan = !existing?.currentPlan && currentReport && api;
  const hasCards = Array.isArray(existing?.cards) && existing.cards.length > 0;

  const currentRunDetail = existing?.currentRunDetail
    ?? (shouldLoadRunDetail ? await effectiveLoadRunDetail(normalizedPayload.current.runDir).catch(() => null) : null);
  const currentReportNotes = existing?.currentReportNotes
    ?? (shouldLoadNotes ? await readTextIfExists(currentReport.generated_files.notes).catch(() => null) : null);
  const currentPlan = existing?.currentPlan
    ?? (shouldLoadPlan
      ? await buildPlanFromJobSpec(api, {
          jobSpecPath: currentReport.generated_files?.job_spec ?? null,
          jobSpec: currentReport.job_spec ?? null,
          jobNameHint: currentReport.job_id ?? null,
        }).catch(() => null)
      : null);
  const meta = buildRuntimeMetaFromNormalizedChatPayload(normalizedPayload, {
    currentReport,
    currentPlan,
    currentReportNotes,
    currentRunDetail,
  });

  const message = existing && !shouldLoadRunDetail && !shouldLoadNotes && !shouldLoadPlan && hasCards
    ? existing
    : buildAssistantMessage({
        content: existing?.content ?? normalizedPayload.reply,
        kind: existing?.kind ?? normalizedPayload.kind,
        model: (existing?.kind ?? normalizedPayload.kind) === 'chat'
          ? (existing?.model ?? normalizedPayload.session?.selectedModel ?? modelFallback ?? null)
          : null,
        progress: existing?.progress ?? normalizedPayload.progress ?? [],
        meta,
      });

  return {
    kind: normalizedPayload.kind,
    content: normalizedPayload.reply,
    message,
    meta,
    normalizedPayload,
  };
}

export async function hydrateMietChatPromptOutput(api, {
  prompt,
  provider = 'local',
  outputDir = null,
  workspaceRoot = null,
  mode = null,
  model = null,
  historyMessages = null,
  previewRuns = true,
  refreshRuns = false,
} = {}) {
  const context = resolvePluginContext(api);
  const chatOutput = await runMietChatCli(api, {
    prompt,
    provider,
    outputDir: outputDir ?? context.defaultOutputDir,
    workspaceRoot: workspaceRoot ?? context.defaultWorkspaceRoot,
    mode,
    model,
    historyMessages,
    previewRuns,
  });

  return {
    ...await buildChatResponsePayload(api, {
      chatOutput,
      modelFallback: model ?? null,
    }),
    refreshRuns,
  };
}

export function normalizeMietChatHistory(messages, {
  input = null,
} = {}) {
  let historyMessages = (Array.isArray(messages) ? messages : [])
    .filter((message) => ['user', 'assistant'].includes(message?.role))
    .map((message) => ({
      role: message.role,
      content: String(message.content ?? ''),
    }))
    .filter((message) => message.content);

  const lastMessage = historyMessages.at(-1);
  if (input && lastMessage?.role === 'user' && lastMessage.content.trim() === String(input).trim()) {
    historyMessages = historyMessages.slice(0, -1);
  }

  return historyMessages;
}

export async function buildMietChatApiPayload(api, {
  body = {},
  defaultOutputDir = null,
  defaultWorkspaceRoot = null,
} = {}) {
  const context = resolvePluginContext(api);
  const input = String(body.input ?? '').trim();
  if (!input) {
    return {
      statusCode: 400,
      payload: { ok: false, error: 'input is required' },
    };
  }

  const recentRuns = await listMietRuns(api);
  const localModel = await getMietLocalModelStatus();
  const previewRuns = body.previewRuns !== false;
  const outputDir = defaultOutputDir ?? context.defaultOutputDir;
  const workspaceRoot = defaultWorkspaceRoot ?? context.defaultWorkspaceRoot;

  if (input.startsWith('/')) {
    const result = await buildMietChatSlashCommandPayload(api, {
      input,
      previewRuns,
      provider: 'local',
      defaultOutputDir: outputDir,
      defaultWorkspaceRoot: workspaceRoot,
    });
    return {
      statusCode: 200,
      payload: {
        ok: true,
        mode: result.kind,
        message: result.message,
        localModel,
        runs: result.refreshRuns ? await listMietRuns(api) : recentRuns,
      },
    };
  }

  const historyMessages = normalizeMietChatHistory(body.messages, { input });
  const responsePayload = await hydrateMietChatPromptOutput(api, {
    prompt: input,
    provider: body.provider ?? 'local',
    outputDir,
    workspaceRoot,
    mode: body.mode || null,
    model: body.model ?? null,
    historyMessages,
    previewRuns,
    refreshRuns: true,
  });

  return {
    statusCode: 200,
    payload: {
      ok: true,
      mode: responsePayload.kind,
      message: responsePayload.message,
      localModel,
      runs: await listMietRuns(api),
    },
  };
}

export async function hydrateMietAutonomyOutput(api, {
  output = null,
  action = 'draft',
  jobNameHint = null,
} = {}) {
  const context = resolvePluginContext(api);
  const payload = output?.payload ?? {};
  const selectedTemplatePath = payload.selected_template?.path
    ? projectRelativePath(context.projectRoot, payload.selected_template.path)
    : null;
  const generatedSpec = payload.job_spec ?? null;
  const generatedSpecPath = payload.generated_files?.job_spec ?? null;
  const validationRunDir = payload.execution?.validation_run_dir ?? null;
  const finalRunDir = payload.execution?.final_run_dir ?? null;
  const detail = finalRunDir
    ? await readMietRunDetail(finalRunDir, { projectRoot: context.projectRoot })
    : (action === 'run' && validationRunDir
        ? await readMietRunDetail(validationRunDir, { projectRoot: context.projectRoot }).catch(() => null)
        : null);
  const notes = payload.generated_files?.notes
    ? await readTextIfExists(payload.generated_files.notes).catch(() => null)
    : null;
  const plan = await buildPlanFromJobSpec(api, {
    jobSpecPath: generatedSpecPath,
    jobSpec: generatedSpec,
    jobNameHint: payload.job_id ?? jobNameHint ?? null,
  }).catch(() => null);

  return {
    payload,
    plan,
    notes,
    detail,
    selectedTemplatePath,
    generatedSpec,
    generatedSpecPath,
    workspaceDirRelative: payload.workspace_dir
      ? projectRelativePath(context.projectRoot, payload.workspace_dir)
      : null,
    };
}

async function executeHydratedMietAutonomy(api, {
  subcommand = 'autonomy-draft',
  prompt = '',
  provider = 'local',
  outputDir = null,
  workspaceRoot = null,
  mode = null,
  templatePath = null,
  jobNameHint = null,
  materialName = null,
  dryRun = false,
  hydrateAction = 'draft',
} = {}) {
  const context = resolvePluginContext(api);
  const output = await runMietAutonomyCli(api, {
    subcommand,
    prompt,
    provider,
    outputDir: outputDir ?? context.defaultOutputDir,
    workspaceRoot: workspaceRoot ?? context.defaultWorkspaceRoot,
    mode,
    templatePath,
    jobNameHint,
    materialName,
    dryRun,
  });

  return {
    output,
    ...await hydrateMietAutonomyOutput(api, {
      output,
      action: hydrateAction,
      jobNameHint,
    }),
  };
}

function buildMietAutonomyToolResponse({
  presentation = 'api',
  action = 'draft',
  payload = null,
  plan = null,
  notes = null,
  detail = null,
  generatedSpecPath = null,
  previewConfirmable = false,
} = {}) {
  const isSlash = presentation === 'slash';
  const content = isSlash
    ? (action === 'run'
        ? (previewConfirmable
            ? `Execution preview ready for ${payload?.job_id ?? 'unknown job'}. Nothing has run yet — review the plan and confirm before launching.`
            : (detail
                ? `Run complete.\n\n${formatMietRunDetailForChat(detail)}`
                : `Run launched for ${payload?.job_id ?? 'unknown job'}.`))
        : [
            `Draft ready: ${payload?.job_id}`,
            `Mode: ${payload?.mode}`,
            `Template: ${payload?.selected_template?.file_name ?? 'unknown'}`,
            `Workspace: ${payload?.workspace_dir ?? 'unknown'}`,
          ].join('\n'))
    : (previewConfirmable
        ? `Execution preview ready for ${payload?.job_id ?? 'this task'}. Nothing has run yet — review the plan below and confirm when you are ready.`
        : `Draft preview ready for ${payload?.job_id ?? 'this task'}. You can review the generated plan, assumptions, and files below.`);

  return {
    refreshRuns: isSlash ? Boolean(detail) : false,
    ...buildToolResponsePayload({
      content,
      currentReport: payload,
      currentPlan: plan,
      currentReportNotes: notes,
      currentRunDetail: detail,
      previewConfirmable,
      previewExecuted: Boolean(detail),
      jobSpecPath: generatedSpecPath,
    }),
  };
}

export function formatMietPlanForChat(plan, { jobNameHint = null } = {}) {
  const steps = Array.isArray(plan)
    ? plan
    : (Array.isArray(plan?.plan) ? plan.plan : []);
  const resolvedJobName = jobNameHint ?? plan?.job_id ?? 'this job';

  if (!steps.length) {
    return `Plan ready for ${resolvedJobName}. No step list was produced.`;
  }

  return [
    `Plan ready for ${resolvedJobName}.`,
    `Mode: ${plan?.mode ?? 'unknown'}`,
    `Steps: ${steps.length}`,
    '',
    ...steps.map((step, index) => `${index + 1}. ${step.id ?? `step.${index + 1}`} — ${step.description ?? step.detail ?? step.stage ?? 'No additional detail.'}`),
  ].join('\n');
}

export function hydrateMietPlanActionOutput({
  output = null,
  jobNameHint = null,
} = {}) {
  const plan = output?.plan ?? null;
  const fallbackName = output?.jobSpecPath ? path.basename(output.jobSpecPath) : null;

  return {
    plan,
    ...buildToolResponsePayload({
      content: formatMietPlanForChat(plan, {
        jobNameHint: jobNameHint ?? plan?.job_id ?? fallbackName ?? null,
      }),
      currentPlan: plan,
    }),
  };
}

export async function hydrateMietRunSlashCommandOutput(api, {
  command = '/inspect',
  runId = null,
  runDir = null,
  target = 'auto',
} = {}) {
  const context = resolvePluginContext(api);
  const effectiveRunId = runId ?? null;
  const usage = command === '/artifacts'
    ? 'Usage: /artifacts <run-id>'
    : (command === '/logs'
        ? 'Usage: /logs <run-id> [md|kmc|summary]'
        : 'Usage: /inspect <run-id>');

  if (!effectiveRunId && !runDir) {
    return buildToolResponsePayload({ content: usage });
  }

  const { resolvedRunDir, detail } = await loadMietRunDetailById(api, {
    runId: effectiveRunId,
    runDir,
    allowMissing: true,
  });

  if (!detail) {
    return buildToolResponsePayload({
      content: `Run not found: ${effectiveRunId ?? path.basename(resolvedRunDir)}`,
    });
  }

  const content = command === '/artifacts'
    ? formatMietRunArtifactsForChat(detail)
    : (command === '/logs'
        ? await formatMietRunLogsForChat(resolvedRunDir, { target, projectRoot: context.projectRoot })
        : formatMietRunDetailForChat(detail));

  return {
    detail,
    ...buildToolResponsePayload({
      content,
      currentRunDetail: detail,
    }),
  };
}

export async function hydrateMietAutonomySlashCommandOutput(api, {
  command = '/draft',
  prompt = '',
  previewRuns = true,
  provider = 'local',
  outputDir = null,
  workspaceRoot = null,
} = {}) {
  const context = resolvePluginContext(api);
  const trimmedPrompt = String(prompt ?? '').trim();
  if (!trimmedPrompt) {
    return {
      refreshRuns: false,
      ...buildToolResponsePayload({
        content: `Usage: ${command} <prompt>`,
      }),
    };
  }

  const shouldPreviewRun = command === '/run' && previewRuns;
  const {
    output,
    payload,
    plan,
    notes,
    detail,
    generatedSpecPath,
  } = await executeHydratedMietAutonomy(api, {
    subcommand: command === '/run' && !previewRuns ? 'autonomy-run' : 'autonomy-draft',
    prompt: trimmedPrompt,
    provider,
    outputDir: outputDir ?? context.defaultOutputDir,
    workspaceRoot: workspaceRoot ?? context.defaultWorkspaceRoot,
    dryRun: false,
    hydrateAction: command === '/run' && !previewRuns ? 'run' : 'draft',
  });

  return {
    output,
    payload,
    plan,
    notes,
    detail,
    ...buildMietAutonomyToolResponse({
      presentation: 'slash',
      action: command === '/run' ? 'run' : 'draft',
      payload,
      plan,
      notes,
      detail,
      generatedSpecPath,
      previewConfirmable: shouldPreviewRun,
    }),
  };
}

export async function hydrateMietAutonomyActionOutput(api, {
  action = 'draft',
  prompt = '',
  provider = 'local',
  outputDir = null,
  workspaceRoot = null,
  mode = null,
  templatePath = null,
  jobNameHint = null,
  materialName = null,
  dryRun = false,
  confirmable = false,
} = {}) {
  const {
    output,
    payload,
    plan,
    notes,
    detail,
    selectedTemplatePath,
    generatedSpec,
    generatedSpecPath,
    workspaceDirRelative,
  } = await executeHydratedMietAutonomy(api, {
    subcommand: action === 'draft' ? 'autonomy-draft' : 'autonomy-run',
    prompt,
    provider,
    outputDir,
    workspaceRoot,
    mode,
    templatePath,
    jobNameHint,
    materialName,
    dryRun,
    hydrateAction: action,
  });

  return {
    output,
    payload,
    plan,
    notes,
    detail,
    selectedTemplatePath,
    generatedSpec,
    generatedSpecPath,
    workspaceDirRelative,
    ...buildMietAutonomyToolResponse({
      presentation: 'api',
      action,
      payload,
      plan,
      notes,
      detail,
      generatedSpecPath,
      previewConfirmable: Boolean(confirmable),
    }),
  };
}

export async function hydrateMietRunActionOutput(api, {
  action = 'inspect',
  output = null,
  runDir = null,
} = {}) {
  const context = resolvePluginContext(api);
  const effectiveRunDir = runDir ?? output?.runDir ?? null;
  const { detail } = effectiveRunDir
    ? await loadMietRunDetailById(api, {
        runDir: effectiveRunDir,
        outputDir: output?.outputDir ?? null,
        allowMissing: true,
      })
    : { detail: null };

  const content = action === 'inspect'
    ? (detail ? formatMietRunDetailForChat(detail) : 'Run not found. Review the provided run id or directory.')
    : (
        detail
          ? `Run complete for ${detail.id}. The result card below shows the finished steps and the evidence that was produced.`
          : 'Run finished. Review the run card below for the result.'
      );

  return {
    detail,
    ...buildToolResponsePayload({
      content,
      currentRunDetail: detail,
    }),
  };
}

export async function buildMietChatSlashCommandPayload(api, {
  input,
  previewRuns = true,
  provider = 'local',
  defaultOutputDir = null,
  defaultWorkspaceRoot = null,
} = {}) {
  const context = resolvePluginContext(api);
  const tokens = tokenizeMietCommand(input);
  const command = tokens[0] ?? '';
  const args = tokens.slice(1);
  const outputDir = defaultOutputDir ?? context.defaultOutputDir;
  const workspaceRoot = defaultWorkspaceRoot ?? context.defaultWorkspaceRoot;

  if (command === '/status' || command === '/tools' || command === '/runs') {
    return await hydrateMietChatPromptOutput(api, {
      prompt: input,
      provider,
      outputDir,
      workspaceRoot,
      previewRuns,
      refreshRuns: false,
    });
  }

  if (command === '/help') {
    return {
      refreshRuns: false,
      ...buildToolResponsePayload({
        content: renderMietSlashCommandHelpText(),
      }),
    };
  }

  if (command === '/inspect' || command === '/artifacts' || command === '/logs') {
    return {
      ...await hydrateMietRunSlashCommandOutput(api, {
        command,
        runId: args[0] ?? null,
        target: command === '/logs' ? (args[1] ?? 'auto') : 'auto',
      }),
      refreshRuns: false,
    };
  }

  if (command === '/draft' || command === '/run') {
    return await hydrateMietAutonomySlashCommandOutput(api, {
      command,
      prompt: args.join(' ').trim(),
      previewRuns,
      provider,
      outputDir,
      workspaceRoot,
    });
  }

  if (command === '/moire-run') {
    const caseDir = args[0];
    const workdir = args[1];
    if (!caseDir) {
      return {
        refreshRuns: false,
        ...buildToolResponsePayload({
          content: 'Usage: /moire-run <MoRe-case-dir> [workdir]',
        }),
      };
    }

    const prompt = workdir
      ? `请直接在本机上跑 MoRe 的 LAMMPS，然后把结果接到 KMC：${caseDir} ${workdir}`
      : `请直接在本机上跑 MoRe 的 LAMMPS，然后把结果接到 KMC：${caseDir}`;

    return await hydrateMietChatPromptOutput(api, {
      prompt,
      provider,
      outputDir,
      workspaceRoot,
      previewRuns,
      refreshRuns: true,
    });
  }

  if (command === '/moire-compare') {
    if (args.length < 3) {
      return {
        refreshRuns: false,
        ...buildToolResponsePayload({
          content: 'Usage: /moire-compare <MoRe-case-dir> <event-a.json> <event-b.json> [event-c.json ...] [workdir]',
        }),
      };
    }

    const caseDir = args[0];
    let trailing = args.slice(1);
    let workdir = null;
    const lastArg = trailing.at(-1) ?? null;
    if (lastArg && !lastArg.endsWith('.json')) {
      workdir = lastArg;
      trailing = trailing.slice(0, -1);
    }
    const eventJsons = trailing.filter((item) => item.endsWith('.json'));
    if (eventJsons.length < 2) {
      return {
        refreshRuns: false,
        ...buildToolResponsePayload({
          content: 'Usage: /moire-compare <MoRe-case-dir> <event-a.json> <event-b.json> [event-c.json ...] [workdir]',
        }),
      };
    }

    const prompt = [
      '请比较这些 MoRe event 的 LAMMPS barrier，并把结果接到 KMC：',
      caseDir,
      ...eventJsons,
      workdir ?? '',
    ].filter(Boolean).join(' ');

    return await hydrateMietChatPromptOutput(api, {
      prompt,
      provider,
      outputDir,
      workspaceRoot,
      previewRuns,
      refreshRuns: true,
    });
  }

  if (command === '/moire-diffusion-sweep') {
    const eventJson = args[0];
    const caseDir = args[1];
    const workdir = args[2];
    if (!eventJson || !caseDir) {
      return {
        refreshRuns: false,
        ...buildToolResponsePayload({
          content: 'Usage: /moire-diffusion-sweep <event.json> <MoRe-case-dir> [workdir]',
        }),
      };
    }

    const prompt = [
      '请先用这个 event.json 跑 MoRe 的 LAMMPS 算 barrier，再把 barrier 接到 KMC 做温度扫描，输出扩散系数与温度的关系，并用 OVITO 可视化：',
      eventJson,
      caseDir,
      workdir ?? '',
    ].filter(Boolean).join(' ');

    return await hydrateMietChatPromptOutput(api, {
      prompt,
      provider,
      outputDir,
      workspaceRoot,
      previewRuns,
      refreshRuns: true,
    });
  }

  return {
    refreshRuns: false,
    ...buildToolResponsePayload({
      content: `Unknown command: ${command}`,
    }),
  };
}

export async function buildMietActionApiPayload(api, {
  action,
  body = {},
  defaultOutputDir = null,
} = {}) {
  const context = resolvePluginContext(api);
  const jobSpec = body.jobSpec ?? null;
  const payload = {
    jobSpecPath: body.jobSpecPath ?? null,
    jobSpec: jobSpec ? resolveMietJobSpecPaths(api, jobSpec, { templatePath: body.templatePath ?? null }) : null,
    jobNameHint: body.jobNameHint ?? null,
    outputDir: body.outputDir ?? defaultOutputDir ?? context.defaultOutputDir,
    dryRun: Boolean(body.dryRun),
    runDir: body.runDir ?? null,
    resume: Boolean(body.resume),
    overwriteExisting: Boolean(body.overwriteExisting ?? body.overwrite_existing),
  };

  if (action === 'plan') {
    const output = await runMietCli(api, {
      subcommand: 'plan',
      jobSpecPath: payload.jobSpecPath,
      jobSpec: payload.jobSpec,
      jobNameHint: payload.jobNameHint,
    });
    const responsePayload = hydrateMietPlanActionOutput({
      output,
      jobNameHint: payload.jobNameHint,
    });
    return {
      statusCode: 200,
      payload: {
        ok: true,
        action,
        output,
        plan: responsePayload.plan,
        ...responsePayload,
      },
    };
  }

  if (action === 'run' || action === 'resume') {
    const output = await runMietCli(api, {
      subcommand: 'run',
      jobSpecPath: payload.jobSpecPath,
      jobSpec: payload.jobSpec,
      jobNameHint: payload.jobNameHint,
      outputDir: payload.outputDir,
      dryRun: payload.dryRun,
      runDir: action === 'resume' ? payload.runDir : null,
      resume: action === 'resume' || payload.resume,
      overwriteExisting: action !== 'resume' && payload.overwriteExisting,
    });
    const responsePayload = await hydrateMietRunActionOutput(api, {
      action,
      output,
    });
    return {
      statusCode: 200,
      payload: {
        ok: true,
        action,
        output,
        detail: responsePayload.detail,
        ...responsePayload,
      },
    };
  }

  if (action === 'inspect') {
    if (!payload.runDir) {
      return {
        statusCode: 400,
        payload: { ok: false, error: 'runDir is required for inspect' },
      };
    }
    const responsePayload = await hydrateMietRunActionOutput(api, {
      action,
      runDir: payload.runDir,
    });
    return {
      statusCode: 200,
      payload: {
        ok: true,
        action,
        detail: responsePayload.detail,
        ...responsePayload,
      },
    };
  }

  if (action === 'transcript-evidence') {
    if (!body.transcriptPath) {
      return {
        statusCode: 400,
        payload: { ok: false, error: 'transcriptPath is required for transcript-evidence' },
      };
    }
    try {
      const evidence = await loadMietTranscriptEvidence({
        transcriptPath: body.transcriptPath,
        traceId: body.traceId ?? null,
        eventIndex: body.eventIndex ?? null,
      });
      return {
        statusCode: 200,
        payload: {
          ok: true,
          action,
          evidence,
        },
      };
    } catch (error) {
      return {
        statusCode: 404,
        payload: { ok: false, error: error instanceof Error ? error.message : String(error) },
      };
    }
  }

  return {
    statusCode: 404,
    payload: { ok: false, error: `Unknown action: ${action}` },
  };
}

export async function buildMietAutonomyApiPayload(api, {
  action,
  body = {},
  defaultOutputDir = null,
  defaultWorkspaceRoot = null,
} = {}) {
  const context = resolvePluginContext(api);
  const prompt = String(body.prompt ?? '').trim();
  if (!prompt) {
    return {
      statusCode: 400,
      payload: { ok: false, error: 'prompt is required for autonomy actions' },
    };
  }

  const responsePayload = await hydrateMietAutonomyActionOutput(api, {
    action,
    prompt,
    provider: body.provider ?? 'local',
    outputDir: body.outputDir ?? defaultOutputDir ?? context.defaultOutputDir,
    workspaceRoot: body.workspaceRoot ?? defaultWorkspaceRoot ?? context.defaultWorkspaceRoot,
    mode: body.mode || null,
    templatePath: body.templatePath || null,
    jobNameHint: body.jobNameHint || null,
    materialName: body.materialName || null,
    dryRun: Boolean(body.dryRun),
    confirmable: Boolean(body.confirmable),
  });

  return {
    statusCode: 200,
    payload: {
      ok: true,
      action: `autonomy-${action}`,
      ...responsePayload,
    },
  };
}

export async function runMietChatCli(api, {
  prompt,
  provider,
  outputDir,
  workspaceRoot,
  mode,
  model,
  historyMessages,
  previewRuns = true,
}) {
  const context = resolvePluginContext(api);
  const finalOutputDir = resolveMaybePath(context.projectRoot, outputDir) || context.defaultOutputDir;
  const command = [
    context.pythonBin,
    '-m',
    'miet_claw.cli',
    'chat',
    '--once',
    prompt,
    '--json',
    '--provider',
    provider || 'auto',
    '--project-root',
    context.projectRoot,
    '--workspace-root',
    resolveMaybePath(context.projectRoot, workspaceRoot || path.join(context.projectRoot, '.autonomy')),
    '--output-dir',
    finalOutputDir,
  ];

  if (mode) command.push('--mode', mode);
  if (model) command.push('--model', model);

  const historyPath = await materializeChatHistory(finalOutputDir, historyMessages);
  if (historyPath) {
    command.push('--history-file', historyPath);
  }

  const env = {
    ...process.env,
    PYTHONPATH: joinPythonPath(context.projectRoot, process.env.PYTHONPATH),
    MIETCLAW_PREVIEW_RUNS: previewRuns ? '1' : '0',
  };

  try {
    const result = await runCommand(command, {
      cwd: context.projectRoot,
      env,
    });

    if (result.code !== 0) {
      throw new Error(`mietclaw chat CLI failed (${result.code}): ${result.stderr || result.stdout}`.trim());
    }

    return {
      command,
      cwd: context.projectRoot,
      stdout: result.stdout.trim(),
      stderr: result.stderr.trim(),
      payload: JSON.parse(result.stdout.trim()),
    };
  } finally {
    if (historyPath) {
      await fsp.unlink(historyPath).catch(() => {});
    }
  }
}

export async function runMietJsonCli(api, {
  subcommand,
  projectRoot,
  outputDir,
  runDir,
  runName,
  target,
  maxLines,
  limit,
  eventJson,
  eventJsons,
  nebTxt,
  barrier,
  barrierEV,
  caseDir,
  workdir,
  validate,
  temperaturesK,
  kmcSeed,
  kmcSeeds,
  runTime,
  statsStep,
  ovito,
  ovitoPython,
  lammpsOnly,
  dataLmp,
}) {
  const context = resolvePluginContext(api);
  const finalOutputDir = resolveMaybePath(context.projectRoot, outputDir) || context.defaultOutputDir;
  const command = [
    context.pythonBin,
    '-m',
    'miet_claw.cli',
    subcommand,
  ];
  const listArg = (value) => Array.isArray(value) ? value.join(',') : value;

  if (subcommand === 'runs') {
    command.push('--output-dir', finalOutputDir);
    if (limit) command.push('--limit', String(limit));
  }

  if (subcommand === 'inspect' || subcommand === 'artifacts' || subcommand === 'logs') {
    const runTarget = runDir ? String(runDir) : runName;
    if (runTarget) command.push(runTarget);
    command.push('--output-dir', finalOutputDir);
    if (subcommand === 'artifacts' && limit) command.push('--limit', String(limit));
    if (subcommand === 'logs') {
      command.push('--target', target || 'auto');
      if (maxLines) command.push('--max-lines', String(maxLines));
    }
  }

  if (subcommand === 'bridge') {
    if (!eventJson) {
      throw new Error('event_json is required for miet_kmc_bridge.');
    }
    command.push(resolveMaybePath(context.projectRoot, eventJson));
    if (nebTxt) command.push('--neb-txt', resolveMaybePath(context.projectRoot, nebTxt));
    if (typeof barrier === 'number') command.push('--barrier', String(barrier));
    command.push('--output-dir', finalOutputDir);
    command.push('--workdir', resolveMaybePath(context.projectRoot, workdir || path.join(finalOutputDir, `bridge-${Date.now()}`)));
    if (validate !== false) command.push('--validate');
  }

  if (subcommand === 'doctor') {
    command.push('--project-root', resolveMaybePath(context.projectRoot, projectRoot || context.projectRoot), '--json');
  }

  if (subcommand === 'moire-run') {
    if (!caseDir) throw new Error('case_dir is required for miet_moire_run.');
    command.push(resolveMaybePath(context.projectRoot, caseDir));
    command.push('--output-dir', finalOutputDir);
    command.push('--workdir', resolveMaybePath(context.projectRoot, workdir || path.join(finalOutputDir, `moire-run-${Date.now()}`)));
    if (eventJson) command.push('--event-json', resolveMaybePath(context.projectRoot, eventJson));
    if (validate) command.push('--validate');
    if (kmcSeed) command.push('--kmc-seed', String(kmcSeed));
    if (kmcSeeds) command.push('--kmc-seeds', String(listArg(kmcSeeds)));
    if (ovito) command.push('--ovito');
    if (ovitoPython) command.push('--ovito-python', resolveMaybePath(context.projectRoot, ovitoPython));
  }

  if (subcommand === 'moire-lammps') {
    if (!caseDir) throw new Error('case_dir is required for miet_moire_lammps.');
    command.push(resolveMaybePath(context.projectRoot, caseDir));
    command.push('--output-dir', finalOutputDir);
    command.push('--workdir', resolveMaybePath(context.projectRoot, workdir || path.join(finalOutputDir, `moire-lammps-${Date.now()}`)));
    if (eventJson) command.push('--event-json', resolveMaybePath(context.projectRoot, eventJson));
    if (ovito) command.push('--ovito');
    if (ovitoPython) command.push('--ovito-python', resolveMaybePath(context.projectRoot, ovitoPython));
  }

  if (subcommand === 'moire-kmc') {
    const resolvedBarrier = barrierEV ?? barrier;
    if (typeof resolvedBarrier !== 'number') throw new Error('barrier_eV is required for miet_moire_kmc.');
    command.push(String(resolvedBarrier));
    command.push('--output-dir', finalOutputDir);
    command.push('--workdir', resolveMaybePath(context.projectRoot, workdir || path.join(finalOutputDir, `moire-kmc-${Date.now()}`)));
    if (eventJson) command.push('--event-json', resolveMaybePath(context.projectRoot, eventJson));
    if (dataLmp) command.push('--data-lmp', resolveMaybePath(context.projectRoot, dataLmp));
    if (kmcSeed) command.push('--kmc-seed', String(kmcSeed));
    if (kmcSeeds) command.push('--kmc-seeds', String(listArg(kmcSeeds)));
    if (ovito) command.push('--ovito');
    if (ovitoPython) command.push('--ovito-python', resolveMaybePath(context.projectRoot, ovitoPython));
  }

  if (subcommand === 'moire-compare') {
    if (!caseDir) throw new Error('case_dir is required for miet_moire_compare.');
    if (!Array.isArray(eventJsons) || eventJsons.length < 2) {
      throw new Error('event_jsons must contain at least two event files for miet_moire_compare.');
    }
    command.push(resolveMaybePath(context.projectRoot, caseDir));
    command.push(...eventJsons.map((item) => resolveMaybePath(context.projectRoot, item)));
    command.push('--output-dir', finalOutputDir);
    command.push('--workdir', resolveMaybePath(context.projectRoot, workdir || path.join(finalOutputDir, `moire-compare-${Date.now()}`)));
    if (validate) command.push('--validate');
    if (kmcSeed) command.push('--kmc-seed', String(kmcSeed));
    if (kmcSeeds) command.push('--kmc-seeds', String(listArg(kmcSeeds)));
    if (ovito) command.push('--ovito');
    if (ovitoPython) command.push('--ovito-python', resolveMaybePath(context.projectRoot, ovitoPython));
    if (lammpsOnly) command.push('--lammps-only');
  }

  if (subcommand === 'moire-diffusion-sweep') {
    if (!eventJson) throw new Error('event_json is required for miet_moire_diffusion_sweep.');
    if (!caseDir) throw new Error('case_dir is required for miet_moire_diffusion_sweep.');
    command.push(resolveMaybePath(context.projectRoot, eventJson));
    command.push(resolveMaybePath(context.projectRoot, caseDir));
    command.push('--output-dir', finalOutputDir);
    command.push('--workdir', resolveMaybePath(context.projectRoot, workdir || path.join(finalOutputDir, `moire-diffusion-${Date.now()}`)));
    if (temperaturesK) command.push('--temperatures', String(listArg(temperaturesK)));
    if (validate) command.push('--validate');
    if (kmcSeed) command.push('--kmc-seed', String(kmcSeed));
    if (kmcSeeds) command.push('--kmc-seeds', String(listArg(kmcSeeds)));
    if (runTime) command.push('--run-time', String(runTime));
    if (statsStep) command.push('--stats-step', String(statsStep));
    if (ovito) command.push('--ovito');
    if (ovitoPython) command.push('--ovito-python', resolveMaybePath(context.projectRoot, ovitoPython));
  }

  const env = {
    ...process.env,
    PYTHONPATH: joinPythonPath(context.projectRoot, process.env.PYTHONPATH),
  };
  const result = await runCommand(command, {
    cwd: context.projectRoot,
    env,
  });

  if (result.code !== 0 && subcommand !== 'doctor') {
    throw new Error(`mietclaw JSON CLI failed (${result.code}): ${result.stderr || result.stdout}`.trim());
  }

  const payload = JSON.parse(result.stdout.trim());
  return {
    command,
    cwd: context.projectRoot,
    stdout: result.stdout.trim(),
    stderr: result.stderr.trim(),
    payload,
  };
}

export async function loadRunSnapshot(runDir) {
  const resolvedRunDir = path.resolve(runDir);
  const readJsonIfExists = async (filePath) => {
    if (!fs.existsSync(filePath)) return null;
    return JSON.parse(await fsp.readFile(filePath, 'utf-8'));
  };
  const readTextIfExists = async (filePath) => {
    if (!fs.existsSync(filePath)) return null;
    return await fsp.readFile(filePath, 'utf-8');
  };

  const state = await readJsonIfExists(path.join(resolvedRunDir, 'state.json'));
  const manifest = await readJsonIfExists(path.join(resolvedRunDir, 'archive', 'manifest.json'));
  const summary = await readTextIfExists(path.join(resolvedRunDir, 'explain', 'summary.md'));
  const summaryJson = await readJsonIfExists(path.join(resolvedRunDir, 'summary.json'));

  if (!state && !manifest && !summary && !summaryJson) {
    throw new Error(`No mietclaw run data found in ${resolvedRunDir}`);
  }

  return {
    runDir: resolvedRunDir,
    state,
    manifest,
    summary,
    summaryJson,
  };
}

function toNumber(value) {
  if (value === null || value === undefined || value === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function buildWorkflowExecutionProvenance(mdExecution = null, kmcExecution = null) {
  const stages = {};
  if (mdExecution && typeof mdExecution === 'object') {
    stages.md = {
      mode: mdExecution.mode ?? null,
      reason: mdExecution.reason ?? null,
      command: mdExecution.command ?? null,
    };
  }
  if (kmcExecution && typeof kmcExecution === 'object') {
    stages.kmc = {
      mode: kmcExecution.mode ?? null,
      reason: kmcExecution.reason ?? null,
      diffusionMode: kmcExecution.diffusion_mode ?? null,
      command: kmcExecution.command ?? null,
    };
  }
  const hasSimulatedOutputs = Object.values(stages).some((stage) => (
    stage.mode === 'dry-run'
    || stage.mode === 'simulated'
    || stage.diffusionMode === 'simulated'
  ));
  return {
    stages,
    hasSimulatedOutputs,
    label: hasSimulatedOutputs ? 'simulated/dry-run outputs present' : 'real/file-backed execution',
  };
}

function parseCsv(text) {
  if (!text || !text.trim()) return [];
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) return [];
  const headers = lines[0].split(',').map((header) => header.trim());
  return lines.slice(1).map((line) => {
    const values = line.split(',');
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? '';
    });
    return row;
  });
}

function firstExisting(...paths) {
  return paths.find((candidate) => candidate && fs.existsSync(candidate)) ?? null;
}

async function readJsonIfExists(filePath) {
  if (!filePath || !fs.existsSync(filePath)) return null;
  return JSON.parse(await fsp.readFile(filePath, 'utf-8'));
}

function projectRelativePath(projectRoot, filePath) {
  if (!filePath) return null;
  if (!projectRoot) return filePath;
  return path.relative(projectRoot, filePath) || '.';
}

function availableMietRunRoots(context, { outputDir = null } = {}) {
  const primaryRoot = resolveMaybePath(context.projectRoot, outputDir) || context.defaultOutputDir;
  const legacyRoot = path.join(context.projectRoot, '.runs');
  return [primaryRoot, legacyRoot]
    .filter((root, index, allRoots) => allRoots.indexOf(root) === index && fs.existsSync(root));
}

export function resolveMietRunDir(api, runId, { outputDir = null } = {}) {
  if (!runId) return null;
  if (path.isAbsolute(runId) && fs.existsSync(runId)) return runId;
  const context = resolvePluginContext(api);
  for (const root of availableMietRunRoots(context, { outputDir })) {
    const candidate = path.join(root, runId);
    if (fs.existsSync(candidate)) return candidate;
  }
  return path.join(resolveMaybePath(context.projectRoot, outputDir) || context.defaultOutputDir, runId);
}

function deriveRunStatus(state) {
  const steps = Object.values(state?.steps ?? {});
  if (steps.some((step) => step.status === 'failed')) return 'Failed';
  if (steps.length > 0 && steps.every((step) => step.status === 'completed')) return 'Completed';
  if (steps.some((step) => step.status === 'running' || step.status === 'started')) return 'Running';
  return 'Idle';
}

function summaryRunKind(summaryJson) {
  if (!summaryJson) return 'summary_run';
  if (summaryJson.kmc && summaryJson.source_case_dir) return 'moire_lammps_to_kmc';
  if (summaryJson.files && summaryJson.barrier_eV !== undefined) return 'bridge_kmc_lookup';
  return 'summary_run';
}

function summaryMaterialName(summaryJson, resolvedRunDir) {
  if (typeof summaryJson?.source_case_dir === 'string' && summaryJson.source_case_dir.trim()) {
    const parts = summaryJson.source_case_dir.split('/').filter(Boolean);
    return `MoRe case · ${parts.slice(-3).join('/')}`;
  }
  if (summaryJson?.files) return 'KMC bridge';
  return path.basename(resolvedRunDir);
}

function summaryStepStatuses(summaryJson) {
  const kind = summaryRunKind(summaryJson);
  if (kind === 'moire_lammps_to_kmc') {
    return {
      lammps: summaryJson?.lammps?.status ?? null,
      postprocess: summaryJson?.postprocess?.status ?? null,
      kmc: summaryJson?.kmc?.status ?? null,
    };
  }
  if (kind === 'bridge_kmc_lookup') {
    return {
      bridge: summaryJson?.status ?? null,
      validation: summaryJson?.validation_passed === undefined ? null : (summaryJson.validation_passed ? 'completed' : 'failed'),
    };
  }
  return { summary: summaryJson?.status ?? null };
}

function normalizeSummaryStepStatus(status) {
  const value = String(status ?? '').toLowerCase();
  if (['completed', 'executed', 'ok', 'healthy', 'passed', 'success'].includes(value)) return 'completed';
  if (['failed', 'error', 'aborted'].includes(value)) return 'failed';
  if (value) return 'running';
  return 'idle';
}

function deriveSummaryRunStatus(summaryJson) {
  const statuses = Object.values(summaryStepStatuses(summaryJson));
  if (statuses.some((item) => normalizeSummaryStepStatus(item) === 'failed')) return 'Failed';
  if (statuses.length > 0 && statuses.every((item) => normalizeSummaryStepStatus(item) === 'completed')) return 'Completed';
  const runtimeHealth = String(summaryJson?.runtime_health?.status ?? summaryJson?.kmc?.runtime_health?.status ?? '').toLowerCase();
  if (runtimeHealth === 'ok') return 'Completed';
  if (runtimeHealth === 'failed' || runtimeHealth === 'error') return 'Failed';
  if (statuses.some((item) => normalizeSummaryStepStatus(item) === 'running')) return 'Running';
  return String(summaryJson?.status ?? '').toLowerCase() === 'completed' ? 'Completed' : 'Idle';
}

function summaryBarrierEvents(summaryJson) {
  const kmc = summaryJson?.kmc ?? summaryJson ?? {};
  const assignment = kmc?.barrier_assignment ?? {};
  const events = Object.entries(assignment)
    .filter(([species]) => species !== 'note')
    .map(([species, barrier]) => ({
      species,
      barrier_ev: Number(barrier),
      barrier_source: 'lammps-neb',
    }))
    .filter((item) => Number.isFinite(item.barrier_ev));
  if (events.length) return events;
  const barrier = Number(kmc?.barrier_eV ?? summaryJson?.barrier_eV);
  if (Number.isFinite(barrier)) {
    return [{ species: 'shared', barrier_ev: barrier, barrier_source: 'lammps-neb' }];
  }
  return [];
}

function summaryArtifacts(resolvedRunDir) {
  return fs.readdirSync(resolvedRunDir, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => {
      const relativeParent = entry.parentPath ? path.relative(resolvedRunDir, entry.parentPath) : '.';
      return relativeParent === '.' ? entry.name : path.join(relativeParent, entry.name);
    })
    .sort();
}

function summaryPreviewText(summaryJson, resolvedRunDir, projectRoot = null) {
  const lines = [
    `Summary file: ${projectRelativePath(projectRoot, path.join(resolvedRunDir, 'summary.json'))}`,
    `Status: ${summaryJson?.status ?? 'unknown'}`,
  ];
  if (summaryJson?.source_case_dir) lines.push(`Source case: ${summaryJson.source_case_dir}`);
  const barrier = summaryJson?.kmc?.barrier_eV ?? summaryJson?.barrier_eV;
  if (barrier !== undefined) lines.push(`Barrier: ${barrier} eV`);
  if (summaryJson?.kmc?.parsed_run?.accepted_events !== undefined) {
    lines.push(`Accepted events: ${summaryJson.kmc.parsed_run.accepted_events}`);
  }
  return lines.join('\n');
}

function summarizeSteps(state) {
  return Object.entries(state?.steps ?? {}).map(([stepId, record]) => ({
    id: stepId,
    status: record.status,
    startedAt: record.started_at ?? null,
    heartbeatAt: record.heartbeat_at ?? null,
    completedAt: record.completed_at ?? null,
    pid: record.pid ?? null,
    detail: record.detail ?? null,
    error: record.error ?? null,
    outputs: record.outputs ?? {},
  }));
}

function deriveActiveStep(steps) {
  return steps.find((step) => step.status === 'running' || step.status === 'started') ?? null;
}

export async function readMietRunDetail(runDir, { projectRoot = null } = {}) {
  const snapshot = await loadRunSnapshot(runDir);
  const resolvedRunDir = snapshot.runDir;
  const summaryJson = snapshot.summaryJson ?? null;
  const stats = await fsp.stat(resolvedRunDir);

  if (!snapshot.state && summaryJson) {
    const kind = summaryRunKind(summaryJson);
    const stepStatusMap = summaryStepStatuses(summaryJson);
    const steps = Object.entries(stepStatusMap).map(([stepId, status]) => ({
      id: stepId,
      status,
      startedAt: null,
      heartbeatAt: null,
      completedAt: null,
      pid: null,
      detail: null,
    }));
    const completedSteps = steps.filter((step) => normalizeSummaryStepStatus(step.status) === 'completed').length;
    const totalSteps = steps.length;
    const events = summaryBarrierEvents(summaryJson);
    const artifactPaths = summaryArtifacts(resolvedRunDir);
    const generatedInputPath = summaryJson?.kmc?.files?.input_kmc ?? summaryJson?.files?.input_ml ?? null;
    const summaryText = summaryPreviewText(summaryJson, resolvedRunDir, projectRoot);

    return {
      id: path.basename(resolvedRunDir),
      runDir: resolvedRunDir,
      runDirRelative: projectRelativePath(projectRoot, resolvedRunDir),
      jobId: path.basename(resolvedRunDir),
      mode: kind,
      materialName: summaryMaterialName(summaryJson, resolvedRunDir),
      updatedAt: stats.mtime.toISOString(),
      createdAt: stats.birthtime.toISOString(),
      status: deriveSummaryRunStatus(summaryJson),
      completedSteps,
      totalSteps,
      activeStep: null,
      summary: summaryText,
      summaryPreview: summaryText.slice(0, 800),
      manifest: {
        artifacts: artifactPaths.map((artifactPath) => ({ path: artifactPath })),
      },
      spec: null,
      steps,
      md: {
        barriers: {
          metadata: {
            workflow_kind: kind,
            barrier_source_mode: kind === 'bridge_kmc_lookup' ? 'lookup-bridge' : 'lammps-neb',
          },
          events,
        },
        execution: summaryJson?.lammps ?? null,
        referenceEnergyEv: null,
      },
      chain: {
        eventRows: [],
        eventTablePath: null,
      },
      kmc: {
        execution: summaryJson?.kmc ?? null,
        generatedInputPath: projectRelativePath(projectRoot, generatedInputPath),
        generatedInput: await readTextIfExists(generatedInputPath),
        diffusionRows: [],
        diffusionPath: null,
        latestDiffusion: null,
      },
      commandActions: buildMietRunCommandActions(path.basename(resolvedRunDir)),
    };
  }

  const state = snapshot.state ?? {};
  const spec = await readJsonIfExists(path.join(resolvedRunDir, 'job_spec.resolved.json'));
  const barriers = await readJsonIfExists(path.join(resolvedRunDir, 'artifacts', 'md', 'barriers.json'));
  const mdExecution = await readJsonIfExists(path.join(resolvedRunDir, 'artifacts', 'md', 'md_execution.json'));
  const kmcExecution = await readJsonIfExists(path.join(resolvedRunDir, 'artifacts', 'kmc', 'kmc_execution.json'));
  const executionProvenance = buildWorkflowExecutionProvenance(mdExecution, kmcExecution);

  const eventTablePath = firstExisting(
    path.join(resolvedRunDir, 'artifacts', 'chain', 'event_table.csv'),
    path.join(resolvedRunDir, 'artifacts', 'kmc', 'event_table.csv'),
  );
  const diffusionPath = firstExisting(
    path.join(resolvedRunDir, 'artifacts', 'kmc', 'diffusion.csv'),
  );
  const generatedInputPath = firstExisting(
    path.join(resolvedRunDir, 'artifacts', 'kmc', 'generated_kmc.in'),
  );

  const eventRows = parseCsv(await readTextIfExists(eventTablePath));
  const diffusionRows = parseCsv(await readTextIfExists(diffusionPath));
  const latestDiffusion = diffusionRows.at(-1) ?? null;
  const steps = summarizeSteps(state);
  const completedSteps = steps.filter((step) => step.status === 'completed').length;
  const totalSteps = steps.length;
  const activeStep = deriveActiveStep(steps);

  return {
    id: path.basename(resolvedRunDir),
    runDir: resolvedRunDir,
    runDirRelative: projectRelativePath(projectRoot, resolvedRunDir),
    jobId: state.job_id ?? spec?.job_id ?? path.basename(resolvedRunDir),
    mode: state.mode ?? spec?.mode ?? 'unknown',
    materialName: spec?.material_system?.name ?? 'Unnamed material system',
    updatedAt: state.updated_at ?? null,
    createdAt: state.created_at ?? null,
    status: deriveRunStatus(state),
    completedSteps,
    totalSteps,
    activeStep,
    executionProvenance,
    hasSimulatedOutputs: executionProvenance.hasSimulatedOutputs,
    summary: snapshot.summary ?? '',
    summaryPreview: snapshot.summary ? snapshot.summary.slice(0, 800) : '',
    manifest: snapshot.manifest ?? null,
    spec,
    steps,
    md: {
      barriers,
      execution: mdExecution,
      referenceEnergyEv: barriers?.metadata?.reference_energy_ev ?? null,
    },
    chain: {
      eventRows: eventRows.map((row) => ({
        ...row,
        barrier_ev: toNumber(row.barrier_ev),
        prefactor_hz: toNumber(row.prefactor_hz),
        temperature_k: toNumber(row.temperature_k),
        rate_hz: toNumber(row.rate_hz),
      })),
      eventTablePath: projectRelativePath(projectRoot, eventTablePath),
    },
      kmc: {
        execution: kmcExecution,
        generatedInputPath: projectRelativePath(projectRoot, generatedInputPath),
        generatedInput: await readTextIfExists(generatedInputPath),
      diffusionRows: diffusionRows.map((row) => ({
        ...row,
        'No.': toNumber(row['No.']),
        jumps: toNumber(row.jumps),
        msd: toNumber(row.msd),
        simulation_time: toNumber(row.simulation_time),
        'jump frequency': toNumber(row['jump frequency']),
        'diffusion coefficient': toNumber(row['diffusion coefficient']),
      })),
      diffusionPath: projectRelativePath(projectRoot, diffusionPath),
      latestDiffusion: latestDiffusion
        ? {
            jumps: toNumber(latestDiffusion.jumps),
            msd: toNumber(latestDiffusion.msd),
            simulationTime: toNumber(latestDiffusion.simulation_time),
            jumpFrequency: toNumber(latestDiffusion['jump frequency']),
            diffusionCoefficient: toNumber(latestDiffusion['diffusion coefficient']),
          }
        : null,
      },
      commandActions: buildMietRunCommandActions(path.basename(resolvedRunDir)),
    };
}

export async function listMietRuns(api, { outputDir = null } = {}) {
  const context = resolvePluginContext(api);
  const runDirs = [];
  const seen = new Set();

  for (const root of availableMietRunRoots(context, { outputDir })) {
    const entries = await fsp.readdir(root, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory() || entry.name.startsWith('_') || seen.has(entry.name)) continue;
      seen.add(entry.name);
      runDirs.push(path.join(root, entry.name));
    }
  }

  if (!runDirs.length) return [];

  const details = await Promise.all(
    runDirs.map(async (runDir) => {
      try {
        return await readMietRunDetail(runDir, { projectRoot: context.projectRoot });
      } catch {
        return null;
      }
    }),
  );

  return details
    .filter(Boolean)
    .sort((left, right) => String(right.updatedAt ?? right.createdAt ?? '').localeCompare(String(left.updatedAt ?? left.createdAt ?? '')))
    .map((run) => ({
      id: run.id,
      jobId: run.jobId,
      mode: run.mode,
      materialName: run.materialName,
      updatedAt: run.updatedAt,
      status: run.status,
      completedSteps: run.completedSteps,
      totalSteps: run.totalSteps,
      activeStep: run.activeStep,
      latestDiffusionCoefficient: run.kmc.latestDiffusion?.diffusionCoefficient ?? null,
      latestSimulationTime: run.kmc.latestDiffusion?.simulationTime ?? null,
      referenceEnergyEv: run.md.referenceEnergyEv ?? null,
      summaryPreview: run.summaryPreview,
      runDirRelative: run.runDirRelative,
      commandActions: buildMietRunCommandActions(run.id, {
        includeInspect: true,
        includeSummaryLog: false,
        includeKmcLog: true,
      }),
    }));
}

export async function loadMietRunDetailById(api, {
  runId = null,
  runDir = null,
  outputDir = null,
  allowMissing = false,
} = {}) {
  const context = resolvePluginContext(api);
  const resolvedRunDir = runDir
    ? resolveMietRunDir(api, runDir, { outputDir })
    : resolveMietRunDir(api, runId, { outputDir });

  if (!fs.existsSync(resolvedRunDir)) {
    if (allowMissing) {
      return { resolvedRunDir, detail: null };
    }
    throw new Error(`Run not found: ${runId ?? path.basename(resolvedRunDir)}`);
  }

  return {
    resolvedRunDir,
    detail: await readMietRunDetail(resolvedRunDir, { projectRoot: context.projectRoot }),
  };
}

export async function buildMietRunsApiPayload(api, { outputDir = null } = {}) {
  return {
    runs: await listMietRuns(api, { outputDir }),
  };
}

export async function buildMietRunDetailApiPayload(api, {
  runId,
  outputDir = null,
} = {}) {
  const { resolvedRunDir, detail } = await loadMietRunDetailById(api, {
    runId,
    outputDir,
  });
  return {
    runDir: resolvedRunDir,
    run: detail,
  };
}

export async function buildMietWebBootstrap(api, {
  includeSystem = true,
  includeLocalModel = true,
  includeCommandHints = false,
} = {}) {
  const [system, runs, templates, localModel] = await Promise.all([
    includeSystem ? getMietSystemStatus(api) : Promise.resolve(undefined),
    listMietRuns(api),
    listMietTemplates(api),
    includeLocalModel ? getMietLocalModelStatus() : Promise.resolve(undefined),
  ]);

  return {
    ...(includeSystem ? { system } : {}),
    ...(includeLocalModel ? { localModel } : {}),
    runs,
    templates,
    defaultTemplatePath: templates[0]?.path ?? null,
    hintChips: [...MIET_WEB_HINT_CHIPS],
    ...(includeCommandHints ? {
      commandHints: [...MIET_COMMAND_HINTS],
      commandDetails: listMietCommandDetails(),
    } : {}),
  };
}

export async function buildMietWebApiPayload(api, {
  method = 'GET',
  pathname = '/',
  body = {},
  defaultOutputDir = null,
  defaultWorkspaceRoot = null,
} = {}) {
  const normalizedMethod = String(method ?? 'GET').toUpperCase();
  const normalizedPathname = String(pathname ?? '/');

  if (normalizedMethod === 'GET' && normalizedPathname === '/api/bootstrap') {
    return {
      statusCode: 200,
      payload: await buildMietWebBootstrap(api, {
        includeSystem: true,
        includeLocalModel: true,
        includeCommandHints: false,
      }),
    };
  }

  if (normalizedMethod === 'GET' && normalizedPathname === '/api/chat/bootstrap') {
    return {
      statusCode: 200,
      payload: await buildMietWebBootstrap(api, {
        includeSystem: false,
        includeLocalModel: true,
        includeCommandHints: true,
      }),
    };
  }

  if (normalizedMethod === 'POST' && normalizedPathname === '/api/chat') {
    return await buildMietChatApiPayload(api, {
      body,
      defaultOutputDir,
      defaultWorkspaceRoot,
    });
  }

  if (normalizedMethod === 'GET' && normalizedPathname === '/api/runs') {
    return {
      statusCode: 200,
      payload: await buildMietRunsApiPayload(api, { outputDir: defaultOutputDir }),
    };
  }

  if (normalizedMethod === 'GET' && normalizedPathname.startsWith('/api/runs/')) {
    const runId = decodeURIComponent(normalizedPathname.replace('/api/runs/', ''));
    try {
      return {
        statusCode: 200,
        payload: await buildMietRunDetailApiPayload(api, {
          runId,
          outputDir: defaultOutputDir,
        }),
      };
    } catch (error) {
      return {
        statusCode: 404,
        payload: { ok: false, error: `Run not found: ${runId}` },
      };
    }
  }

  if (normalizedMethod === 'POST' && normalizedPathname.startsWith('/api/actions/')) {
    const action = normalizedPathname.replace('/api/actions/', '');
    return await buildMietActionApiPayload(api, {
      action,
      body,
      defaultOutputDir,
    });
  }

  if (normalizedMethod === 'POST' && normalizedPathname.startsWith('/api/autonomy/')) {
    const action = normalizedPathname.replace('/api/autonomy/', '');
    return await buildMietAutonomyApiPayload(api, {
      action,
      body,
      defaultOutputDir,
      defaultWorkspaceRoot,
    });
  }

  return {
    statusCode: 404,
    payload: { ok: false, error: `Unknown API route: ${normalizedPathname}` },
  };
}

export function buildMietStaticWebPayload({
  distDir,
  pathname = '/',
} = {}) {
  if (!fs.existsSync(distDir)) {
    return {
      kind: 'text',
      statusCode: 503,
      contentType: 'text/plain; charset=utf-8',
      body: 'dist/ not found. Run `npm run build --workspace @miet-claw/web` first.',
    };
  }

  const requested = pathname === '/' ? '/index.html' : pathname;
  const candidate = path.resolve(distDir, `.${requested}`);
  const safeCandidate = candidate.startsWith(distDir) ? candidate : path.join(distDir, 'index.html');

  let filePath = safeCandidate;
  if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    filePath = path.join(distDir, 'index.html');
  }

  const extension = path.extname(filePath);
  return {
    kind: 'file',
    statusCode: 200,
    filePath,
    contentType: MIET_WEB_MIME_TYPES[extension] ?? 'application/octet-stream',
  };
}

export async function buildMietWebHttpPayload(api, {
  request,
  method = null,
  pathname = '/',
  distDir = null,
} = {}) {
  const normalizedMethod = String(method ?? request?.method ?? 'GET').toUpperCase();
  const normalizedPathname = String(pathname ?? '/');

  if (normalizedPathname.startsWith('/api/')) {
    const body = normalizedMethod === 'POST'
      ? await readMietWebRequestBody(request)
      : {};
    const result = await buildMietWebApiPayload(api, {
      method: normalizedMethod,
      pathname: normalizedPathname,
      body,
    });
    return {
      kind: 'json',
      statusCode: result.statusCode,
      payload: result.payload,
    };
  }

  return buildMietStaticWebPayload({
    distDir,
    pathname: normalizedPathname,
  });
}

export function writeMietWebHttpResponse(response, result) {
  if (result.kind === 'json') {
    response.writeHead(result.statusCode, { 'content-type': 'application/json; charset=utf-8' });
    response.end(JSON.stringify(result.payload, null, 2));
    return;
  }

  if (result.kind === 'text') {
    response.writeHead(result.statusCode, { 'content-type': result.contentType ?? 'text/plain; charset=utf-8' });
    response.end(result.body ?? '');
    return;
  }

  response.writeHead(result.statusCode, { 'content-type': result.contentType ?? 'application/octet-stream' });
  fs.createReadStream(result.filePath).pipe(response);
}

export function createMietWebServer({
  projectRoot = null,
  webRoot = null,
  pythonBin = 'python3',
} = {}) {
  const resolvedWebRoot = resolveMaybePath(process.cwd(), webRoot) || process.cwd();
  const distDir = path.join(resolvedWebRoot, 'dist');
  const api = createMietWebApiContext({
    projectRoot,
    pythonBin,
  });

  return http.createServer(async (request, response) => {
    try {
      const url = new URL(request.url ?? '/', 'http://127.0.0.1');
      const result = await buildMietWebHttpPayload(api, {
        request,
        method: request.method,
        pathname: url.pathname,
        distDir,
      });
      writeMietWebHttpResponse(response, result);
    } catch (error) {
      writeMietWebHttpResponse(response, {
        kind: 'json',
        statusCode: 500,
        payload: {
          ok: false,
          error: String(error.message || error),
        },
      });
    }
  });
}

export async function startMietWebServer({
  argv = [],
  projectRoot = null,
  webRoot = null,
  pythonBin = 'python3',
  logger = console.log,
} = {}) {
  const { host, port } = parseMietWebServerArgs(argv);
  const server = createMietWebServer({
    projectRoot,
    webRoot,
    pythonBin,
  });

  await new Promise((resolve) => {
    server.listen(port, host, resolve);
  });

  if (typeof logger === 'function') {
    logger(`mietclaw web console listening at http://${host}:${port}`);
  }

  return { server, host, port };
}

function summaryLogCandidates(resolvedRunDir, summaryJson) {
  const kind = summaryRunKind(summaryJson);
  if (kind === 'moire_lammps_to_kmc') {
    return {
      md: summaryJson?.lammps?.log ?? path.join(resolvedRunDir, 'lammps_run.out'),
      kmc: summaryJson?.kmc?.files?.run_out ?? path.join(resolvedRunDir, 'kmc_bridge', 'run.out'),
      summary: path.join(resolvedRunDir, 'summary.json'),
    };
  }
  if (kind === 'bridge_kmc_lookup') {
    return {
      md: summaryJson?.files?.barriers_tsv ?? path.join(resolvedRunDir, 'summary.json'),
      kmc: summaryJson?.files?.run_out ?? path.join(resolvedRunDir, 'run.out'),
      summary: path.join(resolvedRunDir, 'summary.json'),
    };
  }
  return {
    md: path.join(resolvedRunDir, 'summary.json'),
    kmc: path.join(resolvedRunDir, 'summary.json'),
    summary: path.join(resolvedRunDir, 'summary.json'),
  };
}

export function formatMietRunDetailForChat(detail) {
  const lines = [
    `Run: ${detail.id}`,
    `Status: ${detail.status}`,
    `Mode: ${detail.mode}`,
    `Material: ${detail.materialName}`,
    `Workflow: ${detail.md?.barriers?.metadata?.workflow_kind ?? 'unknown'}`,
    `Barrier source: ${detail.md?.barriers?.metadata?.barrier_source_mode ?? 'unknown'}`,
  ];

  const events = detail.md?.barriers?.events ?? [];
  if (events.length) {
    lines.push('Barriers:');
    for (const event of events.slice(0, 8)) {
      lines.push(`- ${event.species}: ${Number(event.barrier_ev).toFixed(6)} eV (${event.barrier_source ?? 'unknown'})`);
    }
  }

  if (detail.summary) {
    lines.push('', detail.summary.slice(0, 1800));
  }

  return lines.join('\n');
}

export function formatMietRunArtifactsForChat(detail) {
  const artifacts = detail.manifest?.artifacts ?? [];
  if (!artifacts.length) return `Run ${detail.id} has no archived artifacts yet.`;
  return [
    `Artifacts for ${detail.id}:`,
    ...artifacts.slice(0, 80).map((item) => `- ${item.path}`),
  ].join('\n');
}

export async function formatMietRunLogsForChat(runDir, {
  target = 'auto',
  projectRoot = null,
} = {}) {
  const snapshot = await loadRunSnapshot(runDir).catch(() => null);
  const resolvedRunDir = path.resolve(runDir);
  const candidates = snapshot?.summaryJson && !snapshot?.state
    ? summaryLogCandidates(resolvedRunDir, snapshot.summaryJson)
    : {
        md: path.join(resolvedRunDir, 'artifacts', 'md', 'md_execution.log'),
        kmc: path.join(resolvedRunDir, 'artifacts', 'kmc', 'log.spparks'),
        summary: path.join(resolvedRunDir, 'explain', 'summary.md'),
      };

  let selected = target;
  if (selected === 'auto') {
    selected = ['md', 'kmc', 'summary'].find((key) => fs.existsSync(candidates[key])) ?? 'summary';
  }

  const filePath = candidates[selected];
  if (!filePath || !fs.existsSync(filePath)) {
    return `No ${selected} log found for ${path.basename(resolvedRunDir)}.`;
  }

  const text = await fsp.readFile(filePath, 'utf-8');
  const lines = text.split(/\r?\n/);
  return [
    `Log excerpt (${selected})`,
    `Path: ${projectRelativePath(projectRoot, filePath)}`,
    '',
    lines.slice(-80).join('\n'),
  ].join('\n');
}

export function renderRunSnapshot(snapshot) {
  if (!snapshot.state && snapshot.summaryJson) {
    const payload = snapshot.summaryJson;
    const kmc = payload.kmc || {};
    const parsedRun = kmc.parsed_run || {};
    const barrier = kmc.barrier_eV ?? payload.barrier_eV ?? null;
    const sourceCase = payload.source_case_dir ?? null;
    return [
      `Run directory: ${snapshot.runDir}`,
      `Status: ${payload.status ?? 'unknown'}`,
      sourceCase ? `Source case: ${sourceCase}` : null,
      barrier !== null ? `Barrier: ${barrier} eV` : null,
      parsedRun.accepted_events !== undefined ? `Accepted events: ${parsedRun.accepted_events}` : null,
      parsedRun.final_time !== undefined ? `Final time: ${parsedRun.final_time}` : null,
      `Summary file: ${path.join(snapshot.runDir, 'summary.json')}`,
    ]
      .filter(Boolean)
      .join('\n\n');
  }
  const stepLines = Object.entries(snapshot.state?.steps ?? {})
    .map(([stepId, record]) => `- ${stepId}: ${record.status}`)
    .join('\n');
  const fileCount = snapshot.manifest?.files?.length ?? 0;
  return [
    `Run directory: ${snapshot.runDir}`,
    stepLines ? `Steps:\n${stepLines}` : 'Steps: unavailable',
    `Archived files: ${fileCount}`,
    snapshot.summary ? `\nSummary preview:\n${snapshot.summary.slice(0, 1200)}` : '',
  ]
    .filter(Boolean)
    .join('\n\n');
}

function textToolResult(text, details) {
  return {
    content: [
      {
        type: 'text',
        text,
      },
    ],
    details,
  };
}

function buildMietToolEntry({
  name,
  description,
  parameters = TOOL_PARAM_SCHEMA,
  options = undefined,
  execute,
}) {
  return {
    definition: {
      name,
      description,
      parameters,
      async execute(_id, params) {
        return await execute(params || {});
      },
    },
    options,
  };
}

function buildMietJsonTextResult(output) {
  return textToolResult(JSON.stringify(output.payload, null, 2), output);
}

function buildMietRunSnapshotToolResult(output, summary) {
  return textToolResult(renderRunSnapshot(summary), {
    ...output,
    snapshot: summary,
  });
}

export const MIET_PYTHON_MCP_TOOL_DEFINITIONS = [
  {
    name: 'miet_runtime_doctor',
    description: 'Check whether the local model, LAMMPS runtime, MoRe case path, and misa-kmc binary are ready.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: { project_root: { type: 'string' } },
    },
  },
  {
    name: 'miet_list_runs',
    description: 'List recent mietclaw run directories and their status.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        output_dir: { type: 'string' },
        limit: { type: 'integer', minimum: 1, maximum: 50 },
      },
    },
  },
  {
    name: 'miet_inspect_run',
    description: 'Inspect one mietclaw run and summarize its current state.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        run_dir: { type: 'string' },
        run_name: { type: 'string' },
        output_dir: { type: 'string' },
      },
    },
  },
  {
    name: 'miet_get_logs',
    description: 'Read the MD, KMC, or summary log excerpt for a run.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        run_dir: { type: 'string' },
        run_name: { type: 'string' },
        output_dir: { type: 'string' },
        target: { type: 'string', enum: ['auto', 'md', 'kmc', 'summary'] },
        max_lines: { type: 'integer', minimum: 1, maximum: 400 },
      },
    },
  },
  {
    name: 'miet_list_artifacts',
    description: 'List archived artifacts for a run.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        run_dir: { type: 'string' },
        run_name: { type: 'string' },
        output_dir: { type: 'string' },
        limit: { type: 'integer', minimum: 1, maximum: 500 },
      },
    },
  },
  {
    name: 'miet_autonomy_draft',
    description: 'Turn a natural-language MD/KMC task into a generated draft workspace.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        prompt: { type: 'string' },
        provider: { type: 'string' },
        workspace_root: { type: 'string' },
        mode: { type: 'string', enum: ['md_only', 'kmc_only', 'md_to_kmc_chain'] },
        template_path: { type: 'string' },
        job_id: { type: 'string' },
        material_name: { type: 'string' },
      },
      required: ['prompt'],
    },
  },
  {
    name: 'miet_autonomy_run',
    description: 'Draft, validate, and optionally run a natural-language MD/KMC task.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        prompt: { type: 'string' },
        provider: { type: 'string' },
        workspace_root: { type: 'string' },
        output_dir: { type: 'string' },
        mode: { type: 'string', enum: ['md_only', 'kmc_only', 'md_to_kmc_chain'] },
        template_path: { type: 'string' },
        job_id: { type: 'string' },
        material_name: { type: 'string' },
        dry_run_only: { type: 'boolean' },
        resume_existing: { type: 'boolean' },
        overwrite_existing: { type: 'boolean' },
      },
      required: ['prompt'],
    },
  },
  {
    name: 'miet_plan_job',
    description: 'Load an existing job spec and return the deterministic execution plan.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: { job_spec_path: { type: 'string' } },
      required: ['job_spec_path'],
    },
  },
  {
    name: 'miet_run_job',
    description: 'Run an existing job spec.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        job_spec_path: { type: 'string' },
        output_dir: { type: 'string' },
        dry_run: { type: 'boolean' },
      },
      required: ['job_spec_path'],
    },
  },
  {
    name: 'miet_kmc_bridge',
    description: 'Turn event.json plus neb.txt or a barrier value into a KMC lookup file and optionally validate it with misa-kmc.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        event_json: { type: 'string' },
        neb_txt: { type: 'string' },
        barrier: { type: 'number' },
        workdir: { type: 'string' },
        validate: { type: 'boolean' },
      },
      required: ['event_json', 'workdir'],
    },
  },
  {
    name: 'miet_moire_run',
    description: 'Run a real MoRe LAMMPS NEB case on this computer, auto-generate a KMC seed event if needed, then write a repo-compatible KMC input and continue the simulation with the repo misa-kmc binary.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        event_json: { type: 'string' },
        case_dir: { type: 'string' },
        workdir: { type: 'string' },
        validate: { type: 'boolean' },
        kmc_seed: { type: 'integer' },
        kmc_seeds: { type: 'array', items: { type: 'integer' } },
        ovito: { type: 'boolean' },
        ovito_python: { type: 'string' },
      },
      required: ['case_dir', 'workdir'],
    },
  },
  {
    name: 'miet_moire_compare',
    description: 'Compare multiple MoRe event.json files on one local case, and optionally continue each event into the repo misa-kmc stage.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        case_dir: { type: 'string' },
        event_jsons: { type: 'array', items: { type: 'string' } },
        workdir: { type: 'string' },
        validate: { type: 'boolean' },
        kmc_seed: { type: 'integer' },
        kmc_seeds: { type: 'array', items: { type: 'integer' } },
        ovito: { type: 'boolean' },
        ovito_python: { type: 'string' },
        lammps_only: { type: 'boolean' },
      },
      required: ['case_dir', 'event_jsons', 'workdir'],
    },
  },
  {
    name: 'miet_moire_diffusion_sweep',
    description: 'Run one MoRe LAMMPS barrier from an event.json, then sweep repo misa-kmc across temperatures and summarize diffusion coefficient vs temperature.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        event_json: { type: 'string' },
        case_dir: { type: 'string' },
        workdir: { type: 'string' },
        temperatures_k: { type: 'array', items: { type: 'number' } },
        validate: { type: 'boolean' },
        kmc_seed: { type: 'integer' },
        kmc_seeds: { type: 'array', items: { type: 'integer' } },
        run_time: { type: 'string' },
        stats_step: { type: 'string' },
        ovito: { type: 'boolean' },
        ovito_python: { type: 'string' },
      },
      required: ['event_json', 'case_dir', 'workdir'],
    },
  },
  {
    name: 'miet_moire_lammps',
    description: 'Run only the local MoRe LAMMPS NEB case on this computer and return the resulting neb.txt plus parsed barrier.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        event_json: { type: 'string' },
        case_dir: { type: 'string' },
        workdir: { type: 'string' },
        ovito: { type: 'boolean' },
        ovito_python: { type: 'string' },
      },
      required: ['case_dir', 'workdir'],
    },
  },
  {
    name: 'miet_moire_kmc',
    description: 'Generate a repo-compatible KMC initial state directly from MoRe event.json or, if none is provided, auto-generate a seed event from data.lmp; then write a transparent repo misa-kmc input from a MoRe LAMMPS barrier and run local repo KMC.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        event_json: { type: 'string' },
        barrier_eV: { type: 'number' },
        workdir: { type: 'string' },
        data_lmp: { type: 'string' },
        kmc_seed: { type: 'integer' },
        kmc_seeds: { type: 'array', items: { type: 'integer' } },
        ovito: { type: 'boolean' },
        ovito_python: { type: 'string' },
      },
      required: ['barrier_eV', 'workdir'],
    },
  },
];

export const MIET_PLUGIN_EXTENSION_TOOL_DEFINITIONS = [
  {
    name: 'miet_resume_job',
    description: 'Resume a previously started mietclaw run from its existing state file.',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        run_dir: { type: 'string' },
        job_spec_path: { type: 'string' },
        job_spec: { type: 'object', additionalProperties: true },
        job_name_hint: { type: 'string' },
        output_dir: { type: 'string' },
        dry_run: { type: 'boolean' },
      },
      required: ['run_dir'],
    },
    options: { optional: true },
  },
];

const MIET_OPENCLAW_OPTIONAL_TOOL_NAMES = new Set([
  'miet_run_job',
  'miet_moire_run',
  'miet_moire_compare',
  'miet_moire_diffusion_sweep',
  'miet_moire_lammps',
  'miet_moire_kmc',
  'miet_resume_job',
]);

export const MIET_TOOL_DEFINITIONS = [
  ...MIET_PYTHON_MCP_TOOL_DEFINITIONS,
  ...MIET_PLUGIN_EXTENSION_TOOL_DEFINITIONS,
].map((tool) => ({
  ...tool,
  parameters: tool.parameters ?? tool.inputSchema ?? TOOL_PARAM_SCHEMA,
  options: tool.options ?? (MIET_OPENCLAW_OPTIONAL_TOOL_NAMES.has(tool.name) ? { optional: true } : undefined),
}));

export function listMietClawToolNames() {
  return MIET_TOOL_DEFINITIONS.map((tool) => tool.name);
}

export function createMietClawToolHandlers(api) {
  return {
    async miet_runtime_doctor(params) {
      const output = await runMietJsonCli(api, {
        subcommand: 'doctor',
        projectRoot: params.project_root,
      });
      return buildMietJsonTextResult(output);
    },
    async miet_list_runs(params) {
      const output = await runMietJsonCli(api, {
        subcommand: 'runs',
        outputDir: params.output_dir,
        limit: params.limit,
      });
      return buildMietJsonTextResult(output);
    },
    async miet_autonomy_draft(params) {
      const output = await runMietAutonomyCli(api, {
        subcommand: 'autonomy-draft',
        prompt: params.prompt,
        provider: params.provider,
        workspaceRoot: params.workspace_root,
        mode: params.mode,
        templatePath: params.template_path,
        jobNameHint: params.job_id ?? params.job_name_hint,
        materialName: params.material_name,
      });
      return buildMietJsonTextResult(output);
    },
    async miet_autonomy_run(params) {
      const output = await runMietAutonomyCli(api, {
        subcommand: 'autonomy-run',
        prompt: params.prompt,
        provider: params.provider,
        outputDir: params.output_dir,
        workspaceRoot: params.workspace_root,
        mode: params.mode,
        templatePath: params.template_path,
        jobNameHint: params.job_id ?? params.job_name_hint,
        materialName: params.material_name,
        dryRun: Boolean(params.dry_run_only ?? params.dry_run),
        resumeExisting: Boolean(params.resume_existing),
        overwriteExisting: Boolean(params.overwrite_existing),
      });
      return buildMietJsonTextResult(output);
    },
    async miet_plan_job(params) {
      const output = await runMietCli(api, {
        subcommand: 'plan',
        jobSpecPath: params.job_spec_path,
        jobSpec: params.job_spec,
        jobNameHint: params.job_name_hint,
      });
      return textToolResult(JSON.stringify(output.plan, null, 2), output);
    },
    async miet_run_job(params) {
      const output = await runMietCli(api, {
        subcommand: 'run',
        jobSpecPath: params.job_spec_path,
        jobSpec: params.job_spec,
        jobNameHint: params.job_name_hint,
        outputDir: params.output_dir,
        dryRun: Boolean(params.dry_run),
        overwriteExisting: Boolean(params.overwrite_existing),
      });
      const summary = await loadRunSnapshot(output.runDir);
      return buildMietRunSnapshotToolResult(output, summary);
    },
    async miet_resume_job(params) {
      const output = await runMietCli(api, {
        subcommand: 'run',
        jobSpecPath: params.job_spec_path,
        jobSpec: params.job_spec,
        jobNameHint: params.job_name_hint,
        outputDir: params.output_dir,
        dryRun: Boolean(params.dry_run),
        runDir: params.run_dir,
        resume: true,
      });
      const summary = await loadRunSnapshot(output.runDir);
      return buildMietRunSnapshotToolResult(output, summary);
    },
    async miet_inspect_run(params) {
      const output = await runMietJsonCli(api, {
        subcommand: 'inspect',
        runDir: params.run_dir,
        runName: params.run_name,
        outputDir: params.output_dir,
      });
      return buildMietJsonTextResult(output);
    },
    async miet_get_logs(params) {
      const output = await runMietJsonCli(api, {
        subcommand: 'logs',
        runDir: params.run_dir,
        runName: params.run_name,
        outputDir: params.output_dir,
        target: params.target,
        maxLines: params.max_lines,
      });
      return buildMietJsonTextResult(output);
    },
    async miet_list_artifacts(params) {
      const output = await runMietJsonCli(api, {
        subcommand: 'artifacts',
        runDir: params.run_dir,
        runName: params.run_name,
        outputDir: params.output_dir,
        limit: params.limit,
      });
      return buildMietJsonTextResult(output);
    },
    async miet_kmc_bridge(params) {
      const output = await runMietJsonCli(api, {
        subcommand: 'bridge',
        outputDir: params.output_dir,
        eventJson: params.event_json,
        nebTxt: params.neb_txt,
        barrier: params.barrier,
        workdir: params.workdir,
        validate: params.validate,
      });
      return buildMietJsonTextResult(output);
    },
    async miet_moire_run(params) {
      const output = await runMietJsonCli(api, {
        subcommand: 'moire-run',
        outputDir: params.output_dir,
        eventJson: params.event_json,
        caseDir: params.case_dir,
        workdir: params.workdir,
        validate: params.validate,
        kmcSeed: params.kmc_seed,
        kmcSeeds: params.kmc_seeds,
        ovito: params.ovito,
        ovitoPython: params.ovito_python,
      });
      return buildMietJsonTextResult(output);
    },
    async miet_moire_compare(params) {
      const output = await runMietJsonCli(api, {
        subcommand: 'moire-compare',
        outputDir: params.output_dir,
        caseDir: params.case_dir,
        eventJsons: params.event_jsons,
        workdir: params.workdir,
        validate: params.validate,
        kmcSeed: params.kmc_seed,
        kmcSeeds: params.kmc_seeds,
        ovito: params.ovito,
        ovitoPython: params.ovito_python,
        lammpsOnly: params.lammps_only,
      });
      return buildMietJsonTextResult(output);
    },
    async miet_moire_diffusion_sweep(params) {
      const output = await runMietJsonCli(api, {
        subcommand: 'moire-diffusion-sweep',
        outputDir: params.output_dir,
        eventJson: params.event_json,
        caseDir: params.case_dir,
        workdir: params.workdir,
        temperaturesK: params.temperatures_k,
        validate: params.validate,
        kmcSeed: params.kmc_seed,
        kmcSeeds: params.kmc_seeds,
        runTime: params.run_time,
        statsStep: params.stats_step,
        ovito: params.ovito,
        ovitoPython: params.ovito_python,
      });
      return buildMietJsonTextResult(output);
    },
    async miet_moire_lammps(params) {
      const output = await runMietJsonCli(api, {
        subcommand: 'moire-lammps',
        outputDir: params.output_dir,
        eventJson: params.event_json,
        caseDir: params.case_dir,
        workdir: params.workdir,
        ovito: params.ovito,
        ovitoPython: params.ovito_python,
      });
      return buildMietJsonTextResult(output);
    },
    async miet_moire_kmc(params) {
      const output = await runMietJsonCli(api, {
        subcommand: 'moire-kmc',
        outputDir: params.output_dir,
        eventJson: params.event_json,
        barrierEV: params.barrier_eV,
        workdir: params.workdir,
        dataLmp: params.data_lmp,
        kmcSeed: params.kmc_seed,
        kmcSeeds: params.kmc_seeds,
        ovito: params.ovito,
        ovitoPython: params.ovito_python,
      });
      return buildMietJsonTextResult(output);
    },
  };
}

export function createMietClawToolCatalog(api) {
  const handlers = createMietClawToolHandlers(api);
  return MIET_TOOL_DEFINITIONS.map((tool) => ({
    ...tool,
    execute: handlers[tool.name],
  }));
}

export function createMietClawTools(api) {
  return createMietClawToolCatalog(api).map((tool) => buildMietToolEntry(tool));
}

export function registerMietClawTools(api) {
  for (const tool of createMietClawTools(api)) {
    api.registerTool(tool.definition, tool.options);
  }
}
