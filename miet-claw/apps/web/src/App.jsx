import { useEffect, useMemo, useRef, useState } from 'react';

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      'content-type': 'application/json',
      ...(options.headers ?? {}),
    },
    ...options,
  });

  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function commandText(value) {
  if (Array.isArray(value)) return value.join(' ');
  return String(value ?? '');
}

function barrierHintText(barriers) {
  if (!barriers || typeof barriers !== 'object') return null;
  const items = Object.entries(barriers);
  if (!items.length) return null;
  return items.map(([species, value]) => `${species}=${Number(value).toFixed(3)} eV`).join(' · ');
}

function buildCommandRows(report) {
  const rows = [];
  if (report?.job_spec?.md?.command?.length) {
    rows.push({ label: 'MD command', value: commandText(report.job_spec.md.command) });
  }
  if (report?.job_spec?.kmc?.command?.length) {
    rows.push({ label: 'KMC command', value: commandText(report.job_spec.kmc.command) });
  }

  const generatedFiles = report?.generated_files ?? {};
  [
    ['Plan script', generatedFiles.plan_script],
    ['Dry-run script', generatedFiles.dry_run_script],
    ['Run script', generatedFiles.run_script],
  ].forEach(([label, value]) => {
    if (value) rows.push({ label, value });
  });

  return rows;
}

function buildGeneratedFileRows(report) {
  return Object.entries(report?.generated_files ?? {}).map(([label, value]) => ({ label, value }));
}

function buildFactRows(report) {
  const facts = report?.facts ?? {};
  const rows = [];
  if (facts.temperature_k !== undefined && facts.temperature_k !== null) {
    rows.push({ label: 'Temperature', value: `${facts.temperature_k} K` });
  }
  if (facts.owner) {
    rows.push({ label: 'Owner', value: facts.owner });
  }
  const barriers = barrierHintText(facts.barrier_hints_ev);
  if (barriers) {
    rows.push({ label: 'Barrier hints', value: barriers });
  }
  if (report?.selected_template?.file_name) {
    rows.push({ label: 'Template', value: report.selected_template.file_name });
  }
  if (report?.provider_used) {
    rows.push({ label: 'Provider', value: report.provider_used });
  }
  if (report?.workspace_dir) {
    rows.push({ label: 'Workspace', value: report.workspace_dir });
  }
  return rows;
}

function buildToolTraceRows(summary) {
  if (!summary || typeof summary !== 'object') return [];
  const rows = [];
  if (summary.toolStepCount !== null && summary.toolStepCount !== undefined) {
    rows.push({ label: 'Tool steps', value: String(summary.toolStepCount) });
  }
  if (summary.finishStatus) {
    rows.push({ label: 'Finish status', value: summary.finishStatus });
  }
  if (summary.finishReason) {
    rows.push({ label: 'Finish reason', value: summary.finishReason });
  }
  if (summary.latestAssistantFinalAnswer) {
    rows.push({ label: 'Latest assistant answer', value: summary.latestAssistantFinalAnswer });
  }
  if (summary.toolActions?.length) {
    rows.push({ label: 'Tool actions', value: summary.toolActions.join(' → ') });
  }
  return rows;
}

function summarizeToolEvidenceItem(item) {
  if (!item || typeof item !== 'object') return 'Unknown tool evidence';
  const action = item.action ?? 'unknown';
  const source = item.source ? ` via ${item.source}` : '';
  const suffix = item.ok === false ? ' (error)' : '';
  return `step ${item.step ?? '?'} · ${action}${source}${suffix}`;
}

function summarizeTraceReplayItem(item) {
  if (!item || typeof item !== 'object') return 'unknown trace event';
  if (item.kind === 'tool_use') {
    return `${item.index}. tool_use · ${item.action ?? 'unknown'} (${item.source ?? 'unknown'})`;
  }
  if (item.kind === 'permission_decision') {
    return `${item.index}. permission · ${item.action ?? 'unknown'} → ${item.decision ?? 'unknown'}`;
  }
  if (item.kind === 'tool_result_block') {
    return `${item.index}. tool_result · ${item.action ?? 'unknown'} (${item.ok === false ? 'error' : 'ok'})`;
  }
  if (item.kind === 'assistant_action_block') {
    const toolActions = Array.isArray(item.toolActions) ? item.toolActions.join(', ') : 'none';
    return `${item.index}. assistant block · ${toolActions}`;
  }
  if (item.kind === 'turn_finish') {
    return `${item.index}. finish · ${item.status ?? 'unknown'}`;
  }
  return `${item.index}. ${item.kind ?? 'event'}`;
}

function normalizeToolTimeline(message, fallback = {}) {
  if (Array.isArray(message?.toolTimeline)) return message.toolTimeline;
  if (Array.isArray(fallback.toolTimeline)) return fallback.toolTimeline;
  const replay = Array.isArray(message?.toolTraceReplay)
    ? message.toolTraceReplay
    : (Array.isArray(fallback.toolTraceReplay) ? fallback.toolTraceReplay : []);
  return replay.map((item) => ({
    index: item.index ?? null,
    kind: item.kind ?? 'event',
    stage: item.kind ?? 'event',
    status:
      item.kind === 'turn_finish'
        ? (item.status === 'finish' ? 'completed' : (item.status === 'error' ? 'failed' : 'info'))
        : (item.kind === 'permission_decision'
            ? (item.decision === 'allow' ? 'completed' : 'blocked')
            : (item.kind === 'tool_result_block'
                ? (item.ok === false ? 'failed' : 'completed')
                : 'info')),
    title: summarizeTraceReplayItem(item),
    detail: item.reason ?? item.outputPreview ?? item.reply ?? null,
    source: item.source ?? null,
  }));
}

function uniqueList(values) {
  return [...new Set((values ?? []).filter(Boolean))];
}

function cardFocusGroup(cardType) {
  if (cardType === 'transparency') return 'transparency';
  if (cardType === 'plan_result') return 'plan';
  if (cardType === 'runtime_snapshot') return 'runtime';
  if (cardType === 'run_result') return 'run_result';
  return null;
}

function timelineGroupsForItem(item, message) {
  const action = item?.action ?? null;
  const groups = [];

  if (item?.kind === 'assistant_turn' || item?.kind === 'assistant_action_block') {
    if (message?.report) groups.push('transparency');
    else if (message?.plan) groups.push('plan');
    else groups.push('runtime');
    return uniqueList(groups);
  }

  if (item?.kind === 'permission_decision' || item?.kind === 'turn_finish') {
    return ['runtime'];
  }

  if (item?.kind === 'tool_use' || item?.kind === 'tool_result' || item?.kind === 'tool_result_block') {
    if ((action === 'inspect' || action === 'logs' || action === 'artifacts' || action === 'run' || action === 'resume') && message?.runDetail) {
      groups.push('run_result');
    }
    if ((action === 'draft' || action === 'autonomy-draft' || action === 'run' || action === 'autonomy-run') && message?.report) {
      groups.push('transparency');
    }
    if ((action === 'plan' || action === 'plan-job') && message?.plan) {
      groups.push('plan');
    }
    groups.push('runtime');
  }

  if (!groups.length) groups.push('runtime');
  return uniqueList(groups);
}

function timelineGroupLabel(group) {
  switch (group) {
    case 'transparency':
      return 'draft evidence';
    case 'plan':
      return 'execution plan';
    case 'runtime':
      return 'runtime snapshot';
    case 'run_result':
      return 'execution result';
    default:
      return group ?? 'linked card';
  }
}

function focusedDetailForGroup(group, item) {
  if (!group || !item) return null;
  if (group === 'transparency') {
    if (item.kind === 'assistant_turn' || item.kind === 'assistant_action_block') return 'notes';
    if (item.kind === 'tool_use' || item.kind === 'tool_result' || item.kind === 'tool_result_block') return 'toolEvidence';
    return 'toolTraceReplay';
  }
  if (group === 'plan') return 'planJson';
  if (group === 'runtime') {
    if (item.kind === 'tool_result' || item.kind === 'tool_result_block') return 'toolEvidence';
    if (item.kind === 'assistant_turn' || item.kind === 'assistant_action_block') return 'payload';
    return 'toolTraceReplay';
  }
  if (group === 'run_result') {
    if (item.kind === 'tool_result' || item.kind === 'tool_result_block') return 'detailJson';
    return 'summary';
  }
  return null;
}

function createAssistantMessageId() {
  return `assistant-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function normalizeAssistantMessage({
  message = null,
  localModel = null,
  selectedModel = null,
  fallback = {},
} = {}) {
  const kind = message?.kind ?? fallback.kind ?? 'chat';
  const cards = Array.isArray(message?.cards)
    ? message.cards
    : (Array.isArray(fallback.cards) ? fallback.cards : []);
  const report = message?.currentReport ?? fallback.report ?? null;
  const plan = message?.currentPlan ?? fallback.plan ?? null;
  const notes = message?.currentReportNotes ?? fallback.notes ?? null;
  const runDetail = message?.currentRunDetail ?? fallback.runDetail ?? null;
  const previewConfirmable = Boolean(message?.previewConfirmable ?? fallback.previewConfirmable);
  const previewExecuted = Boolean(message?.previewExecuted ?? runDetail ?? fallback.previewExecuted);

  return {
    id: fallback.id ?? createAssistantMessageId(),
    role: message?.role ?? fallback.role ?? 'assistant',
    kind,
    content: message?.content ?? fallback.content ?? '',
    model:
      kind === 'chat'
        ? (message?.model ?? localModel?.defaultModel ?? selectedModel ?? fallback.model ?? null)
        : null,
    progress: message?.progress ?? fallback.progress ?? [],
    session: message?.session ?? fallback.session ?? null,
    current: message?.current ?? fallback.current ?? null,
    report,
    plan,
    notes,
    runDetail,
    toolTraceSummary: message?.toolTraceSummary ?? fallback.toolTraceSummary ?? null,
    toolEvidence: Array.isArray(message?.toolEvidence) ? message.toolEvidence : (Array.isArray(fallback.toolEvidence) ? fallback.toolEvidence : []),
    toolTraceReplay: Array.isArray(message?.toolTraceReplay) ? message.toolTraceReplay : (Array.isArray(fallback.toolTraceReplay) ? fallback.toolTraceReplay : []),
    toolTimeline: normalizeToolTimeline(message, fallback),
    toolTraceId: message?.toolTraceId ?? fallback.toolTraceId ?? null,
    prompt: fallback.prompt ?? null,
    previewConfirmable,
    previewExecuted,
    jobSpecPath: message?.jobSpecPath ?? report?.generated_files?.job_spec ?? fallback.jobSpecPath ?? null,
    transcriptPath: message?.transcriptPath ?? message?.session?.transcriptPath ?? fallback.transcriptPath ?? null,
    cards,
  };
}

function TimelineStatusPill({ status }) {
  const normalized = status ?? 'info';
  return <span className={`pill timeline-pill ${normalized}`}>{normalized}</span>;
}

function TimelineLinkButton({ active = false, label, onClick }) {
  return (
    <button type="button" className={`secondary-button ${active ? 'active' : ''}`} onClick={onClick}>
      {label}
    </button>
  );
}

function ToolTimelineCard({
  message,
  selectedTimelineEvent = null,
  highlightedCardGroup = null,
  onSelectTimelineItem = null,
}) {
  const timeline = Array.isArray(message.toolTimeline) ? message.toolTimeline : [];
  const [selectedEvidenceKey, setSelectedEvidenceKey] = useState(null);
  const [evidenceByKey, setEvidenceByKey] = useState({});
  const [loadingEvidenceKey, setLoadingEvidenceKey] = useState(null);
  const [evidenceError, setEvidenceError] = useState(null);
  if (!timeline.length) return null;

  async function openTranscriptEvidence(item) {
    const transcriptRef = item?.transcriptRef;
    if (!message.transcriptPath || !transcriptRef?.eventIndex) return;
    const evidenceKey = `${transcriptRef.traceId ?? message.toolTraceId ?? 'trace'}:${transcriptRef.eventIndex}`;
    if (evidenceByKey[evidenceKey]) {
      setSelectedEvidenceKey((current) => (current === evidenceKey ? null : evidenceKey));
      setEvidenceError(null);
      return;
    }
    setLoadingEvidenceKey(evidenceKey);
    setEvidenceError(null);
    try {
      const result = await fetchJson('/api/actions/transcript-evidence', {
        method: 'POST',
        body: JSON.stringify({
          transcriptPath: message.transcriptPath,
          traceId: transcriptRef.traceId ?? message.toolTraceId ?? null,
          eventIndex: transcriptRef.eventIndex,
        }),
      });
      setEvidenceByKey((current) => ({
        ...current,
        [evidenceKey]: result.evidence,
      }));
      setSelectedEvidenceKey(evidenceKey);
    } catch (error) {
      setEvidenceError(error.message);
    } finally {
      setLoadingEvidenceKey(null);
    }
  }

  return (
    <section className="meta-card">
      <div className="meta-header">
        <div>
          <div className="meta-title">Agent timeline</div>
          <div className="meta-subtitle">Replay of the latest tool-backed turn</div>
        </div>
        <div className="pill-row">
          <span className="pill">{timeline.length} events</span>
          {message.transcriptPath ? <span className="pill accent">transcript attached</span> : null}
        </div>
      </div>

      <ol className="timeline-list">
        {timeline.map((item) => (
          (() => {
            const relatedGroups = timelineGroupsForItem(item, message);
            const isSelected = selectedTimelineEvent === item.index;
            const isLinked = !isSelected && highlightedCardGroup && relatedGroups.includes(highlightedCardGroup);
            return (
              <li
                key={`${item.kind}-${item.index}`}
                className={`timeline-item ${isSelected ? 'selected-focus' : ''} ${isLinked ? 'linked-focus' : ''}`}
              >
            <div className="timeline-item-head">
              <div className="timeline-item-title">
                <span className="timeline-index">#{item.index ?? '?'}</span>
                <span>{item.title ?? item.kind ?? 'event'}</span>
              </div>
              <div className="pill-row">
                {item.stage ? <span className="pill">{item.stage}</span> : null}
                <TimelineStatusPill status={item.status} />
              </div>
            </div>
            {item.detail ? <div className="timeline-detail">{item.detail}</div> : null}
            {(item.action || item.source) ? (
              <div className="timeline-meta">
                {item.action ? <span>action: {item.action}</span> : null}
                {item.source ? <span>source: {item.source}</span> : null}
                {item.transcriptRef?.label ? <span>ref: {item.transcriptRef.label}</span> : null}
                {relatedGroups.length ? <span>cards: {relatedGroups.map(timelineGroupLabel).join(', ')}</span> : null}
              </div>
            ) : null}
            {(message.transcriptPath && item.transcriptRef?.eventIndex) || relatedGroups.length ? (
              <div className="timeline-actions pill-row">
                {relatedGroups.length && onSelectTimelineItem ? (
                  <TimelineLinkButton
                    active={isSelected}
                    label={isSelected ? 'Clear related cards' : 'Highlight related cards'}
                    onClick={() => onSelectTimelineItem(item, relatedGroups)}
                  />
                ) : null}
                {message.transcriptPath && item.transcriptRef?.eventIndex ? (
                <button
                  type="button"
                  className="secondary-button"
                  disabled={loadingEvidenceKey === `${item.transcriptRef.traceId ?? message.toolTraceId ?? 'trace'}:${item.transcriptRef.eventIndex}`}
                  onClick={() => openTranscriptEvidence(item)}
                >
                  {loadingEvidenceKey === `${item.transcriptRef.traceId ?? message.toolTraceId ?? 'trace'}:${item.transcriptRef.eventIndex}`
                    ? 'Loading transcript evidence…'
                    : 'Open transcript evidence'}
                </button>
                ) : null}
              </div>
            ) : null}
            {(item.params && Object.keys(item.params).length) || item.finalAnswer || item.reply ? (
              <details className="meta-details">
                <summary>Open event details</summary>
                <pre>{JSON.stringify(item, null, 2)}</pre>
              </details>
            ) : null}
            {selectedEvidenceKey === `${item.transcriptRef?.traceId ?? message.toolTraceId ?? 'trace'}:${item.transcriptRef?.eventIndex ?? 'none'}`
              && evidenceByKey[selectedEvidenceKey] ? (
                <details className="meta-details" open>
                  <summary>Transcript evidence</summary>
                  <div className="timeline-detail">
                    {evidenceByKey[selectedEvidenceKey].lineStart && evidenceByKey[selectedEvidenceKey].lineEnd
                      ? `Transcript lines ${evidenceByKey[selectedEvidenceKey].lineStart}-${evidenceByKey[selectedEvidenceKey].lineEnd}`
                      : message.transcriptPath}
                  </div>
                  <pre>{evidenceByKey[selectedEvidenceKey].excerpt}</pre>
                </details>
              ) : null}
              </li>
            );
          })()
        ))}
      </ol>
      {evidenceError ? <div className="error-banner">{evidenceError}</div> : null}
    </section>
  );
}

function withPreviewState(message, { confirmable = false, executed = false } = {}) {
  const previewExecuted = Boolean(executed);
  const previewConfirmable = Boolean(confirmable && message.jobSpecPath && !previewExecuted);
  return {
    ...message,
    previewConfirmable,
    previewExecuted,
    cards: Array.isArray(message.cards)
      ? message.cards.map((card) => (
          card.type === 'transparency'
            ? { ...card, previewConfirmable, previewExecuted }
            : card
        ))
      : message.cards,
  };
}

function buildPreviewMessage({ prompt, data, confirmable }) {
  if (data.message) {
    return {
      ...withPreviewState(
        normalizeAssistantMessage({ message: data.message, localModel: null, selectedModel: null }),
        {
          confirmable,
          executed: Boolean(data.message?.previewExecuted),
        },
      ),
      prompt,
    };
  }

  const payload = data.payload ?? {};
  const plan = data.plan ?? null;
  const meta = data.meta ?? {};
  return {
    ...withPreviewState(
      normalizeAssistantMessage({
        fallback: {
          kind: 'tool',
          content: confirmable
            ? `Execution preview ready for ${payload.job_id ?? 'this task'}. Nothing has run yet — review the plan below and confirm when you are ready.`
            : `Draft preview ready for ${payload.job_id ?? 'this task'}. You can review the generated plan, assumptions, and files below.`,
          prompt,
          progress: [],
          session: meta.session ?? null,
          current: meta.current ?? null,
          report: payload,
          plan: meta.currentPlan ?? plan,
          notes: meta.currentReportNotes ?? data.notes ?? null,
          runDetail: meta.currentRunDetail ?? data.detail ?? null,
          previewConfirmable: confirmable,
          previewExecuted: false,
          jobSpecPath: data.generatedSpecPath ?? payload.generated_files?.job_spec ?? null,
          transcriptPath: meta.transcriptPath ?? null,
          cards: [],
        },
      }),
      { confirmable, executed: false },
    ),
    prompt,
  };
}

function buildChatMessage(payload, selectedModel) {
  return normalizeAssistantMessage({
    message: payload.message ?? null,
    localModel: payload.localModel ?? null,
    selectedModel,
  });
}

function buildRunResultMessage({ previewMessage, result }) {
  if (result.message) {
    return withPreviewState(
      normalizeAssistantMessage({ message: result.message, localModel: null, selectedModel: null }),
      { confirmable: false, executed: true },
    );
  }

  const detail = result.detail ?? null;
  const meta = result.meta ?? {};
  return withPreviewState(
    normalizeAssistantMessage({
      fallback: {
        kind: 'tool',
        content: detail
          ? `Run complete for ${detail.id}. The result card below shows the finished steps and the evidence that was produced.`
          : 'Run finished. Review the run card below for the result.',
        progress: [],
        session: meta.session ?? previewMessage.session ?? null,
        current: meta.current ?? previewMessage.current ?? null,
        report: previewMessage.report ?? null,
        plan: previewMessage.plan ?? null,
        notes: previewMessage.notes ?? null,
        runDetail: detail,
        previewConfirmable: false,
        previewExecuted: true,
        jobSpecPath: previewMessage.jobSpecPath ?? null,
        transcriptPath: meta.transcriptPath ?? previewMessage.transcriptPath ?? null,
        cards: [],
      },
    }),
    { confirmable: false, executed: true },
  );
}


function activeKindLabel(activeKind) {
  switch (activeKind) {
    case 'report':
      return 'draft/report';
    case 'run':
      return 'run';
    case 'bridge_summary':
      return 'bridge summary';
    case 'moire_summary':
      return 'MoRe summary';
    case 'moire_compare_summary':
      return 'MoRe compare';
    case 'moire_diffusion_summary':
      return 'diffusion sweep';
    default:
      return activeKind ?? 'none';
  }
}

function runtimeSummaryPayload(current) {
  if (!current) return { title: null, payload: null, rows: [] };
  if (current.bridgeSummary) {
    const payload = current.bridgeSummary;
    return {
      title: 'Bridge snapshot',
      payload,
      rows: [
        payload.status ? { label: 'Status', value: String(payload.status) } : null,
        payload.barrier_eV !== undefined ? { label: 'Barrier', value: `${payload.barrier_eV} eV` } : null,
        payload.workdir ? { label: 'Workdir', value: String(payload.workdir) } : null,
      ].filter(Boolean),
    };
  }
  if (current.moireSummary) {
    const payload = current.moireSummary;
    return {
      title: 'MoRe workflow snapshot',
      payload,
      rows: [
        payload.status ? { label: 'Status', value: String(payload.status) } : null,
        payload.source_case_dir ? { label: 'Case', value: String(payload.source_case_dir) } : null,
        payload.barrier_eV !== undefined ? { label: 'Barrier', value: `${payload.barrier_eV} eV` } : null,
        payload.kmc?.status ? { label: 'KMC', value: String(payload.kmc.status) } : null,
      ].filter(Boolean),
    };
  }
  if (current.moireCompareSummary) {
    const payload = current.moireCompareSummary;
    return {
      title: 'MoRe compare snapshot',
      payload,
      rows: [
        payload.status ? { label: 'Status', value: String(payload.status) } : null,
        payload.case_dir ? { label: 'Case', value: String(payload.case_dir) } : null,
        Array.isArray(payload.event_runs) ? { label: 'Events', value: String(payload.event_runs.length) } : null,
      ].filter(Boolean),
    };
  }
  if (current.moireDiffusionSummary) {
    const payload = current.moireDiffusionSummary;
    return {
      title: 'Diffusion sweep snapshot',
      payload,
      rows: [
        payload.status ? { label: 'Status', value: String(payload.status) } : null,
        payload.case_dir ? { label: 'Case', value: String(payload.case_dir) } : null,
        Array.isArray(payload.temperatures_k) ? { label: 'Temperatures', value: String(payload.temperatures_k.length) } : null,
      ].filter(Boolean),
    };
  }
  return { title: null, payload: null, rows: [] };
}

function secondsSince(timestamp) {
  if (!timestamp) return null;
  const value = new Date(timestamp).getTime();
  if (!Number.isFinite(value)) return null;
  return Math.max(0, Math.floor((Date.now() - value) / 1000));
}

function relativeTimeText(timestamp) {
  const seconds = secondsSince(timestamp);
  if (seconds === null) return '—';
  if (seconds < 2) return '刚刚';
  if (seconds < 60) return `${seconds} 秒前`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分钟前`;
  return `${Math.floor(seconds / 3600)} 小时前`;
}

function heartbeatTone(timestamp) {
  const seconds = secondsSince(timestamp);
  if (seconds === null) return 'neutral';
  if (seconds <= 5) return 'success';
  if (seconds <= 20) return 'accent';
  return 'warning';
}

function activeStepFor(run, detail) {
  return detail?.activeStep ?? run?.activeStep ?? null;
}

function normalizePlanSteps(plan) {
  if (Array.isArray(plan)) return plan;
  if (Array.isArray(plan?.plan)) return plan.plan;
  return [];
}

function StepList({ plan, steps }) {
  const items = Array.isArray(steps) && steps.length ? steps : normalizePlanSteps(plan);
  if (!items.length) return <div className="empty-copy">No step timeline was available.</div>;

  return (
    <ol className="step-list">
      {items.map((item) => (
        <li key={item.id} className="step-item">
          <div className="step-topline">
            <span className="step-id">{item.id}</span>
            <span className={`step-badge ${item.status ? item.status.toLowerCase() : 'planned'}`}>
              {item.status ?? item.stage ?? 'planned'}
            </span>
          </div>
          <div className="step-copy">
            {(item.status === 'running' || item.status === 'started')
              ? (item.detail ?? item.description ?? item.outputs?.summary ?? 'No additional detail.')
              : (item.error ?? item.description ?? item.outputs?.summary ?? 'No additional detail.')}
          </div>
        </li>
      ))}
    </ol>
  );
}

function PlanResultCard({
  plan,
  isHighlighted = false,
  onFocusTimeline = null,
  timelineFocusActive = false,
  focusedDetailKey = null,
}) {
  const steps = normalizePlanSteps(plan);
  if (!steps.length) return null;

  const rows = [
    plan?.job_id ? { label: 'Job', value: plan.job_id } : null,
    plan?.mode ? { label: 'Mode', value: plan.mode } : null,
    { label: 'Planned steps', value: String(steps.length) },
  ].filter(Boolean);

  return (
    <section className={`meta-card ${isHighlighted ? 'linked-focus' : ''}`}>
      <div className="meta-header">
        <div>
          <div className="meta-title">Execution plan</div>
          <div className="meta-subtitle">
            {plan?.job_id ?? 'Planned workflow'} · {steps.length} steps
          </div>
        </div>
        <div className="pill-row">
          {plan?.mode ? <span className="pill">{plan.mode}</span> : null}
          <span className="pill accent">{steps.length} steps</span>
          {onFocusTimeline ? (
            <TimelineLinkButton
              active={timelineFocusActive}
              label={timelineFocusActive ? 'Clear timeline highlight' : 'Highlight in timeline'}
              onClick={onFocusTimeline}
            />
          ) : null}
        </div>
      </div>

      <div className="meta-grid two-up">
        <div>
          <div className="section-heading">Plan summary</div>
          <KeyValueList rows={rows} />
        </div>
        <div>
          <div className="section-heading">Planned steps</div>
          <StepList plan={plan} />
        </div>
      </div>

      <details className={`meta-details ${focusedDetailKey === 'planJson' ? 'auto-expanded' : ''}`} open={focusedDetailKey === 'planJson'}>
        <summary>Open plan JSON</summary>
        <pre>{JSON.stringify(plan, null, 2)}</pre>
      </details>
    </section>
  );
}

function KeyValueList({ rows }) {
  if (!rows.length) return <div className="empty-copy">No extra details were produced for this section.</div>;
  return (
    <div className="kv-list">
      {rows.map((row) => (
        <div key={`${row.label}-${row.value}`} className="kv-row">
          <div className="kv-label">{row.label}</div>
          <div className="kv-value">{row.value}</div>
        </div>
      ))}
    </div>
  );
}

function StringList({ items, tone = 'neutral' }) {
  if (!items.length) return <div className="empty-copy">None.</div>;
  return (
    <ul className={`bullet-list ${tone}`}>
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

function ProgressList({ lines }) {
  if (!lines.length) return null;
  return (
    <section className="meta-card">
      <div className="meta-title">Process timeline</div>
      <ul className="progress-list">
        {lines.map((line) => (
          <li key={line} className="progress-item">
            {line.replace(/^\[progress\]\s*/, '')}
          </li>
        ))}
      </ul>
    </section>
  );
}

function TransparencyCard({
  message,
  onConfirmPreview,
  previewBusy,
  isHighlighted = false,
  onFocusTimeline = null,
  timelineFocusActive = false,
  focusedDetailKey = null,
}) {
  const report = message.report;
  if (!report) return null;

  const factRows = buildFactRows(report);
  const commandRows = buildCommandRows(report);
  const generatedFileRows = buildGeneratedFileRows(report);
  const assumptions = report.assumptions ?? [];
  const warnings = report.warnings ?? [];
  const toolTraceRows = buildToolTraceRows(message.toolTraceSummary);
  const toolEvidence = Array.isArray(message.toolEvidence) ? message.toolEvidence : [];
  const toolTraceReplay = Array.isArray(message.toolTraceReplay) ? message.toolTraceReplay : [];
  const isConfirmable = Boolean(message.previewConfirmable && message.jobSpecPath && !message.previewExecuted);

  return (
    <section className={`meta-card ${isHighlighted ? 'linked-focus' : ''}`}>
      <div className="meta-header">
        <div>
          <div className="meta-title">{isConfirmable ? 'Execution preview' : 'Draft evidence'}</div>
          <div className="meta-subtitle">
            {report.job_id} · {report.mode} · {report.material_name}
          </div>
        </div>
        <div className="pill-row">
          <span className="pill">template {report.selected_template?.file_name ?? 'unknown'}</span>
          <span className="pill">provider {report.provider_used ?? 'unknown'}</span>
          {message.previewExecuted ? <span className="pill success">executed</span> : null}
          {isConfirmable ? <span className="pill accent">waiting for confirmation</span> : null}
          {onFocusTimeline ? (
            <TimelineLinkButton
              active={timelineFocusActive}
              label={timelineFocusActive ? 'Clear timeline highlight' : 'Highlight in timeline'}
              onClick={onFocusTimeline}
            />
          ) : null}
        </div>
      </div>

      <div className="meta-grid two-up">
        <div>
          <div className="section-heading">What the agent understood</div>
          <KeyValueList rows={factRows} />
        </div>
        <div>
          <div className="section-heading">Planned steps</div>
          <StepList plan={message.plan} />
        </div>
      </div>

      <div className="meta-grid two-up">
        <div>
          <div className="section-heading">Assumptions</div>
          <StringList items={assumptions} tone="neutral" />
        </div>
        <div>
          <div className="section-heading">Warnings</div>
          <StringList items={warnings} tone="warning" />
        </div>
      </div>

      <div className="meta-grid two-up">
        <div>
          <div className="section-heading">Commands and scripts</div>
          <KeyValueList rows={commandRows} />
        </div>
        <div>
          <div className="section-heading">Generated files</div>
          <KeyValueList rows={generatedFileRows} />
        </div>
      </div>

      {toolTraceRows.length || toolEvidence.length ? (
        <div className="meta-grid two-up">
          <div>
            <div className="section-heading">Tool trace summary</div>
            {toolTraceRows.length ? <KeyValueList rows={toolTraceRows} /> : <div className="empty-copy">No tool trace summary was attached.</div>}
          </div>
          <div>
            <div className="section-heading">Tool-backed evidence</div>
            {toolEvidence.length ? (
              <ul className="bullet-list neutral">
                {toolEvidence.map((item) => (
                  <li key={item.requestId ?? `${item.action}-${item.step}`}>{summarizeToolEvidenceItem(item)}</li>
                ))}
              </ul>
            ) : (
              <div className="empty-copy">No tool-backed evidence was attached to this draft.</div>
            )}
          </div>
        </div>
      ) : null}

      {toolEvidence.length ? (
        <details className={`meta-details ${focusedDetailKey === 'toolEvidence' ? 'auto-expanded' : ''}`} open={focusedDetailKey === 'toolEvidence'}>
          <summary>Open tool-backed evidence trail</summary>
          <pre>{JSON.stringify(toolEvidence, null, 2)}</pre>
        </details>
      ) : null}

      {toolTraceReplay.length ? (
        <details className={`meta-details ${focusedDetailKey === 'toolTraceReplay' ? 'auto-expanded' : ''}`} open={focusedDetailKey === 'toolTraceReplay'}>
          <summary>Open tool trace replay</summary>
          <ul className="bullet-list neutral">
            {toolTraceReplay.map((item) => (
              <li key={`${item.kind}-${item.index}`}>{summarizeTraceReplayItem(item)}</li>
            ))}
          </ul>
          <pre>{JSON.stringify(toolTraceReplay, null, 2)}</pre>
        </details>
      ) : null}

      {message.notes ? (
        <details className={`meta-details ${focusedDetailKey === 'notes' ? 'auto-expanded' : ''}`} open={focusedDetailKey === 'notes'}>
          <summary>Open autonomy notes</summary>
          <pre>{message.notes}</pre>
        </details>
      ) : null}

      <div className="meta-actions">
        {isConfirmable ? (
          <button type="button" className="send-button" disabled={previewBusy} onClick={() => onConfirmPreview(message)}>
            {previewBusy ? 'Running exact preview…' : 'Run this exact preview'}
          </button>
        ) : null}
      </div>
    </section>
  );
}

function RuntimeSnapshotCard({
  message,
  isHighlighted = false,
  onFocusTimeline = null,
  timelineFocusActive = false,
  focusedDetailKey = null,
}) {
  const current = message.current ?? null;
  const session = message.session ?? null;
  const summary = runtimeSummaryPayload(current);
  const toolTraceRows = buildToolTraceRows(message.toolTraceSummary);
  const toolEvidence = Array.isArray(message.toolEvidence) ? message.toolEvidence : [];
  const toolTraceReplay = Array.isArray(message.toolTraceReplay) ? message.toolTraceReplay : [];
  const rows = [
    current?.activeKind ? { label: 'Runtime context', value: activeKindLabel(current.activeKind) } : null,
    session?.approvalPolicy ? { label: 'Approval policy', value: session.approvalPolicy } : null,
    session?.selectedModel ? { label: 'Selected model', value: session.selectedModel } : null,
    session?.historyLength !== null && session?.historyLength !== undefined ? { label: 'Conversation items', value: String(session.historyLength) } : null,
    current?.runDir ? { label: 'Run path', value: current.runDir } : null,
    message.transcriptPath ? { label: 'Transcript', value: message.transcriptPath } : null,
    toolTraceReplay.length ? { label: 'Trace events', value: String(toolTraceReplay.length) } : null,
    ...toolTraceRows,
    ...summary.rows,
  ].filter(Boolean);

  if (!rows.length && !summary.payload && !toolTraceReplay.length) return null;

  return (
    <section className={`meta-card ${isHighlighted ? 'linked-focus' : ''}`}>
      <div className="meta-header">
        <div>
          <div className="meta-title">Runtime snapshot</div>
          <div className="meta-subtitle">
            {current?.activeKind ? activeKindLabel(current.activeKind) : 'session metadata'}
          </div>
        </div>
        <div className="pill-row">
          {current?.activeKind ? <span className="pill">{activeKindLabel(current.activeKind)}</span> : null}
          {session?.approvalPolicy ? <span className="pill accent">{session.approvalPolicy}</span> : null}
          {onFocusTimeline ? (
            <TimelineLinkButton
              active={timelineFocusActive}
              label={timelineFocusActive ? 'Clear timeline highlight' : 'Highlight in timeline'}
              onClick={onFocusTimeline}
            />
          ) : null}
        </div>
      </div>

      <KeyValueList rows={rows} />

      {toolEvidence.length ? (
        <details className={`meta-details ${focusedDetailKey === 'toolEvidence' ? 'auto-expanded' : ''}`} open={focusedDetailKey === 'toolEvidence'}>
          <summary>Open latest tool-backed evidence</summary>
          <pre>{JSON.stringify(toolEvidence, null, 2)}</pre>
        </details>
      ) : null}

      {toolTraceReplay.length ? (
        <details className={`meta-details ${focusedDetailKey === 'toolTraceReplay' ? 'auto-expanded' : ''}`} open={focusedDetailKey === 'toolTraceReplay'}>
          <summary>Open tool trace replay</summary>
          <ul className="bullet-list neutral">
            {toolTraceReplay.map((item) => (
              <li key={`${item.kind}-${item.index}`}>{summarizeTraceReplayItem(item)}</li>
            ))}
          </ul>
          <pre>{JSON.stringify(toolTraceReplay, null, 2)}</pre>
        </details>
      ) : null}

      {summary.payload ? (
        <details className={`meta-details ${focusedDetailKey === 'payload' ? 'auto-expanded' : ''}`} open={focusedDetailKey === 'payload'}>
          <summary>Open {summary.title ?? 'runtime details'}</summary>
          <pre>{JSON.stringify(summary.payload, null, 2)}</pre>
        </details>
      ) : null}
    </section>
  );
}

function RunResultCard({
  detail,
  actions = null,
  onUseCommand,
  isHighlighted = false,
  onFocusTimeline = null,
  timelineFocusActive = false,
  focusedDetailKey = null,
}) {
  if (!detail) return null;

  const barrierEvents = detail.md?.barriers?.events ?? [];
  const latestDiffusion = detail.kmc?.latestDiffusion ?? null;
  const commandActions = Array.isArray(actions)
    ? actions
    : (Array.isArray(detail.commandActions) ? detail.commandActions : []);
  const runRows = [
    { label: 'Run path', value: detail.runDir },
    { label: 'Status', value: detail.status },
    detail.executionProvenance?.label ? { label: 'Execution provenance', value: detail.executionProvenance.label } : null,
    { label: 'Mode', value: detail.mode },
    { label: 'Material', value: detail.materialName },
  ].filter(Boolean);

  if (latestDiffusion?.diffusionCoefficient !== null && latestDiffusion?.diffusionCoefficient !== undefined) {
    runRows.push({
      label: 'Latest diffusion coefficient',
      value: String(latestDiffusion.diffusionCoefficient),
    });
  }

  return (
    <section className={`meta-card result-card ${isHighlighted ? 'linked-focus' : ''}`}>
      <div className="meta-header">
        <div>
          <div className="meta-title">Execution result</div>
          <div className="meta-subtitle">
            {detail.id} · {detail.completedSteps}/{detail.totalSteps} steps completed
          </div>
        </div>
        <div className="pill-row">
          <span className={`pill ${detail.status === 'Completed' ? 'success' : detail.status === 'Failed' ? 'danger' : 'accent'}`}>
            {detail.status}
          </span>
          {onFocusTimeline ? (
            <TimelineLinkButton
              active={timelineFocusActive}
              label={timelineFocusActive ? 'Clear timeline highlight' : 'Highlight in timeline'}
              onClick={onFocusTimeline}
            />
          ) : null}
        </div>
      </div>

      <div className="meta-grid two-up">
        <div>
          <div className="section-heading">Run summary</div>
          <KeyValueList rows={runRows} />
        </div>
        <div>
          <div className="section-heading">Finished steps</div>
          <StepList steps={detail.steps} />
        </div>
      </div>

      <div className="meta-grid two-up">
        <div>
          <div className="section-heading">Barrier evidence</div>
          {barrierEvents.length ? (
            <ul className="bullet-list neutral">
              {barrierEvents.slice(0, 8).map((event) => (
                <li key={`${event.species}-${event.barrier_ev}`}>
                  {event.species}: {Number(event.barrier_ev).toFixed(6)} eV ({event.barrier_source ?? 'unknown source'})
                </li>
              ))}
            </ul>
          ) : (
            <div className="empty-copy">No barrier list was recorded for this run.</div>
          )}
        </div>
        <div>
          <div className="section-heading">Scientist-facing summary</div>
          <pre className="meta-pre">{detail.summaryPreview || detail.summary || 'No summary file was produced.'}</pre>
        </div>
      </div>

      <details className={`meta-details ${focusedDetailKey === 'summary' ? 'auto-expanded' : ''}`} open={focusedDetailKey === 'summary'}>
        <summary>Open run summary text</summary>
        <pre>{detail.summaryPreview || detail.summary || 'No summary file was produced.'}</pre>
      </details>

      <details className={`meta-details ${focusedDetailKey === 'detailJson' ? 'auto-expanded' : ''}`} open={focusedDetailKey === 'detailJson'}>
        <summary>Open run detail snapshot</summary>
        <pre>{JSON.stringify(detail, null, 2)}</pre>
      </details>

      <div className="meta-actions">
        <CommandActionButtons actions={commandActions} onUseCommand={onUseCommand} />
      </div>
    </section>
  );
}

function CollapsedCardNotice({ hiddenGroups, onClearFocus }) {
  if (!hiddenGroups.length) return null;
  return (
    <section className="meta-card collapsed-card">
      <div className="meta-header">
        <div>
          <div className="meta-title">Focused evidence mode</div>
          <div className="meta-subtitle">
            {hiddenGroups.length} unrelated card{hiddenGroups.length > 1 ? 's are' : ' is'} hidden while you focus on one timeline event.
          </div>
        </div>
        {onClearFocus ? (
          <div className="pill-row">
            <TimelineLinkButton label="Show all cards again" onClick={onClearFocus} />
          </div>
        ) : null}
      </div>
      <div className="muted-block">
        Hidden sections: {hiddenGroups.map(timelineGroupLabel).join(', ')}.
      </div>
    </section>
  );
}

function renderStructuredCard(card, {
  message,
  onConfirmPreview,
  onUseCommand,
  previewBusy,
  selectedTimelineEvent,
  selectedTimelineGroups,
  selectedCardGroup,
  selectedTimelineItem,
  onSelectTimelineItem,
  onSelectCardGroup,
}) {
  if (!card || !card.type) return null;
  const focusGroup = cardFocusGroup(card.type);
  const canFocusTimeline = Boolean(focusGroup && Array.isArray(message.toolTimeline) && message.toolTimeline.length);
  const isHighlightedFromTimeline = Boolean(focusGroup && selectedTimelineGroups.includes(focusGroup));
  const isCardFocusActive = Boolean(focusGroup && selectedCardGroup === focusGroup);
  const focusedDetailKey = focusedDetailForGroup(focusGroup, selectedTimelineItem);
  if (card.type === 'transparency') {
    return (
      <TransparencyCard
        key={`card-${card.type}-${card.jobSpecPath ?? card.report?.job_id ?? message.id}`}
        message={{
          ...message,
          report: card.report ?? null,
          plan: card.plan ?? null,
          notes: card.notes ?? null,
          previewConfirmable: Boolean(card.previewConfirmable),
          previewExecuted: Boolean(card.previewExecuted),
          jobSpecPath: card.jobSpecPath ?? null,
          transcriptPath: card.transcriptPath ?? message.transcriptPath ?? null,
          toolTraceSummary: card.toolTraceSummary ?? message.toolTraceSummary ?? null,
          toolEvidence: Array.isArray(card.toolEvidence) ? card.toolEvidence : message.toolEvidence ?? [],
          toolTraceReplay: Array.isArray(card.toolTraceReplay) ? card.toolTraceReplay : message.toolTraceReplay ?? [],
          toolTimeline: Array.isArray(card.toolTimeline) ? card.toolTimeline : message.toolTimeline ?? [],
          toolTraceId: card.toolTraceId ?? message.toolTraceId ?? null,
        }}
        onConfirmPreview={onConfirmPreview}
        previewBusy={previewBusy}
        isHighlighted={isHighlightedFromTimeline}
        onFocusTimeline={canFocusTimeline ? () => onSelectCardGroup?.(focusGroup) : null}
        timelineFocusActive={isCardFocusActive}
        focusedDetailKey={focusedDetailKey}
      />
    );
  }
  if (card.type === 'tool_timeline') {
    return (
      <ToolTimelineCard
        key={`card-${card.type}-${card.transcriptPath ?? message.id}`}
        message={{
          ...message,
          transcriptPath: card.transcriptPath ?? message.transcriptPath ?? null,
          toolTimeline: Array.isArray(card.timeline) ? card.timeline : message.toolTimeline ?? [],
          toolTraceId: card.toolTraceId ?? message.toolTraceId ?? null,
        }}
        selectedTimelineEvent={selectedTimelineEvent}
        highlightedCardGroup={selectedCardGroup}
        onSelectTimelineItem={onSelectTimelineItem}
      />
    );
  }
  if (card.type === 'runtime_snapshot') {
    return (
      <RuntimeSnapshotCard
        key={`card-${card.type}-${card.transcriptPath ?? message.id}`}
        message={{
          ...message,
          session: card.session ?? null,
          current: card.current ?? null,
          transcriptPath: card.transcriptPath ?? message.transcriptPath ?? null,
          toolTraceSummary: card.toolTraceSummary ?? message.toolTraceSummary ?? null,
          toolEvidence: Array.isArray(card.toolEvidence) ? card.toolEvidence : message.toolEvidence ?? [],
          toolTraceReplay: Array.isArray(card.toolTraceReplay) ? card.toolTraceReplay : message.toolTraceReplay ?? [],
          toolTimeline: Array.isArray(card.toolTimeline) ? card.toolTimeline : message.toolTimeline ?? [],
          toolTraceId: card.toolTraceId ?? message.toolTraceId ?? null,
        }}
        isHighlighted={isHighlightedFromTimeline}
        onFocusTimeline={canFocusTimeline ? () => onSelectCardGroup?.(focusGroup) : null}
        timelineFocusActive={isCardFocusActive}
        focusedDetailKey={focusedDetailKey}
      />
    );
  }
  if (card.type === 'plan_result') {
    return (
      <PlanResultCard
        key={`card-${card.type}-${card.plan?.job_id ?? message.id}`}
        plan={card.plan ?? null}
        isHighlighted={isHighlightedFromTimeline}
        onFocusTimeline={canFocusTimeline ? () => onSelectCardGroup?.(focusGroup) : null}
        timelineFocusActive={isCardFocusActive}
        focusedDetailKey={focusedDetailKey}
      />
    );
  }
  if (card.type === 'run_result') {
    return (
      <RunResultCard
        key={`card-${card.type}-${card.detail?.id ?? message.id}`}
        detail={card.detail ?? null}
        actions={card.actions ?? null}
        onUseCommand={onUseCommand}
        isHighlighted={isHighlightedFromTimeline}
        onFocusTimeline={canFocusTimeline ? () => onSelectCardGroup?.(focusGroup) : null}
        timelineFocusActive={isCardFocusActive}
        focusedDetailKey={focusedDetailKey}
      />
    );
  }
  return null;
}

function CommandActionButtons({ actions, onUseCommand }) {
  if (!Array.isArray(actions) || actions.length === 0) return null;
  return (
    <>
      {actions.map((action) => (
        <button
          key={action.command}
          type="button"
          className="secondary-button"
          onClick={() => onUseCommand(action.command)}
        >
          {action.label ?? action.command}
        </button>
      ))}
    </>
  );
}

function runDrawerCommandActions(run) {
  if (Array.isArray(run?.commandActions) && run.commandActions.length) return run.commandActions;
  return run?.id ? [{ command: `/inspect ${run.id}`, label: 'Prepare inspect command' }] : [];
}

function RecentRunDrawerItem({ run, onUseCommand }) {
  const [primaryAction, ...secondaryActions] = runDrawerCommandActions(run);
  if (!primaryAction) return null;

  return (
    <div className="run-drawer-item">
      <button
        type="button"
        className="run-item"
        title={primaryAction.command}
        onClick={() => onUseCommand(primaryAction.command)}
      >
        <div className="run-title">{run.id}</div>
        <div className="run-subtitle">
          {run.status} · {run.mode} · {run.completedSteps}/{run.totalSteps}
        </div>
      </button>
      {secondaryActions.length ? (
        <div className="run-item-actions">
          <CommandActionButtons actions={secondaryActions} onUseCommand={onUseCommand} />
        </div>
      ) : null}
    </div>
  );
}

function LiveRunCard({ run, detail, onUseCommand, compact = false }) {
  if (!run) return null;
  const activeStep = activeStepFor(run, detail);
  const heartbeatAt = activeStep?.heartbeatAt ?? run.updatedAt ?? null;
  const commandActions = Array.isArray(run.commandActions)
    ? run.commandActions
    : (Array.isArray(detail?.commandActions) ? detail.commandActions : []);
  const rows = [
    { label: 'Run', value: run.id },
    { label: '状态', value: run.status },
    { label: '当前步骤', value: activeStep?.id ?? '等待状态刷新' },
    { label: '进度', value: `${run.completedSteps}/${run.totalSteps}` },
    { label: '最近心跳', value: relativeTimeText(heartbeatAt) },
  ];

  if (activeStep?.pid) {
    rows.push({ label: '进程 PID', value: String(activeStep.pid) });
  }
  if (run.runDirRelative) {
    rows.push({ label: '目录', value: run.runDirRelative });
  }

  return (
    <section className={`meta-card live-run-card ${compact ? 'compact' : ''}`}>
      <div className="meta-header">
        <div>
          <div className="meta-title">任务正在运行</div>
          <div className="meta-subtitle">
            {run.materialName} · {run.mode}
          </div>
        </div>
        <div className="pill-row">
          <span className="pill accent">{run.status}</span>
          <span className={`pill ${heartbeatTone(heartbeatAt)}`}>{relativeTimeText(heartbeatAt)}</span>
        </div>
      </div>

      <KeyValueList rows={rows} />

      {activeStep?.detail ? <div className="muted-block">{activeStep.detail}</div> : null}

      {!compact && detail?.steps?.length ? (
        <div className="live-run-steps">
          <div className="section-heading">步骤时间线</div>
          <StepList steps={detail.steps} />
        </div>
      ) : null}

      <div className="meta-actions">
        <CommandActionButtons actions={commandActions} onUseCommand={onUseCommand} />
      </div>
    </section>
  );
}

function Message({ message, onConfirmPreview, onUseCommand, previewBusy }) {
  const isUser = message.role === 'user';
  const hasStructuredCards = Array.isArray(message.cards) && message.cards.length > 0;
  const hasTimeline = Array.isArray(message.toolTimeline) && message.toolTimeline.length > 0;
  const [selectedTimelineEvent, setSelectedTimelineEvent] = useState(null);
  const [selectedTimelineGroups, setSelectedTimelineGroups] = useState([]);
  const [selectedCardGroup, setSelectedCardGroup] = useState(null);
  const selectedTimelineItem = useMemo(
    () => (hasTimeline ? message.toolTimeline.find((item) => item.index === selectedTimelineEvent) ?? null : null),
    [hasTimeline, message.toolTimeline, selectedTimelineEvent],
  );

  function handleSelectTimelineItem(item, relatedGroups) {
    const nextIndex = selectedTimelineEvent === item?.index ? null : item?.index ?? null;
    setSelectedTimelineEvent(nextIndex);
    setSelectedTimelineGroups(nextIndex ? uniqueList(relatedGroups) : []);
    setSelectedCardGroup(null);
  }

  function handleSelectCardGroup(group) {
    const nextGroup = selectedCardGroup === group ? null : group;
    setSelectedCardGroup(nextGroup);
    setSelectedTimelineEvent(null);
    setSelectedTimelineGroups([]);
  }

  const hiddenStructuredGroups = [];
  const visibleStructuredCards = hasStructuredCards
    ? message.cards.filter((card) => {
        if (!selectedTimelineEvent) return true;
        if (card.type === 'tool_timeline') return true;
        const focusGroup = cardFocusGroup(card.type);
        if (!focusGroup) return true;
        const keep = selectedTimelineGroups.includes(focusGroup);
        if (!keep) hiddenStructuredGroups.push(focusGroup);
        return keep;
      })
    : [];

  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      <article className={`message-card ${isUser ? 'user' : 'assistant'} ${message.kind === 'tool' ? 'tool' : ''}`}>
        <div className="message-head">
          <span>{isUser ? 'You' : 'mietclaw'}</span>
          {!isUser && message.model ? <span>{message.model}</span> : null}
          {message.kind === 'tool' ? <span>tool</span> : null}
        </div>
        <pre className="message-body">{message.content}</pre>
        {!isUser ? <ProgressList lines={message.progress ?? []} /> : null}
        {!isUser && hasStructuredCards
          ? visibleStructuredCards.map((card) => renderStructuredCard(card, {
              message,
              onConfirmPreview,
              onUseCommand,
              previewBusy,
              selectedTimelineEvent,
              selectedTimelineGroups,
              selectedCardGroup,
              selectedTimelineItem,
              onSelectTimelineItem: handleSelectTimelineItem,
              onSelectCardGroup: handleSelectCardGroup,
            }))
          : null}
        {!isUser && hasStructuredCards && selectedTimelineEvent && hiddenStructuredGroups.length ? (
          <CollapsedCardNotice
            hiddenGroups={uniqueList(hiddenStructuredGroups)}
            onClearFocus={() => handleSelectTimelineItem({ index: selectedTimelineEvent }, [])}
          />
        ) : null}
        {!isUser && !hasStructuredCards ? (
          <TransparencyCard
            message={message}
            onConfirmPreview={onConfirmPreview}
            previewBusy={previewBusy}
            isHighlighted={selectedTimelineGroups.includes('transparency')}
            onFocusTimeline={hasTimeline ? () => handleSelectCardGroup('transparency') : null}
            timelineFocusActive={selectedCardGroup === 'transparency'}
            focusedDetailKey={focusedDetailForGroup('transparency', selectedTimelineItem)}
          />
        ) : null}
        {!isUser && !hasStructuredCards ? (
          <PlanResultCard
            plan={message.plan}
            isHighlighted={selectedTimelineGroups.includes('plan')}
            onFocusTimeline={hasTimeline ? () => handleSelectCardGroup('plan') : null}
            timelineFocusActive={selectedCardGroup === 'plan'}
            focusedDetailKey={focusedDetailForGroup('plan', selectedTimelineItem)}
          />
        ) : null}
        {!isUser && !hasStructuredCards ? (
          <RuntimeSnapshotCard
            message={message}
            isHighlighted={selectedTimelineGroups.includes('runtime')}
            onFocusTimeline={hasTimeline ? () => handleSelectCardGroup('runtime') : null}
            timelineFocusActive={selectedCardGroup === 'runtime'}
            focusedDetailKey={focusedDetailForGroup('runtime', selectedTimelineItem)}
          />
        ) : null}
        {!isUser && !hasStructuredCards && message.runDetail ? (
          <RunResultCard
            detail={message.runDetail}
            onUseCommand={onUseCommand}
            isHighlighted={selectedTimelineGroups.includes('run_result')}
            onFocusTimeline={hasTimeline ? () => handleSelectCardGroup('run_result') : null}
            timelineFocusActive={selectedCardGroup === 'run_result'}
            focusedDetailKey={focusedDetailForGroup('run_result', selectedTimelineItem)}
          />
        ) : null}
      </article>
    </div>
  );
}

const starterMessages = [
  {
    id: 'welcome',
    role: 'assistant',
    kind: 'chat',
    content:
      '我是 mietclaw。你现在可以在顶部切换审批模式：开启时先预览再执行，关闭时允许直接运行；无论哪种模式，我都会尽量把过程、计划和证据展示出来。',
    progress: [],
  },
];

export default function App() {
  const [boot, setBoot] = useState(null);
  const [messages, setMessages] = useState(starterMessages);
  const [input, setInput] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [activePreviewId, setActivePreviewId] = useState(null);
  const [pendingText, setPendingText] = useState('');
  const [approvalRequired, setApprovalRequired] = useState(true);
  const [liveRunDetails, setLiveRunDetails] = useState({});
  const messageEndRef = useRef(null);

  useEffect(() => {
    let active = true;
    fetchJson('/api/chat/bootstrap')
      .then((data) => {
        if (!active) return;
        setBoot(data);
        setSelectedModel(data.localModel?.defaultModel ?? '');
      })
      .catch((bootError) => {
        if (!active) return;
        setError(String(bootError.message || bootError));
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem('mietclaw-approval-required');
      if (stored === 'false') {
        setApprovalRequired(false);
      }
    } catch {
      // ignore storage failures
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem('mietclaw-approval-required', approvalRequired ? 'true' : 'false');
    } catch {
      // ignore storage failures
    }
  }, [approvalRequired]);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, pending, activePreviewId]);

  const hintChips = useMemo(
    () => (boot?.hintChips ?? [
      '你能做什么？',
      '/runs',
      '我希望你把 MoRe 的迁移能垒计算出来并把迁移能垒作为 KMC 软件的输入进行 KMC 模拟',
      '/draft Create a native MD to KMC vacancy diffusion job for "Chat Draft Demo" at 810 K owned by miet.',
      '/run Create a native MD to KMC vacancy diffusion job for "Chat Run Demo" at 812 K owned by miet with 7 images.',
    ]),
    [boot?.hintChips],
  );

  const commandDetails = useMemo(
    () => (boot?.commandDetails ?? (boot?.commandHints ?? []).map((command) => ({ command, summary: '' }))),
    [boot?.commandDetails, boot?.commandHints],
  );

  const runningRuns = useMemo(
    () => (boot?.runs ?? []).filter((run) => run.status === 'Running'),
    [boot?.runs],
  );
  const runningRunKey = runningRuns.map((run) => run.id).join('|');
  const primaryRunningRun = runningRuns[0] ?? null;
  const primaryRunningDetail = primaryRunningRun ? liveRunDetails[primaryRunningRun.id] ?? null : null;

  async function refreshRuns() {
    try {
      const data = await fetchJson('/api/runs');
      setBoot((current) => ({
        ...(current ?? {}),
        runs: data.runs ?? current?.runs ?? [],
      }));
    } catch {
      // ignore refresh-only failures
    }
  }

  useEffect(() => {
    const shouldPoll = pending || Boolean(activePreviewId) || runningRuns.length > 0;
    if (!shouldPoll) return undefined;

    let cancelled = false;
    let timer = null;

    async function poll() {
      try {
        const data = await fetchJson('/api/runs');
        if (cancelled) return;

        const runs = data.runs ?? [];
        setBoot((current) => ({
          ...(current ?? {}),
          runs,
        }));

        const active = runs.filter((run) => run.status === 'Running').slice(0, 3);
        if (!active.length) {
          setLiveRunDetails({});
        } else {
          const detailEntries = await Promise.all(active.map(async (run) => {
            try {
              const payload = await fetchJson(`/api/runs/${encodeURIComponent(run.id)}`);
              return [run.id, payload.run];
            } catch {
              return [run.id, null];
            }
          }));

          if (cancelled) return;
          setLiveRunDetails(Object.fromEntries(detailEntries.filter(([, detail]) => detail)));
        }
      } catch {
        // ignore live polling failures while a run is in flight
      } finally {
        if (!cancelled) {
          timer = window.setTimeout(poll, 1500);
        }
      }
    }

    poll();
    return () => {
      cancelled = true;
      if (timer) {
        window.clearTimeout(timer);
      }
    };
  }, [pending, activePreviewId, runningRunKey]);

  async function handlePreviewCommand({ trimmed, confirmable }) {
    const prompt = trimmed.replace(/^\/\w+\s*/, '').trim();
    if (!prompt) {
      throw new Error(confirmable ? 'Usage: /run <prompt>' : 'Usage: /draft <prompt>');
    }

    const data = await fetchJson('/api/autonomy/draft', {
      method: 'POST',
      body: JSON.stringify({
        prompt,
        provider: 'local',
        confirmable,
      }),
    });

    return buildPreviewMessage({ prompt, data, confirmable });
  }

  async function handleSend(event) {
    event?.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || pending) return;

    const nextUserMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      kind: trimmed.startsWith('/') ? 'tool' : 'chat',
      content: trimmed,
      progress: [],
    };

    const nextMessages = [...messages, nextUserMessage];
    setMessages(nextMessages);
    setPendingText(trimmed);
    setInput('');
    setPending(true);
    setError('');

    try {
      let nextAssistantMessage;
      if (trimmed === '/draft' || trimmed.startsWith('/draft ')) {
        nextAssistantMessage = await handlePreviewCommand({ trimmed, confirmable: false });
      } else if (approvalRequired && (trimmed === '/run' || trimmed.startsWith('/run '))) {
        nextAssistantMessage = await handlePreviewCommand({ trimmed, confirmable: true });
      } else {
        const payload = await fetchJson('/api/chat', {
          method: 'POST',
          body: JSON.stringify({
            input: trimmed,
            model: selectedModel || null,
            previewRuns: approvalRequired,
            messages: nextMessages.map((item) => ({ role: item.role, content: item.content })),
          }),
        });

        nextAssistantMessage = buildChatMessage(payload, selectedModel);
        setBoot((current) => ({
          ...(current ?? {}),
          localModel: payload.localModel ?? current?.localModel,
          runs: payload.runs ?? current?.runs ?? [],
          commandHints: current?.commandHints ?? [],
        }));
      }

      setMessages((current) => [...current, nextAssistantMessage]);
    } catch (requestError) {
      setError(String(requestError.message || requestError));
    } finally {
      setPending(false);
      setPendingText('');
    }
  }

  async function handleConfirmPreview(message) {
    if (!message?.jobSpecPath || activePreviewId) return;
    setActivePreviewId(message.id);
    setError('');

    try {
      const result = await fetchJson('/api/actions/run', {
        method: 'POST',
        body: JSON.stringify({
          jobSpecPath: message.jobSpecPath,
        }),
      });

      setMessages((current) => [
        ...current.map((item) => (
          item.id === message.id
            ? withPreviewState(item, { confirmable: false, executed: true })
            : item
        )),
        buildRunResultMessage({ previewMessage: message, result }),
      ]);
      await refreshRuns();
    } catch (requestError) {
      setError(String(requestError.message || requestError));
    } finally {
      setActivePreviewId(null);
    }
  }

  function handleComposerKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSend(event);
    }
  }

  function pendingLabel() {
    const trimmed = pendingText.trim();
    const activeStep = activeStepFor(primaryRunningRun, primaryRunningDetail);
    if (primaryRunningRun) {
      return [
        `已检测到运行中的任务：${primaryRunningRun.id}`,
        `当前步骤：${activeStep?.id ?? '等待状态刷新'}`,
        `进度：${primaryRunningRun.completedSteps}/${primaryRunningRun.totalSteps}`,
        `最近心跳：${relativeTimeText(activeStep?.heartbeatAt ?? primaryRunningRun.updatedAt ?? null)}`,
      ].join('\n');
    }
    if (trimmed === '/run' || trimmed.startsWith('/run ')) {
      return approvalRequired ? '正在生成执行预览…' : '正在准备并启动运行…';
    }
    if (trimmed === '/draft' || trimmed.startsWith('/draft ')) return '正在生成草稿预览…';
    return '正在处理你的消息…';
  }

  function queueDrawerCommand(command) {
    setInput(command);
    setDrawerOpen(false);
  }

  return (
    <div className="chat-shell">
      <header className="app-bar">
        <div className="brand-lockup">
          <div className="brand-mark">m</div>
          <div>
            <div className="eyebrow">local agent</div>
            <h1>mietclaw</h1>
          </div>
        </div>

        <div className="app-bar-actions">
          <div className={`status-badge ${boot?.localModel?.healthy ? 'ok' : 'bad'}`}>
            {boot?.localModel?.healthy ? 'local model online' : 'local model offline'}
          </div>
          {runningRuns.length ? <div className="status-badge warn">{runningRuns.length} run active</div> : null}
          <button
            type="button"
            className={`toggle-button ${approvalRequired ? 'active' : ''}`}
            onClick={() => setApprovalRequired((value) => !value)}
          >
            {approvalRequired ? 'Approval required: on' : 'Approval required: off'}
          </button>
          <button type="button" className="ghost-button" onClick={() => setDrawerOpen((value) => !value)}>
            {drawerOpen ? 'Close context' : 'Open context'}
          </button>
        </div>
      </header>

      {runningRuns.length ? (
        <section className="live-run-strip">
          <div className="live-run-strip-head">
            <div className="section-label">live runs</div>
            <div className="muted-line">{runningRuns.length} 个任务正在运行</div>
          </div>
          <div className="live-run-grid">
            {runningRuns.slice(0, 2).map((run) => (
              <LiveRunCard
                key={run.id}
                run={run}
                detail={liveRunDetails[run.id] ?? null}
                onUseCommand={setInput}
                compact
              />
            ))}
          </div>
        </section>
      ) : null}

      <main className="chat-stage">
        <section className="chat-thread">
          <div className="thread-inner">
            {messages.map((message) => (
              <Message
                key={message.id}
                message={message}
                onConfirmPreview={handleConfirmPreview}
                onUseCommand={setInput}
                previewBusy={activePreviewId === message.id}
              />
            ))}
            {pending ? (
              <div className="message-row assistant">
                <article className="message-card assistant pending">
                  <div className="message-head">
                    <span>mietclaw</span>
                    <span>{primaryRunningRun ? 'run active' : 'thinking…'}</span>
                  </div>
                  <pre className="message-body">{pendingLabel()}</pre>
                  {primaryRunningRun ? (
                    <LiveRunCard
                      run={primaryRunningRun}
                      detail={primaryRunningDetail}
                      onUseCommand={setInput}
                    />
                  ) : null}
                </article>
              </div>
            ) : null}
            <div ref={messageEndRef} />
          </div>
        </section>
      </main>

      <footer className="composer-shell">
        <div className="composer-inner">
          <div className="hint-row">
            {hintChips.map((hint) => (
              <button key={hint} type="button" className="hint-chip" onClick={() => setInput(hint)}>
                {hint}
              </button>
            ))}
          </div>

          <form className="composer-frame" onSubmit={handleSend}>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder={
                approvalRequired
                  ? '直接说你想做什么；当前开启审批模式，/run 和 run 类自然语言都会先预览再执行。'
                  : '直接说你想做什么；当前允许直接运行，/run 和 run 类自然语言可以直接开跑。'
              }
              rows={4}
            />
            <div className="composer-footer">
              <div className="composer-note">
                {approvalRequired
                  ? 'Enter 发送 · Shift + Enter 换行 · 当前审批模式已开启：/run 和 run 类自然语言会先生成执行预览'
                  : 'Enter 发送 · Shift + Enter 换行 · 当前审批模式已关闭：允许直接运行，但仍会保留过程与证据展示'}
              </div>
              <button type="submit" className="send-button" disabled={pending || !input.trim()}>
                Send
              </button>
            </div>
          </form>

          {error ? <div className="error-banner">{error}</div> : null}
        </div>
      </footer>

      <div className={`drawer-scrim ${drawerOpen ? 'visible' : ''}`} onClick={() => setDrawerOpen(false)} aria-hidden="true" />
      <aside className={`context-drawer ${drawerOpen ? 'open' : ''}`}>
        <div className="drawer-section">
          <div className="section-label">model</div>
          <select className="model-select" value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
            {(boot?.localModel?.models ?? []).map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
          <div className="muted-line">endpoint: {boot?.localModel?.baseUrl ?? '—'}</div>
        </div>

        <div className="drawer-section">
          <div className="section-label">commands</div>
          <div className="command-list">
            {commandDetails.map((item) => (
              <button key={item.command} type="button" className="command-chip command-detail-chip" onClick={() => queueDrawerCommand(item.command)}>
                <span className="command-chip-command">{item.command}</span>
                {item.summary ? <span className="command-chip-summary">{item.summary}</span> : null}
              </button>
            ))}
          </div>
        </div>

        <div className="drawer-section">
          <div className="section-label">recent runs</div>
          <div className="run-list">
            {(boot?.runs ?? []).slice(0, 10).map((run) => (
              <RecentRunDrawerItem key={run.id} run={run} onUseCommand={queueDrawerCommand} />
            ))}
          </div>
        </div>
      </aside>
    </div>
  );
}
