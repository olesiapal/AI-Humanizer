'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { AIModel } from '@/lib/ai-providers';

interface LogEntry {
  id: string;
  type: 'log' | 'score' | 'highlights' | 'draft' | 'complete';
  level?: 'info' | 'success' | 'warn' | 'error' | 'check';
  message?: string;
  score?: number;
  iteration?: number;
  sentences?: Array<{ text: string; prob: number; impact?: 'high' | 'medium' | 'low'; reason?: string }>;
  text?: string;
  timestamp: number;
}

const MODELS: { value: AIModel; label: string; provider: string; color: string }[] = [
  { value: 'gemini-3-flash-preview', label: 'Gemini 3 Flash Preview', provider: 'Google', color: '#4285f4' },
  { value: 'gemini-3-pro-preview', label: 'Gemini 3 Pro Preview', provider: 'Google', color: '#34a853' },
  { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash', provider: 'Google', color: '#4285f4' },
  { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro', provider: 'Google', color: '#34a853' },
  { value: 'gpt-5.5', label: 'GPT-5.5', provider: 'OpenAI', color: '#10a37f' },
  { value: 'gpt-4o', label: 'GPT-4o', provider: 'OpenAI', color: '#10a37f' },
  { value: 'gpt-4o-mini', label: 'GPT-4o Mini', provider: 'OpenAI', color: '#10a37f' },
  { value: 'claude-opus-4-7', label: 'Claude Opus 4.7', provider: 'Anthropic', color: '#d4a853' },
  { value: 'claude-sonnet-4-6', label: 'Claude Sonnet 4.6', provider: 'Anthropic', color: '#c96442' },
  { value: 'claude-haiku-4-5', label: 'Claude Haiku 4.5', provider: 'Anthropic', color: '#b865c9' },
];

const LEVEL_STYLES: Record<string, string> = {
  info: 'text-blue-400',
  success: 'text-emerald-400',
  warn: 'text-amber-400',
  error: 'text-red-400',
  check: 'text-purple-400',
};

const LEVEL_ICONS: Record<string, string> = {
  info: '→',
  success: '✓',
  warn: '!',
  error: '✗',
  check: '◈',
};

function ScoreBar({ score, iteration }: { score: number; iteration: number }) {
  const pct = Math.round(score * 100);
  const isGood = pct < 50;
  const color = isGood ? '#22d3a8' : pct < 75 ? '#f59e0b' : '#f87171';

  return (
    <div className="my-2 fade-in">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs text-gray-500 font-mono">Итерация {iteration}</span>
        <span className="font-mono text-sm font-semibold" style={{ color }}>
          {pct}% AI
        </span>
      </div>
      <div className="h-2 rounded-full bg-gray-800 overflow-hidden">
        <div
          className="h-full rounded-full score-bar transition-all"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

function HighlightedSentences({
  sentences,
}: {
  sentences: Array<{ text: string; prob: number; impact?: 'high' | 'medium' | 'low'; reason?: string }>;
}) {
  return (
    <div className="mt-2 mb-2 fade-in">
      <p className="text-xs text-gray-500 mb-2 uppercase tracking-wider font-mono">
        AI-предложения ({sentences.length})
      </p>
      <div className="space-y-1">
        {sentences.map((s, i) => {
          const impact = s.impact ?? inferImpactFromReason(s.reason);
          const tone =
            impact === 'high'
              ? {
                  border: 'border-red-500',
                  bg: 'bg-red-500/10',
                  text: 'text-red-300',
                  label: 'text-red-500',
                  muted: 'text-red-200/70',
                }
              : impact === 'medium'
                ? {
                    border: 'border-orange-500',
                    bg: 'bg-orange-500/10',
                    text: 'text-orange-300',
                    label: 'text-orange-500',
                    muted: 'text-orange-200/70',
                  }
                : impact === 'low'
                  ? {
                      border: 'border-yellow-500',
                      bg: 'bg-yellow-500/10',
                      text: 'text-yellow-300',
                      label: 'text-yellow-500',
                      muted: 'text-yellow-200/70',
                    }
                  : {
                      border: 'border-red-500',
                      bg: 'bg-red-500/10',
                      text: 'text-red-300',
                      label: 'text-red-500',
                      muted: 'text-red-200/70',
                    };
          const label = impact ? impact.toUpperCase() : `${Math.round(s.prob * 100)}%`;

          return (
            <div
              key={i}
              className={`text-xs rounded px-3 py-2 border-l-2 ${tone.border} ${tone.bg} ${tone.text} font-mono leading-relaxed`}
            >
              <span className={`${tone.label} mr-2`}>{label}</span>
              {s.text}
              {s.reason && (
                <div className={`mt-1 text-[11px] leading-relaxed ${tone.muted}`}>
                  {s.reason}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function inferImpactFromReason(reason?: string): 'high' | 'medium' | 'low' | undefined {
  if (!reason) return undefined;
  const normalized = reason.toLowerCase();
  if (normalized.includes('manual_high_impact') || normalized.includes('impact: high')) return 'high';
  if (
    normalized.includes('manual_medium_impact') ||
    normalized.includes('impact: medium') ||
    normalized.includes('impact: moderate')
  ) {
    return 'medium';
  }
  if (normalized.includes('manual_low_impact') || normalized.includes('impact: low')) return 'low';
  return undefined;
}

function countManualImpactTargets(value: string): number {
  return value
    .split(/\n+/)
    .map((line) => line.trim())
    .filter((line) => line && !/^(high|medium|moderate|low)\s+ai\s+impact$/i.test(line))
    .length;
}

export default function HumanizerApp() {
  const [inputText, setInputText] = useState('');
  const [inputRevision, setInputRevision] = useState(0);
  const [resultRevision, setResultRevision] = useState(-1);
  const [manualHighImpact, setManualHighImpact] = useState('');
  const [isManualExpanded, setIsManualExpanded] = useState(false);
  const [isStyleExpanded, setIsStyleExpanded] = useState(false);
  const [useInitialRewrite, setUseInitialRewrite] = useState(true);
  const [useGptZero, setUseGptZero] = useState(true);
  const [useDeepRewrite, setUseDeepRewrite] = useState(true);
  const [useWritingTeacher, setUseWritingTeacher] = useState(true);
  const [styleProfile, setStyleProfile] = useState('');
  const [impactAttempts, setImpactAttempts] = useState(2);
  const [fixAllImpactAtOnce, setFixAllImpactAtOnce] = useState(false);
  const [selectedModel, setSelectedModel] = useState<AIModel>('gemini-3-flash-preview');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [finalText, setFinalText] = useState('');
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'error'>('idle');
  const [isRunning, setIsRunning] = useState(false);
  const [wordCount, setWordCount] = useState(0);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const copyResetRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const words = inputText.trim().split(/\s+/).filter(Boolean).length;
    setWordCount(words);
  }, [inputText]);

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('humanizer-settings') ?? '{}');
      if (MODELS.some((model) => model.value === saved.selectedModel)) {
        setSelectedModel(saved.selectedModel);
      }
      if (typeof saved.useInitialRewrite === 'boolean') {
        setUseInitialRewrite(saved.useInitialRewrite);
      }
      if (typeof saved.useGptZero === 'boolean') {
        setUseGptZero(saved.useGptZero);
      }
      if (typeof saved.useDeepRewrite === 'boolean') {
        setUseDeepRewrite(saved.useDeepRewrite);
      }
      if (typeof saved.useWritingTeacher === 'boolean') {
        setUseWritingTeacher(saved.useWritingTeacher);
      }
      if (typeof saved.styleProfile === 'string') {
        setStyleProfile(saved.styleProfile);
      }
      if (typeof saved.fixAllImpactAtOnce === 'boolean') {
        setFixAllImpactAtOnce(saved.fixAllImpactAtOnce);
      }
      if (typeof saved.impactAttempts === 'number') {
        setImpactAttempts(Math.min(8, Math.max(1, Math.floor(saved.impactAttempts))));
      }
    } catch {
      // ignore bad local settings
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(
      'humanizer-settings',
      JSON.stringify({
        selectedModel,
        useInitialRewrite,
        useGptZero,
        useDeepRewrite,
        useWritingTeacher,
        styleProfile,
        fixAllImpactAtOnce,
        impactAttempts,
      })
    );
  }, [
    selectedModel,
    useInitialRewrite,
    useGptZero,
    useDeepRewrite,
    useWritingTeacher,
    styleProfile,
    fixAllImpactAtOnce,
    impactAttempts,
  ]);

  const manualHighImpactCount = countManualImpactTargets(manualHighImpact);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  useEffect(() => {
    setCopyStatus('idle');
  }, [finalText]);

  const addLog = useCallback((entry: Omit<LogEntry, 'id' | 'timestamp'>) => {
    setLogs((prev) => [
      ...prev,
      { ...entry, id: Math.random().toString(36).slice(2), timestamp: Date.now() },
    ]);
  }, []);

  const handleStart = useCallback(async () => {
    const canUseCurrentResult =
      manualHighImpact.trim() && finalText.trim() && resultRevision === inputRevision;
    const textForRequest = canUseCurrentResult ? finalText : inputText;
    if (!textForRequest.trim()) return;

    setIsRunning(true);
    setLogs([]);
    setFinalText('');
    setCopyStatus('idle');

    const abort = new AbortController();
    abortRef.current = abort;

    try {
      const response = await fetch('/api/humanize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: textForRequest,
          model: selectedModel,
          manualHighImpact,
          useInitialRewrite,
          useGptZero,
          useDeepRewrite,
          useWritingTeacher,
          styleProfile,
          impactAttempts,
          fixAllImpactAtOnce,
        }),
        signal: abort.signal,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));

            if (data.type === 'complete') {
              setFinalText(data.text);
              setResultRevision(inputRevision);
              addLog({ type: 'complete', message: 'Готово!', level: 'success' });
            } else if (data.type === 'draft') {
              setFinalText(data.text);
              setResultRevision(inputRevision);
              addLog({
                type: 'draft',
                text: data.text,
                message: data.message ?? 'Черновик обновлён',
                level: 'info',
              });
            } else {
              addLog(data);
            }
          } catch {
            // skip malformed lines
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') {
        addLog({ type: 'log', level: 'error', message: `Ошибка: ${String(e)}` });
      }
    } finally {
      setIsRunning(false);
    }
  }, [
    inputText,
    inputRevision,
    resultRevision,
    finalText,
    selectedModel,
    manualHighImpact,
    useInitialRewrite,
    useGptZero,
    useDeepRewrite,
    useWritingTeacher,
    styleProfile,
    impactAttempts,
    fixAllImpactAtOnce,
    addLog,
  ]);

  const handleStop = () => {
    abortRef.current?.abort();
    setIsRunning(false);
  };

  const handleCopyResult = async () => {
    if (!finalText) return;

    try {
      await navigator.clipboard.writeText(finalText);
      setCopyStatus('copied');
    } catch {
      setCopyStatus('error');
    }

    if (copyResetRef.current) {
      clearTimeout(copyResetRef.current);
    }
    copyResetRef.current = setTimeout(() => {
      setCopyStatus('idle');
      copyResetRef.current = null;
    }, 1600);
  };

  const handleClear = () => {
    setLogs([]);
    setFinalText('');
    setCopyStatus('idle');
    setResultRevision(-1);
    setInputText('');
    setInputRevision((revision) => revision + 1);
    setManualHighImpact('');
    setIsManualExpanded(false);
    setIsStyleExpanded(false);
  };

  const selectedModelInfo = MODELS.find((m) => m.value === selectedModel)!;

  return (
    <div className="flex flex-col h-screen" style={{ background: 'var(--bg-primary)' }}>
      {/* Header */}
      <header
        className="flex items-center justify-between px-6 py-3 border-b shrink-0"
        style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold"
            style={{ background: 'var(--accent)' }}
          >
            H
          </div>
          <div>
            <h1 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
              AI Humanaizer
            </h1>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              GPTZero bypass tool
            </p>
          </div>
        </div>

        {/* Model selector */}
        <div className="flex items-center gap-3">
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Модель:
          </span>
          <div className="relative">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value as AIModel)}
              disabled={isRunning}
              className="text-sm rounded-lg px-3 py-1.5 pr-8 appearance-none cursor-pointer border font-medium"
              style={{
                background: 'var(--bg-card)',
                borderColor: 'var(--border)',
                color: 'var(--text-primary)',
              }}
            >
              {MODELS.map((m) => (
                <option key={m.value} value={m.value}>
                  {m.provider} · {m.label}
                </option>
              ))}
            </select>
            <span
              className="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-xs"
              style={{ color: 'var(--text-muted)' }}
            >
              ▾
            </span>
          </div>
          <div
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: selectedModelInfo.color }}
          />
          <label
            className="flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs cursor-pointer"
            style={{
              borderColor: 'var(--border)',
              color: 'var(--text-muted)',
              background: 'var(--bg-card)',
            }}
          >
            <input
              type="checkbox"
              checked={useInitialRewrite}
              onChange={(event) => setUseInitialRewrite(event.target.checked)}
              disabled={isRunning}
              className="accent-purple-500"
            />
            Первичный rewrite
          </label>
          <label
            className="flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs cursor-pointer"
            style={{
              borderColor: useDeepRewrite ? '#34d399' : 'var(--border)',
              color: useDeepRewrite ? '#6ee7b7' : 'var(--text-muted)',
              background: useDeepRewrite ? 'rgba(52, 211, 153, 0.08)' : 'var(--bg-card)',
            }}
          >
            <input
              type="checkbox"
              checked={useDeepRewrite}
              onChange={(event) => setUseDeepRewrite(event.target.checked)}
              disabled={isRunning}
              className="accent-purple-500"
            />
            Deep rewrite
          </label>
          <label
            className="flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs cursor-pointer"
            style={{
              borderColor: useWritingTeacher ? '#60a5fa' : 'var(--border)',
              color: useWritingTeacher ? '#93c5fd' : 'var(--text-muted)',
              background: useWritingTeacher ? 'rgba(96, 165, 250, 0.08)' : 'var(--bg-card)',
            }}
          >
            <input
              type="checkbox"
              checked={useWritingTeacher}
              onChange={(event) => setUseWritingTeacher(event.target.checked)}
              disabled={isRunning}
              className="accent-purple-500"
            />
            Учитель письма
          </label>
          <label
            className="flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs cursor-pointer"
            style={{
              borderColor: useGptZero ? 'var(--border)' : '#f59e0b',
              color: useGptZero ? 'var(--text-muted)' : '#fbbf24',
              background: useGptZero ? 'var(--bg-card)' : 'rgba(245, 158, 11, 0.08)',
            }}
          >
            <input
              type="checkbox"
              checked={useGptZero}
              onChange={(event) => setUseGptZero(event.target.checked)}
              disabled={isRunning}
              className="accent-purple-500"
            />
            GPTZero API
          </label>
          <label
            className="flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs cursor-pointer"
            style={{
              borderColor: 'var(--border)',
              color: 'var(--text-muted)',
              background: 'var(--bg-card)',
            }}
          >
            <input
              type="checkbox"
              checked={fixAllImpactAtOnce}
              onChange={(event) => setFixAllImpactAtOnce(event.target.checked)}
              disabled={isRunning}
              className="accent-purple-500"
            />
            Все impact сразу
          </label>
          <label
            className="flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs"
            style={{
              borderColor: 'var(--border)',
              color: 'var(--text-muted)',
              background: 'var(--bg-card)',
            }}
          >
            Impact попыток
            <input
              type="number"
              min={1}
              max={8}
              value={impactAttempts}
              onChange={(event) => {
                const value = Number.parseInt(event.target.value, 10);
                setImpactAttempts(Number.isFinite(value) ? Math.min(8, Math.max(1, value)) : 1);
              }}
              disabled={isRunning}
              className="w-12 rounded border px-2 py-0.5 text-xs font-mono"
              style={{
                borderColor: 'var(--border)',
                background: 'var(--bg-input)',
                color: 'var(--text-primary)',
              }}
            />
          </label>
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left panel — Input */}
        <div
          className="flex flex-col w-[45%] border-r shrink-0"
          style={{ borderColor: 'var(--border)' }}
        >
          <div
            className="flex items-center justify-between px-4 py-2 border-b shrink-0"
            style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}
          >
            <span className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
              Исходный текст
            </span>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {wordCount} слов · {inputText.length} символов
            </span>
          </div>

          <textarea
            value={inputText}
            onChange={(e) => {
              setInputText(e.target.value);
              setInputRevision((revision) => revision + 1);
            }}
            placeholder="Вставьте текст, который нужно гуманизировать..."
            disabled={isRunning}
            className="flex-1 p-4 text-sm leading-relaxed font-mono"
            style={{
              background: 'var(--bg-input)',
              color: 'var(--text-primary)',
              caretColor: 'var(--accent)',
            }}
          />

          <div
            className="border-t shrink-0"
            style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}
          >
            <div className="flex items-center justify-between gap-3 px-4 py-2">
              <div>
                <span className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
                  Мой стиль
                </span>
                <p className="mt-0.5 text-[11px]" style={{ color: 'var(--text-muted)' }}>
                  Необязательно: пару правил или примеров тона
                </p>
              </div>
              <div className="flex items-center gap-2">
                {styleProfile.trim() && (
                  <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                    {styleProfile.length} симв.
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => setIsStyleExpanded((value) => !value)}
                  disabled={isRunning}
                  className="rounded border px-2 py-1 text-[11px] transition-all"
                  style={{
                    borderColor: 'var(--border)',
                    color: 'var(--text-muted)',
                    background: 'transparent',
                  }}
                >
                  {isStyleExpanded ? 'Сжать' : 'Открыть'}
                </button>
                {styleProfile && (
                  <button
                    type="button"
                    onClick={() => setStyleProfile('')}
                    disabled={isRunning}
                    className="rounded border px-2 py-1 text-[11px] transition-all"
                    style={{
                      borderColor: 'var(--border)',
                      color: 'var(--text-muted)',
                      background: 'transparent',
                    }}
                  >
                    Очистить
                  </button>
                )}
              </div>
            </div>
            {isStyleExpanded && (
              <textarea
                value={styleProfile}
                onChange={(e) => setStyleProfile(e.target.value)}
                placeholder="Например: пиши как короткий GitHub-коммент от коллеги; без corporate tone; можно чуть неровно, но технические токены не трогать..."
                disabled={isRunning}
                className="h-24 w-full resize-y px-4 pb-3 text-xs leading-relaxed font-mono"
                style={{
                  background: 'var(--bg-input)',
                  color: 'var(--text-primary)',
                  caretColor: 'var(--accent)',
                }}
              />
            )}
          </div>

          <div
            className="border-t shrink-0"
            style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}
          >
            <div className="flex items-center justify-between gap-3 px-4 py-2">
              <div>
                <span className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
                  Impact симптомы из GPTZero
                </span>
                <p className="mt-0.5 text-[11px]" style={{ color: 'var(--text-muted)' }}>
                  Модель будет менять контекст вокруг них, не только эти строки
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                  {manualHighImpactCount} целей
                </span>
                <button
                  type="button"
                  onClick={() => setIsManualExpanded((value) => !value)}
                  disabled={isRunning}
                  className="rounded border px-2 py-1 text-[11px] transition-all"
                  style={{
                    borderColor: 'var(--border)',
                    color: 'var(--text-muted)',
                    background: 'transparent',
                  }}
                >
                  {isManualExpanded ? 'Сжать' : 'Шире'}
                </button>
                {manualHighImpact && (
                  <button
                    type="button"
                    onClick={() => setManualHighImpact('')}
                    disabled={isRunning}
                    className="rounded border px-2 py-1 text-[11px] transition-all"
                    style={{
                      borderColor: 'var(--border)',
                      color: 'var(--text-muted)',
                      background: 'transparent',
                    }}
                  >
                    Очистить
                  </button>
                )}
              </div>
            </div>
            <textarea
              value={manualHighImpact}
              onChange={(e) => setManualHighImpact(e.target.value)}
              placeholder="Вставьте impact-строки из GPTZero UI. Это симптомы, не точечные замены: модель может менять соседние предложения и структуру текста..."
              disabled={isRunning}
              className={`w-full resize-y px-4 pb-3 text-xs leading-relaxed font-mono ${
                isManualExpanded ? 'h-56' : 'h-32'
              }`}
              style={{
                background: 'var(--bg-input)',
                color: 'var(--text-primary)',
                caretColor: 'var(--accent)',
              }}
            />
          </div>

          <div
            className="flex items-center gap-3 p-3 border-t shrink-0"
            style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}
          >
            <button
              onClick={isRunning ? handleStop : handleStart}
              disabled={!inputText.trim() && !isRunning}
              className="flex-1 py-2.5 rounded-lg text-sm font-semibold transition-all flex items-center justify-center gap-2"
              style={{
                background: isRunning ? '#7f1d1d' : 'var(--accent)',
                color: 'white',
                opacity: !inputText.trim() && !isRunning ? 0.4 : 1,
              }}
            >
              {isRunning ? (
                <>
                  <span className="w-3 h-3 rounded-sm bg-white/80 animate-pulse" />
                  Остановить
                </>
              ) : (
                <>
                  <span>▶</span>
                  Гуманизировать
                </>
              )}
            </button>

            <button
              onClick={handleClear}
              disabled={isRunning}
              className="px-4 py-2.5 rounded-lg text-xs transition-all border"
              style={{
                borderColor: 'var(--border)',
                color: 'var(--text-muted)',
                background: 'transparent',
              }}
            >
              Очистить
            </button>
          </div>
        </div>

        {/* Right panel — Logs + Result */}
        <div className="flex flex-col flex-1 overflow-hidden">
          {/* Logs */}
          <div
            className="flex flex-col border-b overflow-hidden"
            style={{ borderColor: 'var(--border)', flex: finalText ? '0 0 55%' : '1 1 auto' }}
          >
            <div
              className="flex items-center justify-between px-4 py-2 border-b shrink-0"
              style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}
            >
              <span className="text-xs font-medium uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
                Процесс
              </span>
              {isRunning && (
                <span className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--accent)' }}>
                  <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
                  работаю...
                </span>
              )}
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-1" style={{ background: 'var(--bg-input)' }}>
              {logs.length === 0 && !isRunning && (
                <div className="flex items-center justify-center h-full">
                  <p className="text-xs text-center" style={{ color: 'var(--text-muted)' }}>
                    Вставьте текст слева и нажмите «Гуманизировать».<br />
                    Здесь будут появляться логи процесса.
                  </p>
                </div>
              )}

              {logs.map((entry) => (
                <div key={entry.id} className="fade-in">
                  {entry.type === 'score' && entry.score !== undefined && entry.iteration ? (
                    <ScoreBar score={entry.score} iteration={entry.iteration} />
                  ) : entry.type === 'highlights' && entry.sentences ? (
                    <HighlightedSentences sentences={entry.sentences} />
                  ) : entry.type === 'draft' && entry.text ? (
                    <div className="my-2 rounded border border-blue-500/30 bg-blue-500/10 p-3">
                      <div className="mb-2 flex items-center gap-2 text-xs font-mono text-blue-300">
                        <span>→</span>
                        <span>{entry.message ?? 'Черновик'}</span>
                      </div>
                      <pre className="whitespace-pre-wrap text-xs leading-relaxed text-blue-100 font-mono">
                        {entry.text}
                      </pre>
                    </div>
                  ) : entry.message ? (
                    <div className="flex items-start gap-2 py-0.5">
                      <span
                        className="shrink-0 font-mono text-xs mt-0.5"
                        style={{ color: entry.level ? undefined : 'var(--text-muted)' }}
                      >
                        <span className={entry.level ? LEVEL_STYLES[entry.level] : 'text-gray-500'}>
                          {entry.level ? LEVEL_ICONS[entry.level] : '·'}
                        </span>
                      </span>
                      <span
                        className={`text-xs font-mono leading-relaxed ${entry.level ? LEVEL_STYLES[entry.level] : ''}`}
                        style={!entry.level ? { color: 'var(--text-muted)' } : undefined}
                      >
                        {entry.message}
                      </span>
                    </div>
                  ) : null}
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>

          {/* Result */}
          {finalText && (
            <div className="flex flex-col" style={{ flex: '1 1 45%', minHeight: 0 }}>
              <div
                className="flex items-center justify-between px-4 py-2 border-b shrink-0"
                style={{ borderColor: 'var(--border)', background: 'var(--bg-secondary)' }}
              >
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <span className="text-xs font-medium uppercase tracking-wider text-emerald-400">
                    {isRunning ? 'Текущий черновик' : 'Готовый текст'}
                  </span>
                </div>
                <button
                  onClick={handleCopyResult}
                  className="text-xs px-3 py-1 rounded border transition-all hover:border-purple-500"
                  style={{
                    borderColor:
                      copyStatus === 'copied'
                        ? '#34d399'
                        : copyStatus === 'error'
                          ? '#f87171'
                          : 'var(--border)',
                    color:
                      copyStatus === 'copied'
                        ? '#34d399'
                        : copyStatus === 'error'
                          ? '#f87171'
                          : 'var(--text-muted)',
                    background:
                      copyStatus === 'copied'
                        ? 'rgba(52, 211, 153, 0.08)'
                        : copyStatus === 'error'
                          ? 'rgba(248, 113, 113, 0.08)'
                          : 'transparent',
                  }}
                >
                  {copyStatus === 'copied'
                    ? 'Скопировано'
                    : copyStatus === 'error'
                      ? 'Ошибка'
                      : 'Скопировать'}
                </button>
              </div>

              <textarea
                readOnly
                value={finalText}
                className="flex-1 p-4 text-sm leading-relaxed font-mono"
                style={{
                  background: 'var(--bg-input)',
                  color: 'var(--text-primary)',
                }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
