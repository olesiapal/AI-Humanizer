#!/usr/bin/env node

import { execFileSync, spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const NEXT_BIN = resolve(ROOT, 'node_modules/next/dist/bin/next');
const MEMORY_LIMIT_MB = readNumber('DEV_MEMORY_LIMIT_MB', 3072);
const MEMORY_WARN_MB = readNumber('DEV_MEMORY_WARN_MB', Math.floor(MEMORY_LIMIT_MB * 0.8));
const POLL_MS = readNumber('DEV_MEMORY_POLL_MS', 2000);
const NODE_HEAP_LIMIT_MB = readNumber(
  'NODE_HEAP_LIMIT_MB',
  Math.min(2048, Math.max(1024, MEMORY_LIMIT_MB - 1024))
);

if (!existsSync(NEXT_BIN)) {
  console.error('[dev-safe] Missing Next.js binary. Run npm install first.');
  process.exit(1);
}

if (process.arch !== 'arm64' && process.platform === 'darwin') {
  console.warn(
    `[dev-safe] Warning: Node is ${process.arch}. On Apple Silicon, install/use native arm64 Node to reduce memory risk.`
  );
}

const child = spawn(process.execPath, [NEXT_BIN, 'dev', '--webpack'], {
  cwd: ROOT,
  env: {
    ...process.env,
    ...readProjectEnv(),
    NEXT_TELEMETRY_DISABLED: '1',
    NODE_OPTIONS: withHeapLimit(process.env.NODE_OPTIONS, NODE_HEAP_LIMIT_MB),
  },
  stdio: 'inherit',
});

console.log(
  `[dev-safe] Started Next with webpack. Memory limit: ${MEMORY_LIMIT_MB} MB, Node heap: ${NODE_HEAP_LIMIT_MB} MB.`
);

let warned = false;
let monitorUnavailable = false;
let shuttingDown = false;

const timer = setInterval(() => {
  if (shuttingDown || child.exitCode !== null || child.signalCode !== null) return;

  const snapshot = readProcessSnapshot();
  if (!snapshot) return;

  const tree = getProcessTree(snapshot, child.pid);
  const rssMb = Math.round(tree.reduce((sum, row) => sum + row.rssKb, 0) / 1024);

  if (!warned && rssMb >= MEMORY_WARN_MB) {
    warned = true;
    console.warn(`[dev-safe] Warning: dev process tree is using about ${rssMb} MB RSS.`);
  }

  if (rssMb >= MEMORY_LIMIT_MB) {
    console.error(
      `[dev-safe] Memory limit exceeded: ${rssMb} MB >= ${MEMORY_LIMIT_MB} MB. Stopping dev server.`
    );
    shutdownTree(tree, 'SIGTERM');
    setTimeout(() => shutdownTree(tree, 'SIGKILL'), 5000).unref();
  }
}, POLL_MS);

timer.unref();

for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    if (shuttingDown) return;
    shuttingDown = true;
    clearInterval(timer);
    try {
      process.kill(child.pid, signal);
    } catch {
      // Process already exited.
    }
  });
}

child.on('exit', (code, signal) => {
  clearInterval(timer);
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});

function readNumber(name, fallback) {
  const raw = process.env[name];
  if (!raw) return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function withHeapLimit(existing, limitMb) {
  const parts = (existing ?? '')
    .split(/\s+/)
    .map((part) => part.trim())
    .filter(Boolean)
    .filter((part) => !part.startsWith('--max-old-space-size='));

  parts.push(`--max-old-space-size=${limitMb}`);
  return parts.join(' ');
}

function readProjectEnv() {
  const env = {};

  for (const file of ['.env', '.env.local']) {
    const path = resolve(ROOT, file);
    if (!existsSync(path)) continue;

    for (const line of readFileSync(path, 'utf8').split(/\r?\n/)) {
      const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
      if (!match) continue;

      const [, key, rawValue] = match;
      if (line.trimStart().startsWith('#')) continue;
      env[key] = parseEnvValue(rawValue);
    }
  }

  return env;
}

function parseEnvValue(value) {
  const trimmed = value.trim();
  const quote = trimmed[0];

  if ((quote === '"' || quote === "'") && trimmed.endsWith(quote)) {
    return trimmed.slice(1, -1);
  }

  const commentIndex = trimmed.indexOf(' #');
  return commentIndex >= 0 ? trimmed.slice(0, commentIndex).trim() : trimmed;
}

function readProcessSnapshot() {
  try {
    const output = execFileSync('/bin/ps', ['-axo', 'pid=,ppid=,rss=,command='], {
      encoding: 'utf8',
      maxBuffer: 16 * 1024 * 1024,
    });

    return output
      .split('\n')
      .map((line) => line.trim().match(/^(\d+)\s+(\d+)\s+(\d+)\s+(.+)$/))
      .filter(Boolean)
      .map((match) => ({
        pid: Number(match[1]),
        ppid: Number(match[2]),
        rssKb: Number(match[3]),
        command: match[4],
      }));
  } catch (error) {
    if (!monitorUnavailable) {
      monitorUnavailable = true;
      console.warn(`[dev-safe] Memory watchdog unavailable: ${error.message}`);
    }
    return null;
  }
}

function getProcessTree(rows, rootPid) {
  const byParent = new Map();
  for (const row of rows) {
    if (!byParent.has(row.ppid)) byParent.set(row.ppid, []);
    byParent.get(row.ppid).push(row);
  }

  const result = [];
  const stack = rows.filter((row) => row.pid === rootPid);
  const seen = new Set();

  while (stack.length > 0) {
    const row = stack.pop();
    if (!row || seen.has(row.pid)) continue;
    seen.add(row.pid);
    result.push(row);
    stack.push(...(byParent.get(row.pid) ?? []));
  }

  return result;
}

function shutdownTree(tree, signal) {
  for (const row of [...tree].sort((a, b) => b.pid - a.pid)) {
    try {
      process.kill(row.pid, signal);
    } catch {
      // Process already exited.
    }
  }
}
