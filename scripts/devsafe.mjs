#!/usr/bin/env node

import { execFileSync, spawn } from 'node:child_process';

const args = process.argv.slice(2);
const commandIndex = args.indexOf('--');
const commandArgs = commandIndex >= 0 ? args.slice(commandIndex + 1) : args;

if (commandArgs.length === 0 || args.includes('-h') || args.includes('--help')) {
  console.log('Usage: devsafe [--] <command> [...args]');
  console.log('Example: devsafe npm run dev');
  console.log('Env: DEV_MEMORY_LIMIT_MB=3072 DEV_MEMORY_WARN_MB=2457 NODE_HEAP_LIMIT_MB=2048');
  process.exit(commandArgs.length === 0 ? 1 : 0);
}

const MEMORY_LIMIT_MB = readNumber('DEV_MEMORY_LIMIT_MB', 3072);
const MEMORY_WARN_MB = readNumber('DEV_MEMORY_WARN_MB', Math.floor(MEMORY_LIMIT_MB * 0.8));
const POLL_MS = readNumber('DEV_MEMORY_POLL_MS', 2000);
const NODE_HEAP_LIMIT_MB = readNumber(
  'NODE_HEAP_LIMIT_MB',
  Math.min(2048, Math.max(1024, MEMORY_LIMIT_MB - 1024))
);

const child = spawn(commandArgs[0], commandArgs.slice(1), {
  env: {
    ...process.env,
    NODE_OPTIONS: withHeapLimit(process.env.NODE_OPTIONS, NODE_HEAP_LIMIT_MB),
  },
  stdio: 'inherit',
});

console.log(
  `[devsafe] Started: ${commandArgs.join(' ')}. Memory limit: ${MEMORY_LIMIT_MB} MB, Node heap: ${NODE_HEAP_LIMIT_MB} MB.`
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
    console.warn(`[devsafe] Warning: process tree is using about ${rssMb} MB RSS.`);
  }

  if (rssMb >= MEMORY_LIMIT_MB) {
    console.error(`[devsafe] Memory limit exceeded: ${rssMb} MB >= ${MEMORY_LIMIT_MB} MB.`);
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
      }));
  } catch (error) {
    if (!monitorUnavailable) {
      monitorUnavailable = true;
      console.warn(`[devsafe] Memory watchdog unavailable: ${error.message}`);
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
