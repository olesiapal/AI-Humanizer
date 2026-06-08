#!/usr/bin/env node

import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const target = process.argv[2] ?? 'production';
const envPath = resolve(process.cwd(), '.env.local');
const keys = ['GPTZERO_API_KEY', 'GEMINI_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'];

if (!existsSync(envPath)) {
  console.error('[sync-vercel-env] .env.local not found');
  process.exit(1);
}

const values = readEnvFile(envPath);

for (const key of keys) {
  const value = values[key];
  if (!value) {
    console.warn(`[sync-vercel-env] Skipping ${key}: missing in .env.local`);
    continue;
  }

  await addEnv(key, value, target);
}

function readEnvFile(path) {
  const env = {};

  for (const line of readFileSync(path, 'utf8').split(/\r?\n/)) {
    if (!line.trim() || line.trimStart().startsWith('#')) continue;

    const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
    if (!match) continue;

    env[match[1]] = parseEnvValue(match[2]);
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

function addEnv(key, value, environment) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn('vercel', ['env', 'add', key, environment, '--sensitive', '--force'], {
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let output = '';
    let error = '';

    child.stdout.on('data', (chunk) => {
      output += chunk;
    });
    child.stderr.on('data', (chunk) => {
      error += chunk;
    });

    child.on('error', reject);
    child.on('close', (code) => {
      const cleaned = `${output}${error}`.replace(value, '<redacted>').trim();

      if (code === 0) {
        console.log(`[sync-vercel-env] ${key}: synced to ${environment}`);
        if (cleaned) console.log(cleaned);
        resolvePromise();
        return;
      }

      reject(new Error(`[sync-vercel-env] ${key}: failed with code ${code}\n${cleaned}`));
    });

    child.stdin.end(`${value}\n`);
  });
}
