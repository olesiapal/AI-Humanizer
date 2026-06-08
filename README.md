# AI Humanizer

AI Humanizer is a Next.js app for rewriting generated text into more natural, human-sounding prose. It can use GPTZero plus AI providers such as Gemini, OpenAI, and Anthropic.

## Features

- Next.js app router frontend and API route.
- Provider integrations for Gemini, OpenAI, and Anthropic.
- GPTZero integration for detection feedback.
- Safe local development scripts that keep Node memory limits predictable.
- Vercel-ready configuration.

## Setup

```bash
npm install
cp .env.local.example .env.local
npm run dev
```

Open the local URL printed by Next.js.

## Environment

Fill the keys you plan to use in `.env.local`:

```bash
GPTZERO_API_KEY=your_gptzero_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Keep real secrets in `.env.local`. The committed `.env.local.example` contains placeholders only.

## Scripts

```bash
npm run dev
npm run dev:next
npm run dev:turbo
npm run build
npm start
npx tsc --noEmit
```

## Git Hygiene

The repository ignores real env files, local Claude and VS Code settings, `.next`, `node_modules`, TypeScript build info, and other generated artifacts.
