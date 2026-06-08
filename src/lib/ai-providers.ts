import type { GPTZeroRewriteFeedback } from './gptzero';

export type AIModel =
  | 'gemini-3-flash-preview'
  | 'gemini-3-pro-preview'
  | 'gemini-2.5-flash'
  | 'gemini-2.5-pro'
  | 'gpt-5.5'
  | 'gpt-4o'
  | 'gpt-4o-mini'
  | 'claude-opus-4-7'
  | 'claude-sonnet-4-6'
  | 'claude-haiku-4-5';

export type AIProvider = 'gemini' | 'openai' | 'anthropic';
export type ImpactLevel = 'high' | 'medium' | 'low';

export interface RewriteAttempt {
  iteration: number;
  score: number;
  text: string;
  feedback: GPTZeroRewriteFeedback;
  sentenceCount: number;
  flaggedSentenceCount: number;
}

export interface RewriteRequest {
  originalText: string;
  currentText: string;
  currentFeedback: GPTZeroRewriteFeedback;
  model: AIModel;
  attempts: RewriteAttempt[];
  sourceSentenceCount: number;
}

export interface InitialRewriteRequest {
  text: string;
  model: AIModel;
  styleProfile?: string;
}

export interface DeepRewriteRequest {
  text: string;
  model: AIModel;
  styleProfile?: string;
}

export interface DeepRewriteDraftRequest extends DeepRewriteRequest {
  analysis: string;
  teacherPlan?: string;
}

export interface CadenceCleanupRequest {
  originalText: string;
  draft: string;
  model: AIModel;
  styleProfile?: string;
  teacherPlan?: string;
}

export interface SemanticCoverageRequest {
  originalText: string;
  rewrittenText: string;
  model: AIModel;
}

export interface SemanticCoverageResult {
  ok: boolean;
  issues: string[];
}

export interface SemanticRepairRequest extends SemanticCoverageRequest {
  coverageIssues: string[];
  styleProfile?: string;
}

export interface ImpactTarget {
  text: string;
  level: ImpactLevel;
}

export interface TargetedRewriteRequest {
  currentText: string;
  targets: ImpactTarget[];
  model: AIModel;
  stageLabel: string;
  attemptIndex?: number;
  styleProfile?: string;
  teacherPlan?: string;
}

export interface WritingTeacherRequest {
  text: string;
  model: AIModel;
  styleProfile?: string;
  semanticAnalysis?: string;
  targets?: ImpactTarget[];
  stageLabel?: string;
  attemptIndex?: number;
}

const CANDIDATE_COUNT = 4;
const TARGETED_CANDIDATE_COUNT = 3;
const REPLACEMENT_COUNT = 6;

export function getProvider(model: AIModel): AIProvider {
  if (model.startsWith('gemini')) return 'gemini';
  if (model.startsWith('gpt')) return 'openai';
  return 'anthropic';
}

export interface SentenceReplacementRequest {
  fullText: string;
  targetSentence: string;
  targetNotes: string[];
  model: AIModel;
  attempts: RewriteAttempt[];
}

const TRANSFORM_SYSTEM_PROMPT = `You rewrite text as a real person would revise a technical PR or issue note.

The output must keep the same language as the input. If the input is Russian, informal "ты" is allowed when the sentence naturally addresses the reader. If the input is English, use "you/your" or neutral phrasing instead of defaulting to "we/our".

The goal is not safe corporate prose and not fake-friendly chat. The goal is a clear, plain maintainer note that breaks the AI-template shape.

Non-negotiables:
- Preserve the meaning, factual claims, names, file paths, identifiers, numbers, status codes, field names, and edge cases.
- Do not invent new facts, fake anecdotes, citations, or personal backstory.
- Do not output notes about the edit. For rewrite prompts, answer with the rewritten text and nothing else. For helper prompts that explicitly ask for analysis, coverage JSON, or a teacher plan, return only the requested tagged helper output.
- Never corrupt technical tokens such as frontend/src/server/public-api/digest-service.ts, listActive, subscribedAt, subscription_not_found, or 409.

Style:
- Write like a maintainer leaving a direct PR/issue comment after reading the code, not like a generated requirements document.
- Start with the concrete broken thing and consequence. Do not start with a greeting, social filler, or "quick heads up".
- Do not lean on "we need", "we should", "we have", "our current", "to resolve this", "ensure", "must", "should", "currently", "functionality", "data integrity", "overall", "you'll want", "make sure", "watch out", "properly", or "correctly".
- Avoid fake human friendliness: no "Hey", "heads up", "quick update", "just", "things like", "can trip you up", "throw a wrench", "catch you off guard", "keep everything neat", or "slipping through the cracks".
- Prefer concrete maintainer phrasing: "Stripe checkout is trusting the browser userId", "the webhook side has to resolve the same owner", "missing owner data should not create billing state", "status and price mapping land in...".
- SMS-ish roughness is allowed in small doses: dont, cant, tho, kinda, a missing apostrophe, an uneven comma, or a sentence that runs a little long. Do not use it in every sentence.
- Light metaphor is allowed only if it is not a cliché. Avoid polished risk metaphors.
- Markdown ticks usually make the text look templated. Avoid them unless the source meaning truly depends on code formatting. Plain text or double quotes is fine.
- Merge related requirements hard. A few dense sentences are better than many clean little instructions.
- Do not keep the source sentence count. Do not rewrite sentence 1 into sentence 1, sentence 2 into sentence 2, and so on.
- Do not turn the text into one giant run-on sentence. Dense is fine; word soup is not.
`;

function buildPrompt(request: RewriteRequest): string {
  const originalSentenceCount = Math.max(
    request.sourceSentenceCount,
    estimateSentenceCount(request.originalText),
    1
  );
  const targetSentenceCount = Math.max(2, Math.ceil(originalSentenceCount * 0.35));
  const attemptHistory = formatAttemptHistory(request.attempts);

  return `Transform the original text into simpler, more human colleague-style versions.

Original text:

${request.originalText}

---
Current failed draft:

${request.currentText}

---
Full iteration history and GPTZero feedback:

${attemptHistory}

---
What the next rewrite must learn from that history:
- The whole passage is failing, not just isolated sentences.
- GPTZero keeps seeing atomic, self-contained requirement statements. That shape has to be broken.
- Do not treat the highlighted sentences as a checklist. They are evidence of the failed pattern.
- If a failed draft shows low sentence "ai" class but high "paraphrased" class, that is closer than pure AI, but still not enough. Keep the meaning and rough shape, then make it less like a cleaned paraphrase: change the opening, add a more natural explanation path, and remove polished requirement cadence.
- The original has about ${originalSentenceCount} sentences. Aim for roughly ${Math.max(
    targetSentenceCount,
    5
  )}-${Math.max(targetSentenceCount + 3, 8)} readable sentences unless the input is very short.
- Each new sentence should usually carry several connected requirements through clauses, commas, parentheses, dashes, or a natural "if...then..." shape.
- Keep sentence density under control: if a sentence starts carrying more than 50-60 words, split it.
- If a previous draft kept repeating "we need", "we should", "must", "should", "currently", or clean issue-template phrasing, move away from it.
- Keep all concrete requirements, but change the surface completely: order, rhythm, paragraphing, sentence boundaries, and framing.
- Use "you" or neutral phrasing before "we". If "we" appears at all, it should be rare, not the skeleton of the text.
- The final should sound like a teammate wrote it quickly but clearly, not like a polished AI instruction block.

Hard bans learned from failed drafts:
- Do not start with "The public digest signup flow", "The digest signup flow", "To fix this", "To address this", "Check out", or "Head to".
- Avoid the phrase pattern "when it comes to", "lastly", "just remember", "make sure", "focus on", "stick with", and "this means".
- Avoid turning the output into imperative documentation. It should read like an actual note, not a task list with softer words.
- Do not use "we need", "we should", "must", or "should" as the backbone. Prefer "you’ll see", "the service ends up", "the fix mostly lives in", "let", "keep", "turns into", "slips through".

Create exactly ${CANDIDATE_COUNT} distinct candidates. They must not share the same opening sentence or paragraph shape.

Candidate 1: one dense teammate note, direct and plain, but still readable.
Candidate 2: more narrative, like explaining the bug trail to one colleague; one light metaphor is ok.
Candidate 3: rougher DM style, with a tiny bit of SMS-ish texture such as dont/cant/tho, but keep all technical tokens exact.
Candidate 4: compressed bug-comment style, no headings, no bullet points, no markdown ticks, and no clean spec cadence, but not a single giant sentence.

Wrap each candidate exactly like this:
<candidate id="1">
text
</candidate>
<candidate id="2">
text
</candidate>
<candidate id="3">
text
</candidate>
<candidate id="4">
text
</candidate>`;
}

function buildSentenceReplacementPrompt(request: SentenceReplacementRequest): string {
  const attemptHistory = formatAttemptHistory(request.attempts);
  const targetNotes =
    request.targetNotes.length > 0
      ? request.targetNotes.map((note) => `- ${note}`).join('\n')
      : '- No sentence-level notes were returned.';

  return `One sentence in this document is driving the AI score. Do not rewrite the whole document.

Full document:

${request.fullText}

---
High-impact sentence to replace:

${request.targetSentence}

---
GPTZero notes for that sentence:
${targetNotes}

---
Recent failed attempts and GPTZero feedback:

${attemptHistory}

---
Create exactly ${REPLACEMENT_COUNT} replacements for ONLY the high-impact sentence.

Rules:
- Preserve the exact meaning and technical tokens.
- The replacement can be one sentence or two short sentences.
- Do not start with the same first words as the target.
- Prefer a small human edit over a full paraphrase. Splitting one polished sentence into two slightly plain sentences is good.
- "So ..." is allowed when it sounds natural.
- Avoid clean list syntax and generated-doc cadence.
- Do not add surrounding context. Return replacement text only inside each tag.

Wrap each replacement exactly like this:
<replacement id="1">
text
</replacement>
<replacement id="2">
text
</replacement>
<replacement id="3">
text
</replacement>
<replacement id="4">
text
</replacement>
<replacement id="5">
text
</replacement>
<replacement id="6">
text
</replacement>`;
}

function buildInitialRewritePrompt(request: InitialRewriteRequest): string {
  const styleProfile = formatStyleProfile(request.styleProfile);

  return `Rewrite this text once into a simple, human, colleague-written version.

The result should be plain and a little alive, not a polished AI task spec. Remove stiff phrases like "The fix is small", "The problem is", "there is/there are", "we need", "ensure", "currently", "should/must" cadence, and other generated-instruction rhythm.

Keep every technical requirement, file path, field name, identifier, number, status code, edge case, and relationship intact. Do not add new facts. Do not turn it into marketing copy. Do not use headings unless the source truly needs them.

Style profile:
${styleProfile}

Return only the rewritten text wrapped in:
<rewrite>
text
</rewrite>

Text:

${request.text}`;
}

function buildDeepAnalysisPrompt(request: DeepRewriteRequest): string {
  return `Read the source as a technical note that needs to be rebuilt, not paraphrased.

Extract the meaning into a compact working brief. Keep exact technical tokens, paths, identifiers, numbers, status codes, collection names, field names, and edge cases. Do not rewrite the prose yet.

Return only:
<analysis>
Problem:
- ...

Required behavior:
- ...

Edge cases:
- ...

Must keep exact:
- ...

Do not invent:
- ...
</analysis>

Source:

${request.text}`;
}

function buildWritingTeacherPrompt(request: WritingTeacherRequest): string {
  const styleProfile = formatStyleProfile(request.styleProfile);
  const impactTargets =
    request.targets && request.targets.length > 0
      ? request.targets
          .map((target, index) => `${index + 1}. ${target.level.toUpperCase()}: ${target.text}`)
          .join('\n')
      : 'No GPTZero impact symptoms were provided.';
  const semanticAnalysis = request.semanticAnalysis?.trim() || 'No semantic analysis was provided.';

  return `You are the Writing Teacher for a technical humanizer pipeline.

Do not rewrite the text. Diagnose the text's shape and produce a concrete editing plan for the next writer model. The plan is tactical: where to change wording, where to rewrite a sentence, where to rebuild a paragraph, where to merge/split/reorder, and where enum/mapping lists need to stop looking like generated reference docs.

Hard rules:
- Do not add facts.
- Do not remove facts.
- Do not write final prose.
- Treat GPTZero impact lines as symptoms, not truth. The real issue may be nearby wording, paragraph rhythm, or the entire requirements cadence.
- Preserve technical tokens, paths, identifiers, numbers, status codes, enum values, field names, and edge cases.
- Style profile is a tone guide only, never a source of facts.

Look especially for:
- polished abstract phrases like "inconsistency affects quality", "essential", "validated emotional state", "selected technique", "normal failure response"
- repeated should/must/needs/has to/ensure grammar
- smooth "Requirement. Requirement. Requirement." cadence
- long enum or mapping paragraphs that read like generated docs
- highlighted sentences that need paragraph-level repair instead of one-line replacement
- places where two short requirements should be merged, or one dense sentence should be split
- openings that sound generic, such as "FunctionName can..." or "The issue is..."

Return a compact plan only. Use these operation labels when useful:
- wording
- sentence
- paragraph
- merge
- split
- reorder
- enum-breakup
- opening
- tests

Return only:
<teacher_plan>
Diagnosis:
- ...

Operations:
- [paragraph] ...
- [wording] ...
- [merge] ...

Writer warnings:
- ...
</teacher_plan>

Style profile:
${styleProfile}

Semantic analysis:
${semanticAnalysis}

GPTZero impact symptoms:
${impactTargets}

Stage: ${request.stageLabel ?? 'deep rewrite'}
Attempt: ${request.attemptIndex ?? 1}

Text to diagnose:

${request.text}`;
}

function buildDeepRewritePrompt(request: DeepRewriteDraftRequest): string {
  const styleProfile = formatStyleProfile(request.styleProfile);
  const teacherPlan = formatTeacherPlan(request.teacherPlan);

  return `Write a new human maintainer-style PR/issue note from the analysis below. Do not paraphrase the source sentence by sentence; rebuild the text from the extracted meaning.

Use the same language as the source. Keep every technical requirement and exact token from the analysis. The output should feel like a maintainer wrote it after reading the broken code: clear, plain, slightly uneven, and specific. Avoid headings unless the source truly needs them.

Start with the concrete broken path or behavior. Do not start with "Hey", "heads up", "quick update", "There is", "The issue is", or "The problem is".
For code-state bugs, prefer the relationship that breaks over a generic function opener: "In claimCoupon, the user row and coupon row drift apart" is usually better than "claimCoupon can...".

Style profile:
${styleProfile}

Writing teacher plan:
${teacherPlan}

Avoid:
- "we need", "we should", "ensure", "currently", "the issue is", "the problem is", "to fix this", "overall", "functionality"
- "you'll want", "make sure", "watch out", "when dealing with", "keep everything neat", "things like", "can trip you up", "throw a wrench", "catch you off guard"
- repetitive "FunctionName can..." openings and clean "Requirement. Requirement. Requirement." cadence
- friendly greeting style or fake casual tone
- clean spec rhythm where every sentence is a standalone requirement
- turning the note into one giant sentence

Analysis:

${request.analysis}

Original source for reference only:

${request.text}

Return only:
<rewrite>
text
</rewrite>`;
}

function buildCadenceCleanupPrompt(request: CadenceCleanupRequest): string {
  const styleProfile = formatStyleProfile(request.styleProfile);
  const teacherPlan = formatTeacherPlan(request.teacherPlan);

  return `Clean up the draft's AI cadence without changing the facts.

Goal: make it sound less like generated instructions and less like fake-friendly chat. It should read like a direct maintainer note. Keep the same language as the draft. Preserve every file path, identifier, number, status code, field name, edge case, and technical relationship.

Style profile:
${styleProfile}

Writing teacher plan:
${teacherPlan}

Remove or soften:
- "we need", "we should", "ensure", "currently", "the issue is", "the problem is", "to fix this", "make sure", "it is important"
- "Hey", "heads up", "quick update", "you'll want", "watch out", "when dealing with", "keep everything neat", "things like", "trip you up", "throw a wrench", "catch you off guard", "slipping through the cracks"
- generic openings like "FunctionName can..." when a more specific state relationship is available
- repeated modals: every sentence saying "needs to", "has to", "should", or "must"
- abstract detector bait when the exact token is not required: "inconsistency affects quality", "essential", "consistently adhere", "chosen technique", "selected technique", "validated emotional state", "normal failure response", "mapping is fixed", "is limited to", "anything unusual", repeated "no saved data"
- repeated sentence openings
- overly polished requirement cadence
- fake enthusiasm and corporate safety language
- generic imperative chains like "Check X, ensure Y, make sure Z"; convert them into a more natural explanation of how the code path should behave
- overly smooth enum/mapping catalogues. Keep the values, but break the rhythm so it reads like notes from a code review, not reference docs.

Do not add new facts. Do not delete requirements. Return only the cleaned text.

Original source:

${request.originalText}

Draft:

${request.draft}

Return only:
<rewrite>
text
</rewrite>`;
}

function buildSemanticCoveragePrompt(request: SemanticCoverageRequest): string {
  return `Compare the rewritten text against the original source.

Check only factual and technical coverage. Ignore style differences. The rewrite is allowed to reorder, merge, split, and sound more casual, but it must not lose or distort requirements, edge cases, file paths, identifiers, numbers, status codes, fields, collections, or relationships.

Return strict JSON inside <coverage>. No markdown.

Schema:
{
  "ok": true | false,
  "issues": ["missing or distorted fact", "..."]
}

Original:

${request.originalText}

Rewrite:

${request.rewrittenText}

Return only:
<coverage>{"ok":true,"issues":[]}</coverage>`;
}

function buildSemanticRepairPrompt(request: SemanticRepairRequest): string {
  const styleProfile = formatStyleProfile(request.styleProfile);
  const issues = request.coverageIssues.map((issue, index) => `${index + 1}. ${issue}`).join('\n');

  return `Repair the rewrite by adding or correcting ONLY the missing/distorted facts listed below.

Keep the text human and colleague-written. Preserve the rewrite's better rhythm. Do not restart from the original source and do not turn it back into a stiff checklist.

Style profile:
${styleProfile}

Coverage issues to fix:
${issues}

Original source:

${request.originalText}

Current rewrite:

${request.rewrittenText}

Return only:
<rewrite>
text
</rewrite>`;
}

function buildTargetedRewritePrompt(
  request: TargetedRewriteRequest,
  outputMode: 'single' | 'candidates' = 'single'
): string {
  const styleProfile = formatStyleProfile(request.styleProfile);
  const teacherPlan = formatTeacherPlan(request.teacherPlan);
  const attemptStrategy = formatTargetedAttemptStrategy(request.attemptIndex ?? 1);
  const targets = request.targets
    .map((target, index) => `${index + 1}. ${target.level.toUpperCase()}: ${target.text}`)
    .join('\n');
  const outputInstruction =
    outputMode === 'candidates'
      ? `Create exactly ${TARGETED_CANDIDATE_COUNT} different full rewrites. They must not share the same opening or paragraph shape.

Candidate 1: keep it plain and explanatory, with the broken state relationship up front.
Candidate 2: make it more like a reviewer comment, with one natural "I'd cover..." or "I'd test..." line if tests/edges are present.
Candidate 3: make it more compressed and uneven, with a different fact order and no polished requirement sequence.

Wrap each candidate exactly like this:
<candidate id="1">
text
</candidate>
<candidate id="2">
text
</candidate>
<candidate id="3">
text
</candidate>`
      : `Return only:
<rewrite>
full rewritten text
</rewrite>`;

  return `Rewrite the full text below using the impact targets as diagnostic clues, not as a literal sentence-replacement checklist.

Stage: ${request.stageLabel}
Attempt: ${request.attemptIndex ?? 1}

Attempt strategy:
${attemptStrategy}

Important context:
- GPTZero impact lines are often symptoms, not the real cause. The sentence before, after, or a whole paragraph rhythm may be what makes the text read as AI.
- High impact can migrate: if one marked sentence disappears, GPTZero may simply move the high-impact label to the next most templated sentence. Your job is to reduce the whole passage's opportunity set, not just satisfy the visible marker.
- Do not chase a one-line swap. Rebuild the surrounding structure so the marked lines stop being isolated, polished requirement atoms.
- You may change adjacent sentences, merge/split paragraphs, reorder connected requirements, and replace framing elsewhere in the text if that makes the whole passage feel more human.
- The listed targets should not survive with the same surface form, but the fix is allowed to happen around them rather than exactly inside them.
- Hunt likely next markers yourself: clean imperative chains like "Check access..., ensure..., make sure...", generic transitions like "Additionally" / "Also" / "If that happens", and polished risk wrap-ups like "slipping through the cracks" can become the next high-impact line even if they are not listed.
- The failed examples to avoid look like this: "Hey, heads up...", "You'll want to...", "make sure...", "When dealing with events...", "Keep everything neat...", "Watch out for...", "things can trip you up." Do not produce that style.

Rules:
- Return the whole text, not just the changed lines.
- Treat every target as a pressure point. Address all of them in one pass, but do not limit edits to those exact strings.
- Preserve all technical facts, file paths, identifiers, field names, status codes, cookie names, and edge cases.
- Keep facts stable, not wording. If a "decent" part helps the AI rhythm, rewrite it too.
- Prefer context-level changes: change the opening path, paragraph shape, transitions, sentence boundaries, and the way requirements are grouped.
- Rewrite formulaic imperative/spec clusters across the whole text, especially "Check X, ensure Y, make sure Z" shapes.
- If several requirements are naturally connected, fold them into a more natural explanation instead of leaving a clean checklist.
- The output should sound like a maintainer wrote it plainly after reading the code, not like an AI spec and not like a cheerful assistant.
- Avoid clean generated cadence: "we need", "we should", "ensure", "currently", "the fix is", "this means", "to resolve this", "you'll want", "make sure", "watch out", "properly", "correctly".
- Avoid overusing "needs", "has to", "should", or "make sure" as the grammar for every requirement. Mix direct facts, cause/effect, and concrete code-path behavior.
- Treat words as signals too. Replace abstract detector bait like "inconsistency affects quality", "essential", "consistently adhere", "chosen technique", "selected technique", "validated emotional state", "normal failure response", "mapping is fixed", "is limited to", "anything unusual", and repeated "no saved data" phrasing when the exact token is not required.
- Long whitelists and mappings are dangerous when they become one polished catalogue. Keep the tokens, but present them like notes from someone reading code: a little broken up, with one practical reason attached, not a perfectly smooth reference paragraph.
- Do not use a chatty opener. Start with the concrete bug or code path.
- Do not open with a bare "FunctionName can..." or "FunctionName currently...". For code bugs, a stronger human opening is usually a relationship or state drift: "In claimCoupon, the user row and coupon row drift apart..." or "Stripe checkout is trusting the browser userId...".
- One small reviewer-like cue is allowed when natural: "I'd cover...", "that part matters because...", "otherwise...". Do not turn the whole text into first-person commentary.
- Do not include explanations, notes, or a changelog.
- If this is attempt 2 or later, make a meaning-preserving structural move, not just synonyms: different opening, different grouping, different paragraph rhythm, or a different explanation order.

Style profile:
${styleProfile}

Writing teacher plan:
${teacherPlan}

Impact symptoms:
${targets}

Current text:

${request.currentText}

${outputInstruction}`;
}

function formatTargetedAttemptStrategy(attemptIndex: number): string {
  const normalized = ((attemptIndex - 1) % 4) + 1;

  switch (normalized) {
    case 1:
      return `- State-drift note. Explain what two pieces of state stop matching, then describe the transaction/update shape that fixes it.
- Use 2-3 paragraphs. Let one sentence be short and blunt.
- Good pattern: "In X, A and B drift apart: ..." then the consequence.`;
    case 2:
      return `- Bug-trail note. Start from the user-visible or data consequence, then walk backward to the code path.
- Change the paragraph count from the current text if possible.
- Include one natural "otherwise" or "that leaves..." consequence instead of a polished risk summary.`;
    case 3:
      return `- Reviewer comment. Write it like a maintainer leaving a focused review: one compact main paragraph, then a smaller test/edge-case paragraph.
- A light first-person reviewer phrase is allowed once, such as "I'd cover..." for tests.
- Avoid making every sentence start with the function name or an imperative verb.`;
    default:
      return `- Structural reset. Reorder the facts more aggressively while keeping meaning: begin with the worst failure mode, then the guard/transaction rule, then edge cases/tests.
- Use uneven sentence lengths. One sentence may carry a parenthetical or afterthought if it sounds natural.
- Do not reuse the current opening phrase or the same three-paragraph skeleton.`;
  }
}

function formatStyleProfile(styleProfile: string | undefined): string {
  const value = styleProfile?.trim();
  if (!value) {
    return '- Default: clear teammate note, plain, direct, a little uneven, never corporate.';
  }

  return value
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => `- ${line}`)
    .join('\n');
}

function formatTeacherPlan(teacherPlan: string | undefined): string {
  const value = teacherPlan?.trim();
  if (!value) {
    return '- No separate teacher plan. Use the surrounding instructions.';
  }

  return value;
}

function formatAttemptHistory(attempts: RewriteAttempt[]): string {
  if (attempts.length === 0) {
    return 'No previous attempts yet.';
  }

  return attempts.map(formatAttempt).join('\n\n---\n\n');
}

function formatAttempt(attempt: RewriteAttempt): string {
  return `Iteration ${attempt.iteration}: ${Math.round(attempt.score * 100)}% AI
Sentence count: ${attempt.sentenceCount}
Flagged sentences: ${attempt.flaggedSentenceCount}

Draft:
${attempt.text}

GPTZero feedback:
${formatFeedback(attempt.feedback)}`;
}

function formatFeedback(feedback: GPTZeroRewriteFeedback): string {
  const documentNotes =
    feedback.documentNotes.length > 0
      ? feedback.documentNotes.map((note) => `- ${note}`).join('\n')
      : '- No document-level diagnostics were returned.';

  const sentenceNotes =
    feedback.sentences.length > 0
      ? feedback.sentences
          .map(
            (sentence, index) =>
              `${index + 1}. ${Math.round(sentence.aiProbability * 100)}% AI: "${
                sentence.text
              }"\n   ${sentence.notes.join('; ')}`
          )
          .join('\n')
      : '- GPTZero did not isolate specific sentences.';

  return `${documentNotes}

Flagged sentence details:
${sentenceNotes}`;
}

function estimateSentenceCount(text: string): number {
  return (
    text
      .split(/(?<=[.!?])\s+|\n+/)
      .map((sentence) => sentence.trim())
      .filter(Boolean).length || 1
  );
}

async function callGemini(model: AIModel, prompt: string): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY не задан в .env.local');

  const { GoogleGenerativeAI } = await import('@google/generative-ai');
  const genAI = new GoogleGenerativeAI(apiKey);
  const genModel = genAI.getGenerativeModel({
    model,
    systemInstruction: TRANSFORM_SYSTEM_PROMPT,
    generationConfig: {
      maxOutputTokens: 12000,
      temperature: 1.25,
      topP: 0.97,
      topK: 80,
    },
  });

  const result = await genModel.generateContent(prompt);
  return result.response.text();
}

async function withAIRetry<T>(operation: () => Promise<T>, label: string): Promise<T> {
  let lastError: unknown;

  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      if (attempt < 3) {
        await sleep(900 * attempt);
      }
    }
  }

  throw new Error(`${label}: ${lastError instanceof Error ? lastError.message : String(lastError)}`);
}

async function callOpenAI(model: AIModel, prompt: string): Promise<string> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY не задан в .env.local');

  if (model.startsWith('gpt-5')) {
    const response = await postJson<OpenAIResponsesResponse>('https://api.openai.com/v1/responses', {
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      body: {
      model,
      instructions: TRANSFORM_SYSTEM_PROMPT,
      input: prompt,
      max_output_tokens: 12000,
      store: false,
      },
    });

    return response.output_text ?? extractOpenAIResponsesText(response);
  }

  const response = await postJson<OpenAIChatResponse>('https://api.openai.com/v1/chat/completions', {
    headers: {
      Authorization: `Bearer ${apiKey}`,
    },
    body: {
    model,
    messages: [
      { role: 'system', content: TRANSFORM_SYSTEM_PROMPT },
      { role: 'user', content: prompt },
    ],
    temperature: 1.15,
    top_p: 0.95,
    frequency_penalty: 0.6,
    presence_penalty: 0.2,
    max_tokens: 12000,
    },
  });

  return response.choices[0]?.message?.content ?? '';
}

async function callAnthropic(model: AIModel, prompt: string): Promise<string> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error('ANTHROPIC_API_KEY не задан в .env.local');

  const response = await postJson<AnthropicMessagesResponse>('https://api.anthropic.com/v1/messages', {
    headers: {
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: {
    model,
    max_tokens: 12000,
    temperature: 1,
    system: TRANSFORM_SYSTEM_PROMPT,
    messages: [{ role: 'user', content: prompt }],
    },
  });

  const block = response.content[0];
  if (block.type !== 'text' || typeof block.text !== 'string') {
    throw new Error('Unexpected response type from Anthropic');
  }
  return block.text;
}

interface JsonPostOptions {
  headers: Record<string, string>;
  body: unknown;
}

interface OpenAIChatResponse {
  choices: Array<{
    message?: {
      content?: string;
    };
  }>;
}

interface OpenAIResponsesResponse {
  output_text?: string;
  output?: Array<{
    content?: Array<{
      type?: string;
      text?: string;
    }>;
  }>;
}

interface AnthropicMessagesResponse {
  content: Array<{
    type: string;
    text?: string;
  }>;
}

async function postJson<T>(url: string, options: JsonPostOptions): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    body: JSON.stringify(options.body),
  });

  const text = await response.text();

  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${text.slice(0, 500)}`);
  }

  return JSON.parse(text) as T;
}

function extractOpenAIResponsesText(response: OpenAIResponsesResponse): string {
  return (
    response.output
      ?.flatMap((item) => item.content ?? [])
      .map((content) => content.text ?? '')
      .join('')
      .trim() ?? ''
  );
}

export async function generateRewriteCandidates(request: RewriteRequest): Promise<string[]> {
  const prompt = buildPrompt(request);
  const raw = await callSelectedProvider(request.model, prompt);

  return parseCandidates(raw);
}

export async function generateSentenceReplacements(
  request: SentenceReplacementRequest
): Promise<string[]> {
  const prompt = buildSentenceReplacementPrompt(request);
  const raw = await callSelectedProvider(request.model, prompt);

  return parseReplacements(raw);
}

export async function generateInitialRewrite(request: InitialRewriteRequest): Promise<string> {
  const raw = await callSelectedProvider(request.model, buildInitialRewritePrompt(request));
  return parseSingleRewrite(raw);
}

export async function analyzeRewriteFacts(request: DeepRewriteRequest): Promise<string> {
  const raw = await callSelectedProvider(request.model, buildDeepAnalysisPrompt(request));
  return parseTaggedBlock(raw, 'analysis');
}

export async function planRewriteWithTeacher(request: WritingTeacherRequest): Promise<string> {
  const raw = await callSelectedProvider(request.model, buildWritingTeacherPrompt(request));
  return parseTaggedBlock(raw, 'teacher_plan');
}

export async function generateDeepRewrite(request: DeepRewriteDraftRequest): Promise<string> {
  const raw = await callSelectedProvider(request.model, buildDeepRewritePrompt(request));
  return parseSingleRewrite(raw);
}

export async function cleanupAiCadence(request: CadenceCleanupRequest): Promise<string> {
  const raw = await callSelectedProvider(request.model, buildCadenceCleanupPrompt(request));
  return parseSingleRewrite(raw);
}

export async function verifySemanticCoverage(
  request: SemanticCoverageRequest
): Promise<SemanticCoverageResult> {
  const raw = await callSelectedProvider(request.model, buildSemanticCoveragePrompt(request));
  return parseSemanticCoverage(raw);
}

export async function repairSemanticCoverage(request: SemanticRepairRequest): Promise<string> {
  const raw = await callSelectedProvider(request.model, buildSemanticRepairPrompt(request));
  return parseSingleRewrite(raw);
}

export async function generateTargetedRewrite(
  request: TargetedRewriteRequest
): Promise<string> {
  const raw = await callSelectedProvider(request.model, buildTargetedRewritePrompt(request));
  return parseSingleRewrite(raw);
}

export async function generateTargetedRewriteCandidates(
  request: TargetedRewriteRequest
): Promise<string[]> {
  const raw = await callSelectedProvider(
    request.model,
    buildTargetedRewritePrompt(request, 'candidates')
  );
  return parseCandidates(raw);
}

async function callSelectedProvider(model: AIModel, prompt: string): Promise<string> {
  const provider = getProvider(model);

  switch (provider) {
    case 'gemini':
      return withAIRetry(() => callGemini(model, prompt), 'Gemini connection error');
    case 'openai':
      return withAIRetry(() => callOpenAI(model, prompt), 'OpenAI connection error');
    case 'anthropic':
      return withAIRetry(() => callAnthropic(model, prompt), 'Anthropic connection error');
  }
}

function postProcessRewrite(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, (block) => block.replace(/```/g, ''))
    .replace(/`([^`\n]+)`/g, '"$1"')
    .replace(/^\s*(?:hey[,!]?\s*)?(?:just\s+)?(?:a\s+)?(?:quick\s+)?heads\s+up(?:\s+about|\s+on)?\s*/i, '')
    .replace(/^\s*(?:hey[,!]?\s*)?(?:just\s+)?(?:a\s+)?quick\s+update(?:\s+on|\s+about)?\s*/i, '')
    .trim();
}

function parseCandidates(raw: string): string[] {
  const tagged = [...raw.matchAll(/<candidate(?:\s+id=["']?\d+["']?)?\s*>([\s\S]*?)<\/candidate>/gi)]
    .map((match) => postProcessRewrite(match[1] ?? ''))
    .filter(Boolean);

  const candidates = tagged.length > 0 ? tagged : splitUntaggedCandidates(raw);
  const unique: string[] = [];
  const seen = new Set<string>();

  for (const candidate of candidates) {
    const normalized = candidate.replace(/\s+/g, ' ').trim().toLowerCase();
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    unique.push(candidate);
    if (unique.length >= CANDIDATE_COUNT) break;
  }

  return unique.length > 0 ? unique : [postProcessRewrite(raw)];
}

function parseSingleRewrite(raw: string): string {
  const tagged = raw.match(/<rewrite\s*>([\s\S]*?)<\/rewrite>/i);
  return postProcessRewrite(tagged?.[1] ?? raw);
}

function parseTaggedBlock(raw: string, tag: string): string {
  const tagged = raw.match(new RegExp(`<${tag}\\s*>([\\s\\S]*?)<\\/${tag}>`, 'i'));
  return (tagged?.[1] ?? raw).trim();
}

function parseSemanticCoverage(raw: string): SemanticCoverageResult {
  const block = parseTaggedBlock(raw, 'coverage')
    .replace(/```(?:json)?/gi, '')
    .replace(/```/g, '')
    .trim();

  try {
    const parsed = JSON.parse(block) as Partial<SemanticCoverageResult>;
    return {
      ok: parsed.ok === true,
      issues: Array.isArray(parsed.issues)
        ? parsed.issues.map((issue) => String(issue).trim()).filter(Boolean)
        : [],
    };
  } catch {
    const hasProblemSignal = /\b(missing|lost|omitted|incorrect|distorted|changed|not covered)\b/i.test(
      block
    );
    return {
      ok: !hasProblemSignal,
      issues: hasProblemSignal ? [block.slice(0, 1200)] : [],
    };
  }
}

function parseReplacements(raw: string): string[] {
  const tagged = [...raw.matchAll(/<replacement(?:\s+id=["']?\d+["']?)?\s*>([\s\S]*?)<\/replacement>/gi)]
    .map((match) => postProcessRewrite(match[1] ?? ''))
    .filter(Boolean);

  const replacements = tagged.length > 0 ? tagged : splitUntaggedCandidates(raw);
  const unique: string[] = [];
  const seen = new Set<string>();

  for (const replacement of replacements) {
    const normalized = replacement.replace(/\s+/g, ' ').trim().toLowerCase();
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    unique.push(replacement);
    if (unique.length >= REPLACEMENT_COUNT) break;
  }

  return unique.length > 0 ? unique : [postProcessRewrite(raw)];
}

function splitUntaggedCandidates(raw: string): string[] {
  return raw
    .split(/(?:^|\n)\s*(?:candidate|вариант)\s*\d+\s*[:.)-]\s*/i)
    .map(postProcessRewrite)
    .filter((candidate) => candidate.length > 0);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
