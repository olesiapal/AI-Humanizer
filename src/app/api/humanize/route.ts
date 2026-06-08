import { NextRequest } from 'next/server';
import {
  buildRewriteFeedback,
  checkWithGPTZero,
  getAllAISentences,
  getHighImpactAISentences,
  GPTZeroDocument,
  GPTZeroSentence,
  hasSentenceImpactMetadata,
  isHuman,
} from '@/lib/gptzero';
import {
  AIModel,
  ImpactLevel,
  ImpactTarget,
  RewriteAttempt,
  analyzeRewriteFacts,
  cleanupAiCadence,
  generateDeepRewrite,
  generateInitialRewrite,
  generateRewriteCandidates,
  generateSentenceReplacements,
  generateTargetedRewrite,
  generateTargetedRewriteCandidates,
  planRewriteWithTeacher,
  repairSemanticCoverage,
  verifySemanticCoverage,
} from '@/lib/ai-providers';

export const runtime = 'nodejs';
export const maxDuration = 300;

const PASS_THRESHOLD = 0.6;
const MAX_ITERATIONS = 8;
const DEBUG_ENV = process.env.DEBUG_ENV === '1';

interface CandidateEvaluation {
  index: number;
  label: string;
  text: string;
  score: number;
  compositeScore: number;
  documentAiProbability: number | null;
  documentHumanProbability: number | null;
  averageSentenceAiClass: number | null;
  averageSentenceParaphrasedClass: number | null;
  readabilityPenalty: number;
  averageWordsPerSentence: number;
  maxWordsPerSentence: number;
  sentenceCount: number;
  flaggedSentenceCount: number;
  doc: GPTZeroDocument;
}

interface CandidateInput {
  label: string;
  text: string;
}

interface EvaluateCandidatesOptions {
  stopOnPass?: boolean;
}

interface LocalCandidateChoice {
  text: string;
  index: number;
  risk: number;
}

interface HumanizeRequestBody {
  text: string;
  model: AIModel;
  manualHighImpact?: string;
  useInitialRewrite?: boolean;
  useGptZero?: boolean;
  useDeepRewrite?: boolean;
  useWritingTeacher?: boolean;
  styleProfile?: string;
  impactAttempts?: number;
  fixAllImpactAtOnce?: boolean;
}

function send(writer: WritableStreamDefaultWriter<Uint8Array>, data: object) {
  const encoder = new TextEncoder();
  return writer.write(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
}

function formatUnknownError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function normalizeImpactAttempts(value: unknown): number {
  const numeric = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(numeric)) return 2;
  return Math.min(8, Math.max(1, Math.floor(numeric)));
}

function parseManualImpactTargets(value: string | undefined): ImpactTarget[] {
  if (!value) return [];

  const seen = new Set<string>();
  const targets: ImpactTarget[] = [];
  let pendingLevel: ImpactLevel | null = null;

  for (const rawLine of value.split(/\n+/)) {
    const line = rawLine
      .trim()
      .replace(/^\d{1,3}%\s*/, '')
      .replace(/^["'“”]+|["'“”]+$/g, '')
      .trim();

    if (!line) continue;
    const impactLevel = parseImpactLabel(line);
    if (impactLevel) {
      pendingLevel = impactLevel;
      continue;
    }

    if (/^(ai probability|class probabilities|perplexity|highlighted as)/i.test(line)) {
      continue;
    }

    const key = normalizeSentenceText(line);
    if (seen.has(key)) continue;

    seen.add(key);
    targets.push({ text: line, level: pendingLevel ?? 'high' });
    pendingLevel = null;
  }

  return targets;
}

function parseImpactLabel(value: string): ImpactLevel | null {
  const normalized = value.trim().toLowerCase();
  if (/^high\s+ai\s+impact$/.test(normalized)) return 'high';
  if (/^(medium|moderate)\s+ai\s+impact$/.test(normalized)) return 'medium';
  if (/^low\s+ai\s+impact$/.test(normalized)) return 'low';
  return null;
}

function getManualImpactSentences(
  manualTargets: ImpactTarget[],
  doc: GPTZeroDocument,
  currentText: string
): GPTZeroSentence[] {
  const resolved: GPTZeroSentence[] = [];
  const used = new Set<string>();

  for (const manualTarget of manualTargets) {
    const sentence = resolveManualSentenceText(manualTarget.text, doc, currentText);
    if (!sentence) continue;

    const key = normalizeSentenceText(sentence);
    if (used.has(key)) continue;

    used.add(key);
    resolved.push({
      sentence,
      generated_prob: 1,
      perplexity: 0,
      highlight_sentence_for_ai: true,
      class_probabilities: { human: 0, ai: 1, paraphrased: 0 },
      interpretability_value: manualTarget.level === 'high' ? 1 : manualTarget.level === 'medium' ? 0.5 : 0.15,
      interpretability_normalized_value:
        manualTarget.level === 'high' ? 1 : manualTarget.level === 'medium' ? 0.5 : 0.15,
      interpretability_designation: manualTarget.level,
      interpretability_alpha: null,
      special_highlight_type: `manual_${manualTarget.level}_impact`,
    });
  }

  return resolved;
}

function resolveManualSentenceText(
  manualSentence: string,
  doc: GPTZeroDocument,
  currentText: string
): string | null {
  if (currentText.includes(manualSentence)) return manualSentence;

  const normalizedManual = normalizeSentenceText(manualSentence);
  const matchingDocSentence = doc.sentences.find((sentence) => {
    const normalizedDocSentence = normalizeSentenceText(sentence.sentence);
    return (
      normalizedDocSentence === normalizedManual ||
      normalizedDocSentence.includes(normalizedManual) ||
      normalizedManual.includes(normalizedDocSentence)
    );
  });

  if (matchingDocSentence && currentText.includes(matchingDocSentence.sentence)) {
    return matchingDocSentence.sentence;
  }

  return null;
}

function normalizeSentenceText(value: string): string {
  return value.replace(/\s+/g, ' ').trim().toLowerCase();
}

export async function POST(req: NextRequest) {
  const {
    text,
    model,
    manualHighImpact,
    useInitialRewrite = true,
    useGptZero = true,
    useDeepRewrite = true,
    useWritingTeacher = true,
    styleProfile,
    impactAttempts = 2,
    fixAllImpactAtOnce = false,
  }: HumanizeRequestBody = await req.json();
  const manualImpactTargets = parseManualImpactTargets(manualHighImpact);
  const normalizedImpactAttempts = normalizeImpactAttempts(impactAttempts);
  const normalizedStyleProfile = typeof styleProfile === 'string' ? styleProfile.trim() : '';

  const { readable, writable } = new TransformStream<Uint8Array, Uint8Array>();
  const writer = writable.getWriter();

  (async () => {
    const originalText = text.trim();
    let currentText = originalText;
    let iteration = 0;
    let bestText = currentText;
    let bestScore = Number.POSITIVE_INFINITY;
    let completed = false;
    const failedAttempts: RewriteAttempt[] = [];

    try {
      await send(writer, {
        type: 'log',
        level: 'info',
        message: `Запускаю процесс гуманизации с моделью ${model}`,
      });

      if (manualImpactTargets.length > 0) {
        await send(writer, {
          type: 'log',
          level: 'info',
          message: `Ручной impact режим: получено ${manualImpactTargets.length} целей из GPTZero UI.`,
        });
      }

      if (DEBUG_ENV) {
        await send(writer, {
          type: 'log',
          level: 'info',
          message: `OpenAI key loaded: ${maskSecret(process.env.OPENAI_API_KEY)}`,
        });
      }

      if (!useGptZero) {
        await runNoGptZeroFlow({
          currentText,
          model,
          manualTargets: manualImpactTargets,
          useInitialRewrite,
          useDeepRewrite,
          useWritingTeacher,
          styleProfile: normalizedStyleProfile,
          fixAllAtOnce: fixAllImpactAtOnce,
          writer,
        });
        return;
      }

      if (manualImpactTargets.length > 0) {
        await runImpactRepairFlow({
          currentText,
          model,
          manualTargets: manualImpactTargets,
          attemptCount: normalizedImpactAttempts,
          fixAllAtOnce: fixAllImpactAtOnce,
          useDeepRewrite,
          useWritingTeacher,
          styleProfile: normalizedStyleProfile,
          writer,
        });
        return;
      }

      if (useInitialRewrite) {
        await runInitialRewriteFlow({
          currentText,
          model,
          useDeepRewrite,
          useWritingTeacher,
          styleProfile: normalizedStyleProfile,
          writer,
        });
      } else {
        await runScanOnlyFlow({ currentText, writer });
      }
      return;

      while (iteration < MAX_ITERATIONS) {
        iteration++;

        await send(writer, {
          type: 'log',
          level: 'check',
          message: `[Итерация ${iteration}/${MAX_ITERATIONS}] Проверяю текст через GPTZero...`,
        });

        let result;
        try {
          result = await checkWithGPTZero(currentText);
        } catch (e) {
          await send(writer, {
            type: 'log',
            level: 'error',
            message: `Ошибка GPTZero: ${formatUnknownError(e)}`,
          });
          break;
        }

        const doc = result.documents[0];
        const aiScore = doc.completely_generated_prob;
        const aiPct = Math.round(aiScore * 100);
        if (aiScore < bestScore) {
          bestScore = aiScore;
          bestText = currentText;
        }

        await send(writer, {
          type: 'score',
          score: aiScore,
          iteration,
          message: `GPTZero: ${aiPct}% AI-контент`,
          level: aiScore < 0.5 ? 'success' : aiScore < 0.75 ? 'warn' : 'error',
        });

        if (isPassingDocument(doc)) {
          await send(writer, {
            type: 'log',
            level: 'success',
            message: `✅ Текст прошёл проверку! AI-вероятность: ${aiPct}% (порог <60%)`,
          });
          await send(writer, { type: 'complete', text: currentText });
          completed = true;
          break;
        }

        const allAISentences = getAllAISentences(doc);
        const impactMetadataAvailable = hasSentenceImpactMetadata(doc);
        const manualTargets =
          manualImpactTargets.length > 0
            ? getManualImpactSentences(manualImpactTargets, doc, currentText)
            : [];
        const highImpactSentences =
          manualTargets.length > 0
            ? manualTargets
            : impactMetadataAvailable
              ? getHighImpactAISentences(doc)
              : [];
        const focusSentences =
          manualTargets.length > 0 ||
          (impactMetadataAvailable && highImpactSentences.length > 0)
            ? highImpactSentences
            : allAISentences;
        const rewriteFeedback = buildRewriteFeedback(doc, focusSentences);
        failedAttempts.push({
          iteration,
          score: aiScore,
          text: currentText,
          feedback: rewriteFeedback,
          sentenceCount: doc.sentences.length,
          flaggedSentenceCount: allAISentences.length,
        });

        if (allAISentences.length === 0) {
          await send(writer, {
            type: 'log',
            level: 'warn',
            message: 'Нет конкретных AI-предложений, переписываю весь текст целиком...',
          });
        } else {
          const diagnosticSummary = rewriteFeedback.documentNotes.slice(0, 3).join(' · ');
          await send(writer, {
            type: 'log',
            level: 'warn',
            message: manualTargets.length > 0
              ? `GPTZero отметил ${allAISentences.length} AI-предложений; ручных impact целей найдено в тексте: ${manualTargets.length}/${manualImpactTargets.length}. В точечную работу беру ручной список. Диагностика: ${diagnosticSummary}`
              : impactMetadataAvailable
              ? `GPTZero отметил ${allAISentences.length} AI-предложений; high-impact: ${highImpactSentences.length}. В точечную работу беру все high-impact. Диагностика: ${diagnosticSummary}`
              : `GPTZero отметил ${allAISentences.length} AI-предложений, но API не отдал impact tiers (1/3-3/3). В точечную работу беру все API-highlighted предложения. Диагностика: ${diagnosticSummary}`,
          });

          if (manualImpactTargets.length > 0 && manualTargets.length < manualImpactTargets.length) {
            await send(writer, {
              type: 'log',
              level: 'warn',
              message: `Не совпало с текущим текстом: ${
                manualImpactTargets.length - manualTargets.length
              } ручных impact строк.`,
            });
          }
          await send(writer, {
            type: 'highlights',
            sentences: rewriteFeedback.sentences.map((s) => ({
              text: s.text,
              prob: s.aiProbability,
              reason: s.notes.join('; '),
            })),
          });
        }

        const surgicalCandidates = await generateSurgicalCandidates({
          currentText,
          model,
          attempts: failedAttempts,
          aiSentences: focusSentences,
          rewriteFeedback,
          writer,
        });
        const evaluations: CandidateEvaluation[] = [];
        let chosen: CandidateEvaluation | null = null;

        if (surgicalCandidates.length > 0) {
          await send(writer, {
            type: 'log',
            level: 'info',
            message: `Получено точечных вариантов: ${surgicalCandidates.length}. Проверяю через GPTZero по очереди и останавливаюсь на первом проходном...`,
          });

          const surgicalEvaluations = await evaluateCandidates(surgicalCandidates, writer, {
            stopOnPass: true,
          });
          evaluations.push(...surgicalEvaluations);
          await logEvaluations(surgicalEvaluations, writer);
          chosen =
            surgicalEvaluations.find((evaluation) => isPassingDocument(evaluation.doc)) ?? null;
        }

        if (!chosen) {
          await send(writer, {
            type: 'log',
            level: 'info',
            message: `Точечные варианты не прошли. Отправляю в ${model} весь контекст и прошу несколько разных full rewrite кандидатов...`,
          });

          try {
            const candidates = await generateRewriteCandidates({
              originalText,
              currentText,
              currentFeedback: rewriteFeedback,
              model,
              attempts: failedAttempts,
              sourceSentenceCount: doc.sentences.length,
            });

            const fullCandidates = candidates.map((candidate, index) => ({
              label: `Полный ${index + 1}`,
              text: candidate,
            }));

            await send(writer, {
              type: 'log',
              level: 'info',
              message: `Получено full rewrite вариантов: ${fullCandidates.length}. Проверяю через GPTZero по очереди и останавливаюсь на первом проходном...`,
            });

            const fullEvaluations = await evaluateCandidates(fullCandidates, writer, {
              stopOnPass: true,
            });
            evaluations.push(...fullEvaluations);
            await logEvaluations(fullEvaluations, writer);
            chosen =
              fullEvaluations.find((evaluation) => isPassingDocument(evaluation.doc)) ??
              null;
          } catch (e) {
            await send(writer, {
              type: 'log',
              level: evaluations.length > 0 ? 'warn' : 'error',
              message: `Full rewrite AI error: ${formatUnknownError(e)}${
                evaluations.length > 0
                  ? '. Продолжаю с уже проверенными точечными кандидатами.'
                  : ''
              }`,
            });

            if (evaluations.length === 0) {
              break;
            }
          }
        }

        if (evaluations.length === 0) {
          await send(writer, {
            type: 'log',
            level: 'error',
            message: 'Не удалось проверить ни одного кандидата через GPTZero.',
          });
          break;
        }

        for (const evaluation of evaluations) {
          if (evaluation.score < bestScore) {
            bestScore = evaluation.score;
            bestText = evaluation.text;
          }
        }

        const selectedCandidate = chosen ?? chooseBestCandidate(evaluations);
        currentText = selectedCandidate.text.trim();

        if (isPassingDocument(selectedCandidate.doc)) {
          await send(writer, {
            type: 'log',
            level: 'success',
            message: `✅ Один из кандидатов прошёл проверку: ${Math.round(
              selectedCandidate.score * 100
            )}% AI`,
          });
          await send(writer, { type: 'complete', text: currentText });
          completed = true;
          break;
        }

        await send(writer, {
          type: 'log',
          level: 'info',
          message: `Выбран вариант "${selectedCandidate.label}"; показываю полный черновик и запускаю следующую итерацию от него...`,
        });
        await send(writer, {
          type: 'draft',
          text: currentText,
          message: `Черновик после итерации ${iteration}`,
        });
      }

      if (!completed && iteration >= MAX_ITERATIONS) {
        await send(writer, {
          type: 'log',
          level: 'warn',
          message: `⚠️ Достигнут лимит итераций (${MAX_ITERATIONS}). Отдаю лучший вариант: ${Math.round(bestScore * 100)}% AI.`,
        });
        await send(writer, { type: 'complete', text: bestText });
      }
    } catch (e) {
      await send(writer, {
        type: 'log',
        level: 'error',
        message: `Критическая ошибка: ${formatUnknownError(e)}`,
      });
    } finally {
      writer.close();
    }
  })();

  return new Response(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  });
}

function maskSecret(value: string | undefined): string {
  if (!value) return '<missing>';
  return `${value.slice(0, 7)}...${value.slice(-4)} len=${value.length}`;
}

async function runInitialRewriteFlow({
  currentText,
  model,
  useDeepRewrite,
  useWritingTeacher,
  styleProfile,
  writer,
}: {
  currentText: string;
  model: AIModel;
  useDeepRewrite: boolean;
  useWritingTeacher: boolean;
  styleProfile: string;
  writer: WritableStreamDefaultWriter<Uint8Array>;
}) {
  await send(writer, {
    type: 'log',
    level: 'info',
    message: useDeepRewrite
      ? 'Этап 1: делаю deep rewrite без GPTZero-петли.'
      : 'Этап 1: делаю один полный human rewrite без GPTZero-петли.',
  });

  const rewritten = useDeepRewrite
    ? await runDeepRewritePipeline({ currentText, model, styleProfile, useWritingTeacher, writer })
    : await generateInitialRewrite({ text: currentText, model, styleProfile });
  await send(writer, {
    type: 'draft',
    text: rewritten,
    message: 'Первичный human rewrite',
  });

  await send(writer, {
    type: 'log',
    level: 'check',
    message: 'Проверяю первичный rewrite через GPTZero один раз...',
  });

  const evaluation = await evaluateTextWithGPTZero(rewritten, 1, writer);

  if (isPassingDocument(evaluation.doc)) {
    await send(writer, {
      type: 'log',
      level: 'success',
      message: `✅ Первичный rewrite прошёл: ${Math.round(evaluation.score * 100)}% AI`,
    });
    await send(writer, { type: 'complete', text: rewritten });
    return;
  }

  await send(writer, {
    type: 'log',
    level: 'warn',
    message:
      'Первичный rewrite не прошёл. Отдаю черновик; скопируй high/medium/low impact из GPTZero UI в поле слева и запусти ручной точечный ремонт.',
  });
  await sendImpactFallbackDetails(evaluation.doc, writer);
}

async function runDeepRewritePipeline({
  currentText,
  model,
  styleProfile,
  useWritingTeacher,
  writer,
}: {
  currentText: string;
  model: AIModel;
  styleProfile: string;
  useWritingTeacher: boolean;
  writer: WritableStreamDefaultWriter<Uint8Array>;
}): Promise<string> {
  await send(writer, {
    type: 'log',
    level: 'info',
    message: 'Анализ смысла: вытаскиваю требования, edge cases и технические токены.',
  });
  const analysis = await analyzeRewriteFacts({ text: currentText, model, styleProfile });

  const teacherPlan = useWritingTeacher
    ? await buildTeacherPlan({
        text: currentText,
        model,
        styleProfile,
        semanticAnalysis: analysis,
        writer,
      })
    : undefined;

  await send(writer, {
    type: 'log',
    level: 'info',
    message: 'Новый черновик: пересобираю текст из смысла, а не из исходных предложений.',
  });
  const draft = await generateDeepRewrite({
    text: currentText,
    model,
    styleProfile,
    analysis,
    teacherPlan,
  });

  await send(writer, {
    type: 'draft',
    text: draft,
    message: 'Новый черновик после semantic rebuild',
  });

  await send(writer, {
    type: 'log',
    level: 'info',
    message: 'Чистка AI-ритма: убираю spec cadence и шаблонные фразы.',
  });
  const cleaned = await cleanupAiCadence({
    originalText: currentText,
    draft,
    model,
    styleProfile,
    teacherPlan,
  });

  await send(writer, {
    type: 'log',
    level: 'info',
    message: 'Проверка смысла: сверяю, что требования не потерялись.',
  });
  const coverage = await verifySemanticCoverage({
    originalText: currentText,
    rewrittenText: cleaned,
    model,
  });

  if (coverage.ok || coverage.issues.length === 0) {
    return cleaned;
  }

  await send(writer, {
    type: 'log',
    level: 'warn',
    message: `Проверка смысла нашла пропуски: ${coverage.issues.length}. Дочиняю текст точечно.`,
  });

  return repairSemanticCoverage({
    originalText: currentText,
    rewrittenText: cleaned,
    coverageIssues: coverage.issues,
    model,
    styleProfile,
  });
}

async function buildTeacherPlan({
  text,
  model,
  styleProfile,
  semanticAnalysis,
  targets,
  stageLabel,
  attemptIndex,
  writer,
}: {
  text: string;
  model: AIModel;
  styleProfile: string;
  semanticAnalysis?: string;
  targets?: ImpactTarget[];
  stageLabel?: string;
  attemptIndex?: number;
  writer: WritableStreamDefaultWriter<Uint8Array>;
}): Promise<string> {
  await send(writer, {
    type: 'log',
    level: 'info',
    message: 'Учитель письма: строю редакторский план.',
  });

  const teacherPlan = await planRewriteWithTeacher({
    text,
    model,
    styleProfile,
    semanticAnalysis,
    targets,
    stageLabel,
    attemptIndex,
  });

  await send(writer, {
    type: 'log',
    level: 'info',
    message: `Учитель письма: план готов (${teacherPlan.length} симв.).`,
  });

  return teacherPlan;
}

async function runScanOnlyFlow({
  currentText,
  writer,
}: {
  currentText: string;
  writer: WritableStreamDefaultWriter<Uint8Array>;
}) {
  await send(writer, {
    type: 'log',
    level: 'info',
    message: 'Первичный rewrite выключен. Проверяю текущий текст без переписывания.',
  });

  const evaluation = await evaluateTextWithGPTZero(currentText, 1, writer);

  if (isPassingDocument(evaluation.doc)) {
    await send(writer, {
      type: 'log',
      level: 'success',
      message: `✅ Текущий текст уже проходит: ${Math.round(evaluation.score * 100)}% AI`,
    });
    await send(writer, { type: 'complete', text: currentText });
    return;
  }

  await send(writer, {
    type: 'draft',
    text: currentText,
    message: 'Текущий текст для ручного impact repair',
  });
  await send(writer, {
    type: 'log',
    level: 'warn',
    message:
      'Текст не прошёл. Вставь high/medium/low impact из GPTZero UI и запусти ручной точечный ремонт.',
  });
  await sendImpactFallbackDetails(evaluation.doc, writer);
}

async function runNoGptZeroFlow({
  currentText,
  model,
  manualTargets,
  useInitialRewrite,
  useDeepRewrite,
  useWritingTeacher,
  styleProfile,
  fixAllAtOnce,
  writer,
}: {
  currentText: string;
  model: AIModel;
  manualTargets: ImpactTarget[];
  useInitialRewrite: boolean;
  useDeepRewrite: boolean;
  useWritingTeacher: boolean;
  styleProfile: string;
  fixAllAtOnce: boolean;
  writer: WritableStreamDefaultWriter<Uint8Array>;
}) {
  await send(writer, {
    type: 'log',
    level: 'info',
    message: 'GPTZero API выключен: делаю rewrite без проверки и без списания GPTZero-кредитов.',
  });

  if (manualTargets.length > 0) {
    const stages: Array<{ label: string; targets: ImpactTarget[] }> = fixAllAtOnce
      ? [{ label: 'All impact', targets: manualTargets }]
      : [
          { label: 'High', targets: manualTargets.filter((target) => target.level === 'high') },
          {
            label: 'Medium',
            targets: manualTargets.filter((target) => target.level === 'medium'),
          },
          { label: 'Low', targets: manualTargets.filter((target) => target.level === 'low') },
        ];

    let rewritten = currentText;
    for (const stage of stages) {
      if (stage.targets.length === 0) continue;

      await send(writer, {
        type: 'log',
        level: 'info',
        message: `Без GPTZero: перерабатываю контекст вокруг ${stage.label} симптомов (${stage.targets.length}).`,
      });

      const teacherPlan = useWritingTeacher
        ? await buildTeacherPlan({
            text: rewritten,
            model,
            styleProfile,
            targets: stage.targets,
            stageLabel: stage.label,
            attemptIndex: 1,
            writer,
          })
        : undefined;

      rewritten = await generateTargetedRewrite({
        currentText: rewritten,
        targets: stage.targets,
        model,
        stageLabel: stage.label,
        attemptIndex: 1,
        styleProfile,
        teacherPlan,
      });

      await send(writer, {
        type: 'draft',
        text: rewritten,
        message: `${stage.label} rewrite без GPTZero`,
      });
    }

    await send(writer, { type: 'complete', text: rewritten });
    return;
  }

  if (!useInitialRewrite && !useDeepRewrite) {
    await send(writer, {
      type: 'log',
      level: 'warn',
      message: 'Первичный rewrite и GPTZero выключены, поэтому возвращаю исходный текст без изменений.',
    });
    await send(writer, { type: 'complete', text: currentText });
    return;
  }

  if (!useInitialRewrite && useDeepRewrite) {
    await send(writer, {
      type: 'log',
      level: 'info',
      message: 'Первичный rewrite выключен, но Deep rewrite включён: запускаю deep rewrite без GPTZero.',
    });
  }

  const rewritten = useDeepRewrite
    ? await runDeepRewritePipeline({ currentText, model, styleProfile, useWritingTeacher, writer })
    : await generateInitialRewrite({ text: currentText, model, styleProfile });
  await send(writer, {
    type: 'draft',
    text: rewritten,
    message: useDeepRewrite ? 'Deep human rewrite без GPTZero' : 'Human rewrite без GPTZero',
  });
  await send(writer, { type: 'complete', text: rewritten });
}

async function runImpactRepairFlow({
  currentText,
  model,
  manualTargets,
  attemptCount,
  fixAllAtOnce,
  useDeepRewrite,
  useWritingTeacher,
  styleProfile,
  writer,
}: {
  currentText: string;
  model: AIModel;
  manualTargets: ImpactTarget[];
  attemptCount: number;
  fixAllAtOnce: boolean;
  useDeepRewrite: boolean;
  useWritingTeacher: boolean;
  styleProfile: string;
  writer: WritableStreamDefaultWriter<Uint8Array>;
}) {
  const levelCounts = countImpactLevels(manualTargets);
  await send(writer, {
    type: 'log',
    level: 'info',
    message: `Этап 2: контекстная переработка по impact-симптомам. High: ${levelCounts.high}, Medium: ${levelCounts.medium}, Low: ${levelCounts.low}. Попыток: ${attemptCount}. Режим: ${
      fixAllAtOnce ? 'все impact сразу' : 'по уровням High → Medium → Low'
    }.`,
  });
  await send(writer, {
    type: 'highlights',
    sentences: manualTargets.map((target) => ({
      text: target.text,
      prob: impactLevelToProbability(target.level),
      impact: target.level,
      reason: `Manual GPTZero impact: ${target.level}`,
    })),
  });

  let workingText = currentText;
  if (useDeepRewrite) {
    await send(writer, {
      type: 'log',
      level: 'info',
      message: 'Deep rewrite включён: сначала пересобираю весь текст, потом добиваю impact-симптомы.',
    });
    workingText = await runDeepRewritePipeline({
      currentText,
      model,
      styleProfile,
      useWritingTeacher,
      writer,
    });
  }

  let bestOverallText = workingText;
  let bestOverallEvaluation: CandidateEvaluation | null = null;
  const stages: Array<{ level: ImpactLevel | 'all'; label: string; targets: ImpactTarget[] }> =
    fixAllAtOnce
      ? [{ level: 'all', label: 'All impact', targets: manualTargets }]
      : [
          {
            level: 'high',
            label: 'High',
            targets: manualTargets.filter((target) => target.level === 'high'),
          },
          {
            level: 'medium',
            label: 'Medium',
            targets: manualTargets.filter((target) => target.level === 'medium'),
          },
          {
            level: 'low',
            label: 'Low',
            targets: manualTargets.filter((target) => target.level === 'low'),
          },
        ];

  for (let stageIndex = 0; stageIndex < stages.length; stageIndex++) {
    const stage = stages[stageIndex];
    const stageTargets = stage.targets;
    if (stageTargets.length === 0) continue;

    await send(writer, {
      type: 'log',
      level: 'info',
      message: `Перерабатываю контекст вокруг ${stage.label} симптомов: ${stageTargets.length}. Проверяю до ${attemptCount} вариантов по очереди.`,
    });

    let bestStageText = workingText;
    let bestStageEvaluation: CandidateEvaluation | null = null;
    let adaptiveTargets = stageTargets;
    let attemptBaseText = workingText;

    for (let attempt = 1; attempt <= attemptCount; attempt++) {
      const teacherPlan = useWritingTeacher
        ? await buildTeacherPlan({
            text: attemptBaseText,
            model,
            styleProfile,
            targets: adaptiveTargets,
            stageLabel: stage.label,
            attemptIndex: attempt,
            writer,
          })
        : undefined;
      const candidateRequest = {
        currentText: attemptBaseText,
        targets: adaptiveTargets,
        model,
        stageLabel: stage.label,
        attemptIndex: attempt,
        styleProfile,
        teacherPlan,
      };
      const targetedCandidates = await generateTargetedRewriteCandidates(candidateRequest);
      let localChoice = chooseLowestLocalRiskCandidate(targetedCandidates);

      if (targetedCandidates.length > 1) {
        await send(writer, {
          type: 'log',
          level: 'info',
          message: `Локальный smell-фильтр выбрал вариант ${localChoice.index + 1}/${
            targetedCandidates.length
          } перед GPTZero.`,
        });
      }

      const cleanedCandidate = await cleanupAiCadence({
        originalText: currentText,
        draft: localChoice.text,
        model,
        styleProfile,
        teacherPlan,
      });
      const cleanedChoice = chooseLowestLocalRiskCandidate([localChoice.text, cleanedCandidate]);
      if (cleanedChoice.index === 1) {
        localChoice = { ...cleanedChoice, index: localChoice.index };
        await send(writer, {
          type: 'log',
          level: 'info',
          message: 'Локальная чистка слов/ритма улучшила кандидат перед GPTZero.',
        });
      }

      const attemptText = localChoice.text;

      await send(writer, {
        type: 'draft',
        text: attemptText,
        message: `${stage.label} · попытка ${attempt}/${attemptCount}`,
      });

      await send(writer, {
        type: 'log',
        level: 'check',
        message: `Проверяю ${stage.label} попытку ${attempt}/${attemptCount} через GPTZero...`,
      });

      const evaluation = await evaluateTextWithGPTZero(
        attemptText,
        stageIndex * attemptCount + attempt,
        writer
      );

      const isStageImprovement = isBetterEvaluation(evaluation, bestStageEvaluation);
      if (isStageImprovement) {
        bestStageEvaluation = evaluation;
        bestStageText = attemptText;
      }

      if (isBetterEvaluation(evaluation, bestOverallEvaluation)) {
        bestOverallEvaluation = evaluation;
        bestOverallText = attemptText;
      }

      if (isPassingDocument(evaluation.doc)) {
        await send(writer, {
          type: 'log',
          level: 'success',
          message: `✅ Прошло после ${stage.label}, попытка ${attempt}: ${Math.round(
            evaluation.score * 100
          )}% AI`,
        });
        await send(writer, { type: 'complete', text: attemptText });
        return;
      }

      const expandedTargets = expandImpactTargetsFromEvaluation(adaptiveTargets, evaluation);
      if (expandedTargets.length > adaptiveTargets.length) {
        await send(writer, {
          type: 'log',
          level: 'info',
          message: `GPTZero сдвинул маркеры: добавляю ${
            expandedTargets.length - adaptiveTargets.length
          } новых симптомов в следующую попытку.`,
        });
        adaptiveTargets = expandedTargets;
      }

      if (isStageImprovement) {
        attemptBaseText = attemptText;
      }
    }

    workingText = bestStageText;
    await send(writer, {
      type: 'log',
      level: 'warn',
      message: `${stage.label} не прошёл за ${attemptCount} попыток. Беру лучший вариант уровня и иду дальше.`,
    });
  }

  await send(writer, {
    type: 'log',
    level: 'warn',
    message: `High → Medium → Low цепочка не пробила GPTZero. Отдаю лучший вариант за весь прогон: ${
      bestOverallEvaluation ? `${Math.round(bestOverallEvaluation.score * 100)}% AI` : 'без оценки'
    }. Можно снова прогнать его через GPTZero и вставить новый impact-список.`,
  });
  await send(writer, {
    type: 'draft',
    text: bestOverallText,
    message: 'Лучший черновик после impact-цепочки',
  });
  await send(writer, { type: 'complete', text: bestOverallText });
}

async function evaluateTextWithGPTZero(
  text: string,
  iteration: number,
  writer: WritableStreamDefaultWriter<Uint8Array>
): Promise<CandidateEvaluation> {
  const result = await checkWithGPTZero(text);
  const doc = result.documents[0];
  const flagged = getAllAISentences(doc);
  const metrics = getSelectionMetrics(text, doc, flagged.length);

  await send(writer, {
    type: 'score',
    score: doc.completely_generated_prob,
    iteration,
    message: `GPTZero: ${Math.round(doc.completely_generated_prob * 100)}% AI-контент`,
    level:
      doc.completely_generated_prob < 0.5
        ? 'success'
        : doc.completely_generated_prob < 0.75
          ? 'warn'
          : 'error',
  });

  return {
    index: 0,
    label: 'Текущий текст',
    text,
    score: doc.completely_generated_prob,
    ...metrics,
    sentenceCount: doc.sentences.length,
    flaggedSentenceCount: flagged.length,
    doc,
  };
}

async function sendImpactFallbackDetails(
  doc: GPTZeroDocument,
  writer: WritableStreamDefaultWriter<Uint8Array>
) {
  const flagged = getAllAISentences(doc);
  if (flagged.length === 0) return;

  await send(writer, {
    type: 'log',
    level: 'warn',
    message: `GPTZero API отметил ${flagged.length} AI-предложений. Impact tiers лучше брать из веб-интерфейса GPTZero.`,
  });
}

function countImpactLevels(targets: ImpactTarget[]) {
  return targets.reduce(
    (counts, target) => {
      counts[target.level]++;
      return counts;
    },
    { high: 0, medium: 0, low: 0 }
  );
}

function impactLevelToProbability(level: ImpactLevel): number {
  if (level === 'high') return 1;
  if (level === 'medium') return 0.66;
  return 0.33;
}

function expandImpactTargetsFromEvaluation(
  currentTargets: ImpactTarget[],
  evaluation: CandidateEvaluation,
  limit = 5
): ImpactTarget[] {
  const seen = new Set(currentTargets.map((target) => normalizeSentenceText(target.text)));
  const expanded = [...currentTargets];

  for (const sentence of getAllAISentences(evaluation.doc)) {
    const text = sentence.sentence.trim();
    const key = normalizeSentenceText(text);

    if (!text || text.length < 20 || seen.has(key)) continue;

    seen.add(key);
    expanded.push({
      text,
      level: sentence.generated_prob >= 0.85 ? 'high' : 'medium',
    });

    if (expanded.length - currentTargets.length >= limit) break;
  }

  return expanded;
}

async function generateSurgicalCandidates({
  currentText,
  model,
  attempts,
  aiSentences,
  rewriteFeedback,
  writer,
}: {
  currentText: string;
  model: AIModel;
  attempts: RewriteAttempt[];
  aiSentences: GPTZeroSentence[];
  rewriteFeedback: ReturnType<typeof buildRewriteFeedback>;
  writer: WritableStreamDefaultWriter<Uint8Array>;
}): Promise<CandidateInput[]> {
  const targets = getSurgicalTargets(aiSentences, currentText, aiSentences.length);
  if (targets.length === 0) return [];

  await send(writer, {
    type: 'log',
    level: 'info',
    message: `Пробую точечные замены high-impact предложений: ${targets.length}`,
  });

  const notesBySentence = new Map(
    rewriteFeedback.sentences.map((sentence) => [sentence.text, sentence.notes])
  );
  const candidates: CandidateInput[] = [];

  for (let targetIndex = 0; targetIndex < targets.length; targetIndex++) {
    const target = targets[targetIndex];
    const notes = notesBySentence.get(target.sentence) ?? getInlineSentenceNotes(target);

    try {
      const replacements = await generateSentenceReplacements({
        fullText: currentText,
        targetSentence: target.sentence,
        targetNotes: notes,
        model,
        attempts,
      });

      for (let replacementIndex = 0; replacementIndex < replacements.length; replacementIndex++) {
        const replaced = replaceFirst(currentText, target.sentence, replacements[replacementIndex]);
        if (!replaced || replaced === currentText) continue;

        candidates.push({
          label: `Точечная ${targetIndex + 1}.${replacementIndex + 1}`,
          text: replaced,
        });
      }
    } catch (e) {
      await send(writer, {
        type: 'log',
        level: 'error',
        message: `Точечные замены для предложения ${targetIndex + 1} не получились: ${
          formatUnknownError(e)
        }`,
      });
    }
  }

  return candidates;
}

function getSurgicalTargets(
  aiSentences: GPTZeroSentence[],
  currentText: string,
  limit: number
): GPTZeroSentence[] {
  return [...aiSentences]
    .filter((sentence) => currentText.includes(sentence.sentence))
    .sort((a, b) => getSentenceImpactScore(b) - getSentenceImpactScore(a))
    .slice(0, limit);
}

function getSentenceImpactScore(sentence: GPTZeroSentence): number {
  const text = sentence.sentence;
  const length = text.length;
  const mappingSignal =
    /\b(use|uses|using|match|matches|matching|via|through)\b/i.test(text) &&
    /\b(uid|userId|app_user_id|id|identifier|field|fields)\b/i.test(text)
      ? 0.35
      : 0;
  const mediumLengthSignal = length >= 120 && length <= 260 ? 0.12 : 0;
  const tooLongPenalty = length > 360 ? -0.12 : 0;
  const interpretability =
    typeof sentence.interpretability_normalized_value === 'number'
      ? sentence.interpretability_normalized_value
      : typeof sentence.interpretability_value === 'number'
        ? sentence.interpretability_value
        : 0;

  return sentence.generated_prob + interpretability * 2 + mappingSignal + mediumLengthSignal + tooLongPenalty;
}

function getInlineSentenceNotes(sentence: GPTZeroSentence): string[] {
  const notes = [`AI probability: ${Math.round(sentence.generated_prob * 100)}%`];
  const classes = sentence.class_probabilities;
  if (classes) {
    notes.push(
      `Class probabilities: human ${formatNullablePercent(classes.human ?? null)}, ai ${formatNullablePercent(
        classes.ai ?? null
      )}, paraphrased ${formatNullablePercent(classes.paraphrased ?? null)}`
    );
  }
  if (typeof sentence.perplexity === 'number') {
    notes.push(`Perplexity: ${sentence.perplexity.toFixed(2)}`);
  }
  if (sentence.highlight_sentence_for_ai) {
    notes.push('Highlighted as contributing to the AI classification');
  }
  return notes;
}

function replaceFirst(text: string, search: string, replacement: string): string | null {
  const index = text.indexOf(search);
  if (index === -1) return null;
  return `${text.slice(0, index)}${replacement}${text.slice(index + search.length)}`;
}

async function evaluateCandidates(
  candidates: CandidateInput[],
  writer: WritableStreamDefaultWriter<Uint8Array>,
  options: EvaluateCandidatesOptions = {}
): Promise<CandidateEvaluation[]> {
  const evaluations: CandidateEvaluation[] = [];

  for (let index = 0; index < candidates.length; index++) {
    try {
      const evaluation = await evaluateCandidateWithRetry(candidates[index], index);
      evaluations.push(evaluation);

      if (options.stopOnPass && isPassingDocument(evaluation.doc)) {
        break;
      }
    } catch (e) {
      await send(writer, {
        type: 'log',
        level: 'error',
        message: `${candidates[index].label}: не удалось проверить через GPTZero после retry: ${
            formatUnknownError(e)
        }`,
      });
    }

    if (index < candidates.length - 1) {
      await sleep(350);
    }
  }

  return evaluations;
}

async function logEvaluations(
  evaluations: CandidateEvaluation[],
  writer: WritableStreamDefaultWriter<Uint8Array>
) {
  for (const evaluation of evaluations) {
    const candidatePct = Math.round(evaluation.score * 100);
    const aiClass = formatNullablePercent(evaluation.averageSentenceAiClass);
    const paraphrasedClass = formatNullablePercent(
      evaluation.averageSentenceParaphrasedClass
    );
    await send(writer, {
      type: 'log',
      level: isPassingDocument(evaluation.doc)
        ? 'success'
        : evaluation.compositeScore < 0.75
          ? 'warn'
          : 'error',
      message: `${evaluation.label}: ${candidatePct}% AI · composite ${Math.round(
        evaluation.compositeScore * 100
      )}% · sent-ai ${aiClass} · paraphrased ${paraphrasedClass} · readability +${Math.round(
        evaluation.readabilityPenalty * 100
      )}% · предложений ${
        evaluation.sentenceCount
      } · flagged ${evaluation.flaggedSentenceCount}`,
    });
  }
}

async function evaluateCandidateWithRetry(
  candidate: CandidateInput,
  index: number
): Promise<CandidateEvaluation> {
  let lastError: unknown;

  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      return await evaluateCandidate(candidate, index);
    } catch (e) {
      lastError = e;
      if (attempt < 3) {
        await sleep(800 * attempt);
      }
    }
  }

  throw lastError;
}

async function evaluateCandidate(
  candidate: CandidateInput,
  index: number
): Promise<CandidateEvaluation> {
  const result = await checkWithGPTZero(candidate.text);
  const doc = result.documents[0];
  const flagged = getAllAISentences(doc);
  const metrics = getSelectionMetrics(candidate.text, doc, flagged.length);

  return {
    index,
    label: candidate.label,
    text: candidate.text,
    score: doc.completely_generated_prob,
    ...metrics,
    sentenceCount: doc.sentences.length,
    flaggedSentenceCount: flagged.length,
    doc,
  };
}

function chooseBestCandidate(evaluations: CandidateEvaluation[]): CandidateEvaluation {
  if (evaluations.length === 0) {
    throw new Error('AI did not return any rewrite candidates');
  }

  return [...evaluations].sort((a, b) => {
    const compositeDelta = a.compositeScore - b.compositeScore;
    if (Math.abs(compositeDelta) > 0.005) return compositeDelta;

    const scoreDelta = a.score - b.score;
    if (Math.abs(scoreDelta) > 0.01) return scoreDelta;

    const flaggedDelta = a.flaggedSentenceCount - b.flaggedSentenceCount;
    if (flaggedDelta !== 0) return flaggedDelta;

    return a.sentenceCount - b.sentenceCount;
  })[0];
}

function chooseLowestLocalRiskCandidate(candidates: string[]): LocalCandidateChoice {
  const cleanCandidates = candidates.map((text) => text.trim()).filter(Boolean);
  if (cleanCandidates.length === 0) {
    throw new Error('AI did not return any rewrite candidates');
  }

  return cleanCandidates
    .map((text, index) => ({
      text,
      index,
      risk: getLocalSpecRiskScore(text),
    }))
    .sort((a, b) => {
      const riskDelta = a.risk - b.risk;
      if (Math.abs(riskDelta) > 0.05) return riskDelta;

      return a.text.length - b.text.length;
    })[0];
}

function getLocalSpecRiskScore(text: string): number {
  const lower = text.toLowerCase();
  const phraseRisk = [
    'we need',
    'we should',
    'ensure',
    'currently',
    'properly',
    'correctly',
    'accurately',
    'essential',
    'consistently',
    'inconsistency',
    'quality of',
    'selected technique',
    'chosen technique',
    'validated emotional state',
    'server-chosen',
    'server-generated',
    'normal failure response',
    'mapping is fixed',
    'is limited to',
    'anything unusual',
    'returned plans and saved plans',
    'no saved data',
    'no saved plan',
    'must be included',
    'should consistently',
    'adhere',
    'the bug in',
    'the issue is',
    'the problem is',
  ].reduce((score, phrase) => score + countOccurrences(lower, phrase) * 1.4, 0);

  const sentences = splitLocalSentences(text);
  const sentenceRisk = sentences.reduce((score, sentence) => {
    const words = sentence.split(/\s+/).filter(Boolean).length;
    const commaCount = countMatches(sentence, /,/g);
    const semicolonCount = countMatches(sentence, /;/g);
    const quotedTokenCount = countMatches(sentence, /"[^"]+"/g);
    const slashTokenCount = countMatches(sentence, /\b[\w-]+\/[\w/{.-]+/g);
    const modalCount = countMatches(sentence, /\b(?:should|must|needs? to|has to|have to)\b/gi);

    return (
      score +
      Math.max(0, words - 42) * 0.08 +
      Math.max(0, words - 68) * 0.12 +
      Math.max(0, commaCount - 4) * 0.35 +
      Math.max(0, semicolonCount - 1) * 0.45 +
      Math.max(0, quotedTokenCount - 5) * 0.45 +
      Math.max(0, slashTokenCount - 2) * 0.35 +
      Math.max(0, modalCount - 1) * 0.7
    );
  }, 0);

  const repeatedStartRisk = getRepeatedSentenceStartRisk(sentences);
  const flatParagraphRisk = text.split(/\n{2,}/).filter((paragraph) => paragraph.trim()).length <= 1 ? 1 : 0;

  return phraseRisk + sentenceRisk + repeatedStartRisk + flatParagraphRisk;
}

function splitLocalSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+|\n+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}

function getRepeatedSentenceStartRisk(sentences: string[]): number {
  const seen = new Map<string, number>();

  for (const sentence of sentences) {
    const key = sentence
      .toLowerCase()
      .replace(/^["'\s]+/, '')
      .split(/\s+/)
      .slice(0, 3)
      .join(' ');
    if (!key) continue;
    seen.set(key, (seen.get(key) ?? 0) + 1);
  }

  return [...seen.values()].reduce((score, count) => score + Math.max(0, count - 1) * 1.2, 0);
}

function countOccurrences(value: string, phrase: string): number {
  let count = 0;
  let index = value.indexOf(phrase);

  while (index !== -1) {
    count++;
    index = value.indexOf(phrase, index + phrase.length);
  }

  return count;
}

function countMatches(value: string, pattern: RegExp): number {
  return value.match(pattern)?.length ?? 0;
}

function isBetterEvaluation(
  candidate: CandidateEvaluation,
  current: CandidateEvaluation | null
): boolean {
  if (!current) return true;

  const scoreDelta = candidate.score - current.score;
  if (Math.abs(scoreDelta) > 0.005) return scoreDelta < 0;

  const compositeDelta = candidate.compositeScore - current.compositeScore;
  if (Math.abs(compositeDelta) > 0.005) return compositeDelta < 0;

  const flaggedDelta = candidate.flaggedSentenceCount - current.flaggedSentenceCount;
  if (flaggedDelta !== 0) return flaggedDelta < 0;

  return candidate.sentenceCount < current.sentenceCount;
}

function isPassingDocument(doc: GPTZeroDocument): boolean {
  const predictedClass = doc.predicted_class?.toLowerCase() ?? '';
  const documentClass = doc.document_classification?.toLowerCase() ?? '';
  const confidence = doc.confidence_category?.toLowerCase() ?? '';

  if (predictedClass === 'human' || predictedClass === 'mixed') return true;
  if (documentClass === 'human_only' || documentClass === 'mixed') return true;
  if (confidence === 'medium' || confidence === 'moderate' || confidence === 'low') return true;

  if (isHuman(doc, PASS_THRESHOLD)) return true;

  const documentAiProbability = doc.class_probabilities?.ai;
  return (
    typeof documentAiProbability === 'number' &&
    documentAiProbability < PASS_THRESHOLD &&
    confidence !== 'high'
  );
}

function getSelectionMetrics(
  text: string,
  doc: GPTZeroDocument,
  flaggedSentenceCount: number
) {
  const documentAiProbability = doc.class_probabilities?.ai ?? null;
  const documentHumanProbability = doc.class_probabilities?.human ?? null;
  const averageSentenceAiClass = averageSentenceClass(doc, 'ai');
  const averageSentenceParaphrasedClass = averageSentenceClass(doc, 'paraphrased');
  const sentenceCount = Math.max(doc.sentences.length, 1);
  const flaggedRatio = flaggedSentenceCount / sentenceCount;
  const readability = getReadabilityMetrics(text, doc);

  const aiScore = doc.completely_generated_prob;
  const docAi = documentAiProbability ?? aiScore;
  const docHuman = documentHumanProbability ?? 0;
  const sentenceAi = averageSentenceAiClass ?? aiScore;
  const paraphrased = averageSentenceParaphrasedClass ?? 0;

  const compositeScore = clamp01(
    aiScore * 0.4 +
      docAi * 0.2 +
      sentenceAi * 0.22 +
      flaggedRatio * 0.12 +
      readability.penalty -
      paraphrased * 0.14 -
      docHuman * 0.12
  );

  return {
    compositeScore,
    documentAiProbability,
    documentHumanProbability,
    averageSentenceAiClass,
    averageSentenceParaphrasedClass,
    readabilityPenalty: readability.penalty,
    averageWordsPerSentence: readability.averageWordsPerSentence,
    maxWordsPerSentence: readability.maxWordsPerSentence,
  };
}

function getReadabilityMetrics(text: string, doc: GPTZeroDocument) {
  const fallbackSentences = text
    .split(/(?<=[.!?])\s+|\n+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
  const sentenceTexts =
    doc.sentences.length > 0 ? doc.sentences.map((sentence) => sentence.sentence) : fallbackSentences;
  const counts = sentenceTexts.map(countWords).filter((count) => count > 0);
  const maxWordsPerSentence = counts.length > 0 ? Math.max(...counts) : countWords(text);
  const averageWordsPerSentence =
    counts.length > 0 ? counts.reduce((sum, count) => sum + count, 0) / counts.length : maxWordsPerSentence;
  const textLength = text.trim().length;
  const terminalPunctuationPenalty = /[.!?]"?$/.test(text.trim()) ? 0 : 0.04;
  const runOnPenalty =
    maxWordsPerSentence > 120
      ? 0.35
      : maxWordsPerSentence > 90
        ? 0.25
        : maxWordsPerSentence > 70
          ? 0.16
          : maxWordsPerSentence > 55
            ? 0.08
            : 0;
  const averagePenalty =
    averageWordsPerSentence > 70
      ? 0.18
      : averageWordsPerSentence > 55
        ? 0.12
        : averageWordsPerSentence > 42
          ? 0.06
          : 0;
  const soupPenalty =
    textLength > 500 && sentenceTexts.length <= 2
      ? 0.22
      : textLength > 350 && sentenceTexts.length <= 1
        ? 0.3
        : 0;

  return {
    averageWordsPerSentence,
    maxWordsPerSentence,
    penalty: clamp01(runOnPenalty + averagePenalty + soupPenalty + terminalPunctuationPenalty),
  };
}

function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function averageSentenceClass(
  doc: GPTZeroDocument,
  key: 'ai' | 'human' | 'mixed' | 'paraphrased'
): number | null {
  const values = doc.sentences
    .map((sentence) => sentence.class_probabilities?.[key])
    .filter((value): value is number => typeof value === 'number');

  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function formatNullablePercent(value: number | null): string {
  return typeof value === 'number' ? `${Math.round(value * 100)}%` : 'n/a';
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
