export interface GPTZeroSentence {
  sentence: string;
  generated_prob: number;
  perplexity: number;
  highlight_sentence_for_ai: boolean;
  class_probabilities?: {
    human?: number;
    ai?: number;
    mixed?: number;
    paraphrased?: number;
  };
  interpretability_value?: number | null;
  interpretability_normalized_value?: number | null;
  interpretability_designation?: string | null;
  interpretability_alpha?: number | null;
  special_highlight_type?: string | null;
}

export interface GPTZeroDocument {
  average_generated_prob: number;
  completely_generated_prob: number;
  overall_burstiness: number;
  document_classification?: string;
  predicted_class?: string;
  result_message?: string | null;
  result_sub_message?: string | null;
  confidence_category?: string;
  confidence_score?: number;
  class_probabilities?: {
    human?: number;
    ai?: number;
    mixed?: number;
  };
  subclass?: {
    ai?: {
      predicted_class?: string;
      result_message?: string;
      confidence_score?: number;
      confidence_category?: string;
      class_probabilities?: {
        pure_ai?: number;
        ai_paraphrased?: number;
      };
    };
  };
  writing_stats?: Record<string, unknown>;
  sentences: GPTZeroSentence[];
  paragraphs?: Array<{
    completely_generated_prob: number;
    sentences: GPTZeroSentence[];
  }>;
}

export interface GPTZeroResponse {
  documents: GPTZeroDocument[];
}

export interface GPTZeroRewriteFeedback {
  documentNotes: string[];
  sentences: Array<{
    text: string;
    aiProbability: number;
    notes: string[];
  }>;
}

export async function checkWithGPTZero(text: string): Promise<GPTZeroResponse> {
  const apiKey = process.env.GPTZERO_API_KEY;
  if (!apiKey) throw new Error('GPTZERO_API_KEY не задан в .env.local');

  const response = await fetch('https://api.gptzero.me/v2/predict/text', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
    },
    body: JSON.stringify({ document: text, multilingual: false }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`GPTZero API error ${response.status}: ${err}`);
  }

  return response.json();
}

export function getAISentences(
  doc: GPTZeroDocument,
  threshold = 0.65,
  maxSentences = 3
): GPTZeroSentence[] {
  return getAllAISentences(doc, threshold).slice(0, maxSentences);
}

export function getAllAISentences(
  doc: GPTZeroDocument,
  threshold = 0.65
): GPTZeroSentence[] {
  const candidates = doc.sentences.filter(
    (sentence) => sentence.generated_prob >= threshold || sentence.highlight_sentence_for_ai
  );

  return candidates.sort((a, b) => getSentenceImpactScore(b) - getSentenceImpactScore(a));
}

export function getHighImpactAISentences(
  doc: GPTZeroDocument,
  threshold = 0.65
): GPTZeroSentence[] {
  return getAllAISentences(doc, threshold).filter(isHighImpactSentence);
}

export function hasSentenceImpactMetadata(doc: GPTZeroDocument): boolean {
  return doc.sentences.some(
    (sentence) =>
      Boolean(sentence.interpretability_designation) ||
      Boolean(sentence.special_highlight_type) ||
      typeof sentence.interpretability_normalized_value === 'number' ||
      typeof sentence.interpretability_value === 'number'
  );
}

export function isHuman(doc: GPTZeroDocument, threshold = 0.5): boolean {
  return doc.completely_generated_prob < threshold;
}

export function buildRewriteFeedback(
  doc: GPTZeroDocument,
  selectedSentences = getAISentences(doc),
  maxSentences = 24
): GPTZeroRewriteFeedback {
  const sentencePool =
    selectedSentences.length > 0
      ? selectedSentences
      : [...doc.sentences].sort((a, b) => b.generated_prob - a.generated_prob);

  return {
    documentNotes: getDocumentNotes(doc),
    sentences: sentencePool.slice(0, maxSentences).map((sentence) => ({
      text: sentence.sentence,
      aiProbability: sentence.generated_prob,
      notes: getSentenceNotes(sentence),
    })),
  };
}

function getDocumentNotes(doc: GPTZeroDocument): string[] {
  const notes: string[] = [];

  if (doc.document_classification || doc.predicted_class) {
    notes.push(
      `Document classification: ${doc.document_classification ?? doc.predicted_class}`
    );
  }

  if (doc.class_probabilities) {
    const probs = [
      formatProbability('human', doc.class_probabilities.human),
      formatProbability('mixed', doc.class_probabilities.mixed),
      formatProbability('ai', doc.class_probabilities.ai),
    ].filter(Boolean);
    if (probs.length > 0) notes.push(`Document class probabilities: ${probs.join(', ')}`);
  }

  if (doc.confidence_category) {
    notes.push(
      `Detector confidence: ${doc.confidence_category}${
        typeof doc.confidence_score === 'number' ? ` (${formatPercent(doc.confidence_score)})` : ''
      }`
    );
  }

  if (typeof doc.overall_burstiness === 'number') {
    const label =
      doc.overall_burstiness <= 0.15
        ? 'very flat rhythm'
        : doc.overall_burstiness <= 0.35
          ? 'low sentence variation'
          : 'sentence variation present';
    notes.push(`Overall burstiness: ${doc.overall_burstiness.toFixed(3)} (${label})`);
  }

  const subclass = doc.subclass?.ai;
  if (subclass?.predicted_class) {
    const probs = [
      formatProbability('pure_ai', subclass.class_probabilities?.pure_ai),
      formatProbability('ai_paraphrased', subclass.class_probabilities?.ai_paraphrased),
    ].filter(Boolean);
    notes.push(
      `AI subclass: ${subclass.predicted_class}${
        subclass.confidence_category ? `, ${subclass.confidence_category} confidence` : ''
      }${probs.length > 0 ? ` (${probs.join(', ')})` : ''}`
    );
  }

  if (doc.result_message) notes.push(doc.result_message);
  if (doc.result_sub_message) notes.push(doc.result_sub_message);

  return notes;
}

function getSentenceNotes(sentence: GPTZeroSentence): string[] {
  const notes: string[] = [`AI probability: ${formatPercent(sentence.generated_prob)}`];

  if (sentence.class_probabilities) {
    const probs = [
      formatProbability('human', sentence.class_probabilities.human),
      formatProbability('ai', sentence.class_probabilities.ai),
      formatProbability('paraphrased', sentence.class_probabilities.paraphrased),
    ].filter(Boolean);
    if (probs.length > 0) notes.push(`Class probabilities: ${probs.join(', ')}`);
  }

  if (typeof sentence.perplexity === 'number') {
    const label =
      sentence.perplexity <= 5
        ? 'extremely predictable wording'
        : sentence.perplexity <= 20
          ? 'predictable wording'
          : 'less predictable wording';
    notes.push(`Perplexity: ${sentence.perplexity.toFixed(2)} (${label})`);
  }

  if (sentence.highlight_sentence_for_ai) {
    notes.push('Highlighted as contributing to the AI classification');
  }

  if (sentence.special_highlight_type) {
    notes.push(`Special highlight: ${sentence.special_highlight_type}`);
  }

  if (sentence.interpretability_designation) {
    notes.push(`Impact: ${sentence.interpretability_designation}`);
  }

  if (typeof sentence.interpretability_normalized_value === 'number') {
    notes.push(
      `Normalized impact value: ${sentence.interpretability_normalized_value.toFixed(3)}`
    );
  }

  return notes;
}

function getSentenceImpactScore(sentence: GPTZeroSentence): number {
  const interpretability =
    typeof sentence.interpretability_normalized_value === 'number'
      ? Math.abs(sentence.interpretability_normalized_value)
      : typeof sentence.interpretability_value === 'number'
        ? Math.abs(sentence.interpretability_value)
        : 0;
  const designation = sentence.interpretability_designation?.toLowerCase() ?? '';
  const designationBoost = designation.includes('high')
    ? 1
    : designation.includes('medium') || designation.includes('moderate')
      ? 0.5
      : designation.includes('low')
        ? 0.15
        : 0;
  const mappingSignal =
    /\b(use|uses|using|match|matches|matching|via|through)\b/i.test(sentence.sentence) &&
    /\b(uid|userId|app_user_id|id|identifier|field|fields)\b/i.test(sentence.sentence)
      ? 0.35
      : 0;
  const lengthSignal =
    sentence.sentence.length >= 110 && sentence.sentence.length <= 280 ? 0.1 : 0;

  return (
    interpretability * 2 +
    designationBoost +
    sentence.generated_prob +
    mappingSignal +
    lengthSignal
  );
}

function isHighImpactSentence(sentence: GPTZeroSentence): boolean {
  const designation = sentence.interpretability_designation?.toLowerCase() ?? '';
  if (designation.includes('high')) return true;

  const specialHighlight = sentence.special_highlight_type?.toLowerCase() ?? '';
  if (specialHighlight.includes('high')) return true;

  const normalizedImpact =
    typeof sentence.interpretability_normalized_value === 'number'
      ? Math.abs(sentence.interpretability_normalized_value)
      : null;

  return Boolean(sentence.highlight_sentence_for_ai && normalizedImpact !== null && normalizedImpact >= 0.66);
}

function formatProbability(label: string, value: number | undefined): string | null {
  return typeof value === 'number' ? `${label} ${formatPercent(value)}` : null;
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}
