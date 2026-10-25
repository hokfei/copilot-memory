import { existsSync } from "node:fs";
import { embedModelPath } from "./paths.mjs";

export const DOC_PREFIX = "";
export const QUERY_PREFIX =
  "Instruct: Given a question, retrieve relevant passages from past conversations that answer the question\nQuery:";

let llama = null;
let model = null;
let context = null;

export function l2normalize(vector) {
  let sum = 0;
  for (let i = 0; i < vector.length; i++) sum += vector[i] * vector[i];
  const norm = Math.sqrt(sum) || 1;
  const out = new Array(vector.length);
  for (let i = 0; i < vector.length; i++) out[i] = vector[i] / norm;
  return out;
}

async function ensureContext() {
  if (context) return context;
  // The chat model and the embedder cannot share VRAM: unload the chat side first.
  try {
    const { disposeChat } = await import("./synthesize.mjs");
    await disposeChat();
  } catch {}
  if (!existsSync(embedModelPath)) {
    throw new Error(
      `Embedding model not found at ${embedModelPath}. Set SESSION_MEMORY_EMBED_MODEL to the Qwen3-Embedding-0.6B Q8_0 GGUF path.`
    );
  }
  const { getLlama } = await import("node-llama-cpp");
  llama = await getLlama();
  model = await llama.loadModel({ modelPath: embedModelPath });
  context = await model.createEmbeddingContext({ contextSize: { max: 2048 } });
  return context;
}

export async function embedDocuments(texts, onProgress) {
  const ctx = await ensureContext();
  const vectors = [];
  for (let i = 0; i < texts.length; i++) {
    const { vector } = await ctx.getEmbeddingFor(DOC_PREFIX + texts[i]);
    vectors.push(l2normalize(vector));
    if (onProgress) onProgress(i + 1, texts.length);
  }
  return vectors;
}

export async function embedQuery(text) {
  const ctx = await ensureContext();
  const { vector } = await ctx.getEmbeddingFor(QUERY_PREFIX + text);
  return l2normalize(vector);
}

export async function disposeEmbedder() {
  if (!llama) {
    llama = model = context = null;
    return;
  }
  try {
    await llama.dispose();
  } finally {
    llama = model = context = null;
  }
}
