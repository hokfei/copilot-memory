import { existsSync } from "node:fs";
import { chatModelPath } from "./paths.mjs";

const CONTEXT_SIZE = 4096;
const RESERVED_FOR_REPLY = 1024;

let llama = null;
let model = null;
let context = null;

async function ensureContext(contextSize = CONTEXT_SIZE) {
  if (context) return context;
  // The embedder and the chat model cannot share VRAM: unload the embedder first.
  try {
    const { disposeEmbedder } = await import("./embed.mjs");
    await disposeEmbedder();
  } catch {}
  if (!existsSync(chatModelPath)) {
    throw new Error(
      `Chat model not found at ${chatModelPath}. Set SESSION_MEMORY_CHAT_MODEL to a local GGUF chat model.`
    );
  }
  const { getLlama } = await import("node-llama-cpp");
  llama = await getLlama();
  model = await llama.loadModel({ modelPath: chatModelPath });
  try {
    context = await model.createContext({ contextSize });
  } catch (err) {
    if (contextSize > 2048) {
      context = await model.createContext({ contextSize: 2048 });
    } else {
      throw err;
    }
  }
  return context;
}

export async function disposeChat() {
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

// Rough token estimate for budgeting: ~4 chars per token.
export function estimateTokens(text) {
  return Math.ceil(text.length / 4);
}

// Fit numbered items into the prompt budget; truncate longest first.
export function fitItems(items, maxTokens) {
  const fitted = items.map((t) => t);
  let total = fitted.reduce((n, t) => n + estimateTokens(t), 0);
  let i = 0;
  while (total > maxTokens && fitted.some((t) => t.length > 200)) {
    const idx = fitted.findIndex(
      (t, j) => t.length >= Math.max(...fitted.map((x) => x.length)) && j >= i
    );
    if (idx < 0) break;
    const cut = fitted[idx].length - 200;
    fitted[idx] = fitted[idx].slice(0, 200) + "…";
    total -= Math.ceil(cut / 4);
    i++;
    if (i > fitted.length * 10) break;
  }
  if (total > maxTokens) {
    // Drop from the end until it fits.
    while (fitted.length > 1 && total > maxTokens) {
      const dropped = fitted.pop();
      total -= estimateTokens(dropped);
    }
  }
  return { items: fitted, estimatedTokens: total, dropped: items.length - fitted.length };
}

export async function runChat(systemPrompt, userPrompt, itemTexts = []) {
  const ctx = await ensureContext();
  const { LlamaChatSession } = await import("node-llama-cpp");
  const budget =
    CONTEXT_SIZE -
    RESERVED_FOR_REPLY -
    estimateTokens(systemPrompt) -
    estimateTokens(userPrompt);
  const { items, dropped } = fitItems(itemTexts, Math.max(512, budget));
  const numbered = items.map((t, i) => `[${i + 1}] ${t}`).join("\n");
  const full = items.length > 0 ? `${userPrompt}\n\n${numbered}` : userPrompt;
  const sequence = ctx.getSequence();
  const chat = new LlamaChatSession({
    contextSequence: sequence,
    systemPrompt,
  });
  try {
    const reply = await chat.prompt(full);
    return { reply: reply.trim(), itemCount: items.length, dropped };
  } finally {
    // The context pool holds a single sequence: reclaim it so the next
    // runChat on this context doesn't throw "No sequences left". Reclaim is
    // async (lock-guarded id return), so it must be awaited — getSequence
    // pops synchronously and would otherwise race the reclaim.
    await sequence.dispose();
  }
}

export function extractJson(text) {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const candidate = (fenced ? fenced[1] : text).trim();
  const start = candidate.indexOf("{");
  const end = candidate.lastIndexOf("}");
  if (start < 0 || end <= start) throw new Error("No JSON object in model reply.");
  return JSON.parse(candidate.slice(start, end + 1));
}
