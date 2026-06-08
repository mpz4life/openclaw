/**
 * Plugin-defined text replacement transforms for stream boundaries.
 *
 * Provider and CLI plugins can rewrite prompt/event text without owning the transport implementation.
 *
 * Safety: text transforms are applied to all text on the message bus, including
 * filesystem paths in tool-call arguments. When a plugin registers a replacement
 * that matches a codename occurring in both legacy directory names and general
 * text, the replacement silently misroutes file operations to the current-name
 * directory. Use applySafeTextReplacements for output text or any text that
 * may contain filesystem paths.
 */
import { shouldLogVerbose } from "../globals.js";
import type { AssistantMessageEvent } from "../llm/types.js";
import type { PluginTextReplacement, PluginTextTransforms } from "../plugins/cli-backend.types.js";
import type { StreamFn } from "./runtime/index.js";
import type { MutableAssistantMessageEventStream } from "./stream-compat.js";
import { createStreamIteratorWrapper } from "./stream-iterator-wrapper.js";

// Applies plugin-defined text replacement transforms to stream input/output.
// Used by provider/CLI plugins that need compatibility rewrites at boundaries.

const SKIP_TRANSFORMS_ENV = "OPENCLAW_SKIP_TEXT_TRANSFORMS";

function isTextTransformsDisabled(): boolean {
  const val = process.env?.[SKIP_TRANSFORMS_ENV];
  return val === "1" || val === "true" || val === "yes";
}

/** Merge multiple plugin text-transform sets. */
export function mergePluginTextTransforms(
  ...transforms: Array<PluginTextTransforms | undefined>
): PluginTextTransforms | undefined {
  const input = transforms.flatMap((entry) => entry?.input ?? []);
  const output = transforms.flatMap((entry) => entry?.output ?? []);
  if (input.length === 0 && output.length === 0) {
    return undefined;
  }
  return {
    ...(input.length > 0 ? { input } : {}),
    ...(output.length > 0 ? { output } : {}),
  };
}

/** Apply sequential plugin text replacements to one string. */
export function applyPluginTextReplacements(
  text: string,
  replacements?: PluginTextReplacement[],
): string {
  if (!replacements || replacements.length === 0 || !text) {
    return text;
  }
  if (isTextTransformsDisabled()) {
    return text;
  }
  let next = text;
  for (const replacement of replacements) {
    const fromStr = typeof replacement.from === "string" ? replacement.from : undefined;
    if (shouldLogVerbose() && typeof replacement.from === "string" && next.includes(fromStr)) {
      console.log(
        `[text-transforms] apply: "${replacement.from}" -> "${replacement.to}" in ${next.length} bytes`,
      );
    }
    next = next.replace(replacement.from, replacement.to);
  }
  return next;
}

/**
 * Apply plugin text replacements with path-context safety.
 *
 * When a replacement target appears adjacent to filesystem path delimiters
 * (`/`, `~`, or after `$HOME`), the replacement is skipped for that
 * occurrence to prevent silent misrouting of filesystem paths in tool-call
 * arguments. This preserves cosmetic text rewrites in conversational text
 * while protecting functional paths from unwanted substitution.
 *
 * Only string-based (non-RegExp) replacements support occurrence-level
 * path detection. RegExp replacements fall through to the normal path.
 */
export function applySafeTextReplacements(
  text: string,
  replacements?: PluginTextReplacement[],
): string {
  if (!replacements || replacements.length === 0 || !text) {
    return text;
  }
  if (isTextTransformsDisabled()) {
    return text;
  }

  let result = text;
  for (const replacement of replacements) {
    if (typeof replacement.from !== "string") {
      const matched = result.match(replacement.from);
      if (matched && shouldLogVerbose()) {
        console.log(
          `[text-transforms] apply (regexp): /${replacement.from.source}/ -> "${replacement.to}" in ${result.length} bytes`,
        );
      }
      result = result.replace(replacement.from, replacement.to);
      continue;
    }

    const fromStr = replacement.from;
    const toStr = replacement.to;
    let pos = 0;
    let chunk = "";
    let lastIndex = 0;
    let applied = false;

    while ((pos = result.indexOf(fromStr, pos)) !== -1) {
      const before = pos > 0 ? result[pos - 1] : "";
      const after = pos + fromStr.length < result.length ? result[pos + fromStr.length] : "";

      const isPathContext =
        before === "/" ||
        before === "~" ||
        after === "/" ||
        (before === "$" && result.slice(Math.max(0, pos - 5), pos) === "$HOME");

      if (isPathContext) {
        chunk += result.slice(lastIndex, pos + fromStr.length);
      } else {
        chunk += result.slice(lastIndex, pos) + toStr;
        applied = true;
      }

      lastIndex = pos + fromStr.length;
      pos = pos + fromStr.length;
    }
    chunk += result.slice(lastIndex);

    if (applied) {
      if (shouldLogVerbose()) {
        console.log(`[text-transforms] safe: "${fromStr}" -> "${toStr}" in ${result.length} bytes`);
      }
      result = chunk;
    }
  }

  return result;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function transformContentText(content: unknown, replacements?: PluginTextReplacement[]): unknown {
  if (typeof content === "string") {
    return applyPluginTextReplacements(content, replacements);
  }
  if (Array.isArray(content)) {
    return content.map((entry) => transformContentText(entry, replacements));
  }
  if (!isRecord(content)) {
    return content;
  }
  const next = { ...content };
  if (typeof next.text === "string") {
    next.text = applyPluginTextReplacements(next.text, replacements);
  }
  if (Object.hasOwn(next, "content")) {
    next.content = transformContentText(next.content, replacements);
  }
  return next;
}

function transformMessageText(message: unknown, replacements?: PluginTextReplacement[]): unknown {
  if (!isRecord(message)) {
    return message;
  }
  const next = { ...message };
  if (Object.hasOwn(next, "content")) {
    next.content = transformContentText(next.content, replacements);
  }
  if (typeof next.errorMessage === "string") {
    next.errorMessage = applyPluginTextReplacements(next.errorMessage, replacements);
  }
  return next;
}

/** Apply input text replacements to a stream context. */
export function transformStreamContextText(
  context: Parameters<StreamFn>[1],
  replacements?: PluginTextReplacement[],
  options?: { systemPrompt?: boolean },
): Parameters<StreamFn>[1] {
  if (!replacements || replacements.length === 0) {
    return context;
  }
  return {
    ...context,
    systemPrompt:
      options?.systemPrompt !== false && typeof context.systemPrompt === "string"
        ? applyPluginTextReplacements(context.systemPrompt, replacements)
        : context.systemPrompt,
    messages: Array.isArray(context.messages)
      ? context.messages.map((message) => transformMessageText(message, replacements))
      : context.messages,
  } as Parameters<StreamFn>[1];
}

function transformAssistantEventText(
  event: unknown,
  replacements?: PluginTextReplacement[],
): AssistantMessageEvent {
  if (!isRecord(event) || !replacements || replacements.length === 0) {
    return event as AssistantMessageEvent;
  }
  const next = { ...event };
  if (next.type === "text_delta" && typeof next.delta === "string") {
    next.delta = applyPluginTextReplacements(next.delta, replacements);
  }
  if (next.type === "text_end" && typeof next.content === "string") {
    next.content = applyPluginTextReplacements(next.content, replacements);
  }
  if (Object.hasOwn(next, "partial")) {
    next.partial = transformMessageText(next.partial, replacements);
  }
  if (Object.hasOwn(next, "message")) {
    next.message = transformMessageText(next.message, replacements);
  }
  if (Object.hasOwn(next, "error")) {
    next.error = transformMessageText(next.error, replacements);
  }
  return next as AssistantMessageEvent;
}

function wrapStreamTextTransforms(
  stream: MutableAssistantMessageEventStream,
  replacements?: PluginTextReplacement[],
): MutableAssistantMessageEventStream {
  if (!replacements || replacements.length === 0) {
    return stream;
  }
  const originalResult = stream.result.bind(stream);
  stream.result = async () => transformMessageText(await originalResult(), replacements) as never;

  // Wrap async iteration so streamed deltas and the final result receive the
  // same output replacement policy.
  const originalAsyncIterator = stream[Symbol.asyncIterator].bind(stream);
  (stream as { [Symbol.asyncIterator]: typeof originalAsyncIterator })[Symbol.asyncIterator] =
    function () {
      const iterator = originalAsyncIterator();
      return createStreamIteratorWrapper({
        iterator,
        next: async (streamIterator) => {
          const result = await streamIterator.next();
          return result.done
            ? result
            : {
                done: false as const,
                value: transformAssistantEventText(result.value, replacements),
              };
        },
      });
    };
  return stream;
}

/** Wrap a stream function with plugin input/output text transforms. */
export function wrapStreamFnTextTransforms(params: {
  streamFn: StreamFn;
  input?: PluginTextReplacement[];
  output?: PluginTextReplacement[];
  transformSystemPrompt?: boolean;
}): StreamFn {
  return (model, context, options) => {
    const nextContext = transformStreamContextText(context, params.input, {
      systemPrompt: params.transformSystemPrompt,
    });
    const maybeStream = params.streamFn(model, nextContext, options);
    if (maybeStream && typeof maybeStream === "object" && "then" in maybeStream) {
      return Promise.resolve(maybeStream).then((stream) =>
        wrapStreamTextTransforms(stream, params.output),
      );
    }
    return wrapStreamTextTransforms(maybeStream, params.output);
  };
}
