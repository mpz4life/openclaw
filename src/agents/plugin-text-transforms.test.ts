// Verifies plugin text transforms rewrite prompts and streamed assistant output.
import type { StreamFn } from "openclaw/plugin-sdk/agent-core";
import {
  createAssistantMessageEventStream,
  type AssistantMessage,
  type Context,
  type Model,
} from "openclaw/plugin-sdk/llm";
import { describe, expect, it } from "vitest";
import {
  applyPluginTextReplacements,
  applySafeTextReplacements,
  mergePluginTextTransforms,
  transformStreamContextText,
  wrapStreamFnTextTransforms,
} from "./plugin-text-transforms.js";

const model = {
  api: "openai-responses",
  provider: "test",
  id: "test-model",
} as Model<"openai-responses">;

function makeAssistantMessage(text: string): AssistantMessage {
  // Output transform tests need a complete assistant message with visible text.
  return {
    role: "assistant",
    content: [{ type: "text", text }],
    stopReason: "stop",
    api: "openai-responses",
    provider: "test",
    model: "test-model",
    usage: {
      input: 1,
      output: 1,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 2,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    timestamp: 0,
  };
}

describe("plugin text transforms", () => {
  it("merges registered transform groups in order", () => {
    const merged = mergePluginTextTransforms(
      { input: [{ from: /red basket/g, to: "blue basket" }] },
      { output: [{ from: /blue basket/g, to: "red basket" }] },
      { input: [{ from: /paper ticket/g, to: "digital ticket" }] },
    );

    expect(merged).toStrictEqual({
      input: [
        { from: /red basket/g, to: "blue basket" },
        { from: /paper ticket/g, to: "digital ticket" },
      ],
      output: [{ from: /blue basket/g, to: "red basket" }],
    });
    expect(applyPluginTextReplacements("red basket paper ticket", merged?.input)).toBe(
      "blue basket digital ticket",
    );
  });

  it("applies ordered string and regexp replacements", () => {
    expect(
      applyPluginTextReplacements("paper ticket on the left shelf", [
        { from: /paper ticket/g, to: "digital ticket" },
        { from: /left shelf/g, to: "right shelf" },
        { from: "digital ticket", to: "counter receipt" },
      ]),
    ).toBe("counter receipt on the right shelf");
  });

  it("rewrites system prompt and message text content before transport", () => {
    const context = transformStreamContextText(
      {
        systemPrompt: "Use orchid mailbox inside north tower",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "Please use the red basket" },
              { type: "image", url: "data:image/png;base64,abc" },
            ],
          },
        ],
      } as Context,
      [
        {
          from: /orchid mailbox/g,
          to: "pine mailbox",
        },
        { from: /red basket/g, to: "blue basket" },
      ],
    ) as unknown as { systemPrompt: string; messages: Array<{ content: unknown[] }> };

    expect(context.systemPrompt).toBe("Use pine mailbox inside north tower");
    const textContent = context.messages[0]?.content[0] as
      | { type?: string; text?: string }
      | undefined;
    expect(textContent?.type).toBe("text");
    expect(textContent?.text).toBe("Please use the blue basket");
    const imageContent = context.messages[0]?.content[1] as
      | { type?: string; url?: string }
      | undefined;
    expect(imageContent?.type).toBe("image");
    expect(imageContent?.url).toBe("data:image/png;base64,abc");
  });

  it("wraps stream functions with inbound and outbound replacements", async () => {
    // The wrapper mutates text-only blocks while preserving non-text content.
    let capturedContext: Context | undefined;
    const baseStreamFn: StreamFn = (_model, context) => {
      capturedContext = context;
      const stream = createAssistantMessageEventStream();
      queueMicrotask(() => {
        const partial = makeAssistantMessage("blue basket on the right shelf");
        stream.push({
          type: "text_delta",
          contentIndex: 0,
          delta: "blue basket on the right shelf",
          partial,
        });
        stream.push({
          type: "done",
          reason: "stop",
          message: makeAssistantMessage("final blue basket on the right shelf"),
        });
        stream.end();
      });
      return stream;
    };

    const wrapped = wrapStreamFnTextTransforms({
      streamFn: baseStreamFn,
      input: [{ from: /red basket/g, to: "blue basket" }],
      output: [
        { from: /blue basket/g, to: "red basket" },
        { from: /right shelf/g, to: "left shelf" },
      ],
      transformSystemPrompt: false,
    });
    const stream = await Promise.resolve(
      wrapped(
        model,
        {
          systemPrompt: "Keep red basket untouched here",
          messages: [{ role: "user", content: "Use red basket" }],
        } as Context,
        undefined,
      ),
    );
    const events = [];
    for await (const event of stream) {
      events.push(event);
    }
    const result = await stream.result();

    expect(capturedContext?.systemPrompt).toBe("Keep red basket untouched here");
    expect(capturedContext?.messages).toEqual([{ role: "user", content: "Use blue basket" }]);
    const firstEvent = events[0] as { type?: string; delta?: string } | undefined;
    expect(firstEvent?.type).toBe("text_delta");
    expect(firstEvent?.delta).toBe("red basket on the left shelf");
    expect(result.content).toEqual([{ type: "text", text: "final red basket on the left shelf" }]);
  });

  describe("applySafeTextReplacements", () => {
    it("skips replacement when matched string is part of a filesystem path", () => {
      const result = applySafeTextReplacements(
        "Run: ls -la ~/openclaw/config && echo use openclaw",
        [{ from: "openclaw", to: "ocplatform" }],
      );
      expect(result).toBe("Run: ls -la ~/openclaw/config && echo use ocplatform");
    });

    it("applies replacement when matched string is not in a path context", () => {
      const result = applySafeTextReplacements("the openclaw project uses openclaw brand", [
        { from: "openclaw", to: "ocplatform" },
      ]);
      expect(result).toBe("the ocplatform project uses ocplatform brand");
    });

    it("handles path starting with home directory reference", () => {
      const result = applySafeTextReplacements("Check $HOME/openclaw/AGENTS.md", [
        { from: "openclaw", to: "ocplatform" },
      ]);
      expect(result).toBe("Check $HOME/openclaw/AGENTS.md");
    });

    it("applies regex replacements normally (no path check)", () => {
      const result = applySafeTextReplacements("~/openclaw/data has openclaw content", [
        { from: /openclaw/g, to: "ocplatform" },
      ]);
      expect(result).toBe("~/ocplatform/data has ocplatform content");
    });

    it("preserves text when no replacements match", () => {
      const result = applySafeTextReplacements("no matching words here", [
        { from: "openclaw", to: "ocplatform" },
      ]);
      expect(result).toBe("no matching words here");
    });

    it("returns original text for empty replacements", () => {
      const result = applySafeTextReplacements("some text", []);
      expect(result).toBe("some text");
    });

    it("returns original text for undefined replacements", () => {
      const result = applySafeTextReplacements("some text", undefined);
      expect(result).toBe("some text");
    });
  });
});
