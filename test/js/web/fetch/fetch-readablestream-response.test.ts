import { describe, test, expect } from "bun:test";
import { isCI } from "harness";

describe("ReadableStream response handling", () => {
  test("should handle server streaming response with client abort", async () => {
    const abortController = new AbortController();
    const { resolve: resolveFirstChunk } = Promise.withResolvers<void>();

    using server = Bun.serve({
      port: 0,
      async fetch() {
        const stream = new ReadableStream({
          async start(controller) {
            try {
              for (let i = 0; i < 10; i++) {
                const chunk = { content: `chunk${i}`, index: i };
                controller.enqueue(Buffer.from(JSON.stringify(chunk) + "\n", "utf-8"));
                if (i === 0) resolveFirstChunk();
                await Bun.sleep(10);
              }
            } catch (error) {
              controller.error(error);
            } finally {
              controller.close();
            }
          },
        });

        return new Response(stream, {
          headers: { "Content-Type": "application/json" },
        });
      },
    });

    const response = await fetch(server.url, { signal: abortController.signal });
    const reader = response.body!.getReader();

    // read first chunk
    const { value: firstValue } = await reader.read();
    expect(new TextDecoder().decode(firstValue)).toContain("chunk0");

    // abort mid-stream
    abortController.abort();

    // next read should throw AbortError
    await expect(reader.read()).rejects.toHaveProperty("name", "AbortError");
  });

  test("should handle nested readable stream wrapping with abort", async () => {
    // simulates the wrapStreamableBody pattern from smartFetch
    using server = Bun.serve({
      port: 0,
      async fetch() {
        const chunks = ["chunk1\n", "chunk2\n", "chunk3\n"];
        let index = 0;
        return new Response(
          new ReadableStream({
            async pull(controller) {
              if (index >= chunks.length) {
                controller.close();
                return;
              }
              await Bun.sleep(10);
              controller.enqueue(new TextEncoder().encode(chunks[index++]));
            },
          }),
        );
      },
    });

    const abortController = new AbortController();
    const response = await fetch(server.url, { signal: abortController.signal });

    // wrap the response body (simulating smartFetch's wrapStreamableBody)
    const originalReader = response.body!.getReader();
    let cleanupCalled = false;

    const wrappedStream = new ReadableStream({
      async pull(controller) {
        try {
          const { done, value } = await originalReader.read();
          if (done) {
            cleanupCalled = true;
            controller.close();
            return;
          }
          controller.enqueue(value);
        } catch (error) {
          cleanupCalled = true;
          controller.error(error);
        }
      },
      cancel(reason) {
        cleanupCalled = true;
        originalReader.cancel(reason);
      },
    });

    const reader = wrappedStream.getReader();

    // read one chunk
    await reader.read();

    // abort
    abortController.abort();

    // should trigger cleanup
    await expect(reader.read()).rejects.toHaveProperty("name", "AbortError");
  });

  test("should handle streamJson pattern with mid-stream abort", async () => {
    // simulates the streamStringsFromReader -> streamJson chain
    using server = Bun.serve({
      port: 0,
      async fetch() {
        const objects = [{ content: "hello" }, { content: "world" }, { content: "test" }];
        let index = 0;

        return new Response(
          new ReadableStream({
            async pull(controller) {
              if (index >= objects.length) {
                controller.close();
                return;
              }
              await Bun.sleep(20);
              controller.enqueue(new TextEncoder().encode(JSON.stringify(objects[index++]) + "\n"));
            },
          }),
        );
      },
    });

    const abortController = new AbortController();
    const { resolve: resolveFirst } = Promise.withResolvers<void>();

    async function* streamStringsFromReader(reader: ReadableStreamDefaultReader<Uint8Array>) {
      const decoder = new TextDecoder();
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          yield decoder.decode(value, { stream: true });
        }
      } finally {
        reader.cancel().catch(() => {});
      }
    }

    async function* streamJson<T>(generator: AsyncGenerator<string>) {
      let buffer = "";
      for await (const chunk of generator) {
        buffer += chunk;
        const lines = buffer.split("\n");
        for (const line of lines.slice(0, -1)) {
          if (line.trim()) {
            yield JSON.parse(line.trim()) as T;
          }
        }
        buffer = lines[lines.length - 1] ?? "";
      }
    }

    const response = await fetch(server.url, { signal: abortController.signal });
    const reader = response.body!.getReader();

    const jsonGenerator = streamJson<{ content: string }>(streamStringsFromReader(reader));

    // get first object
    const { value: first } = await jsonGenerator.next();
    expect(first?.content).toBe("hello");
    resolveFirst();

    // abort mid-stream
    abortController.abort();

    // next iteration should throw
    await expect(jsonGenerator.next()).rejects.toHaveProperty("name", "AbortError");
  });
});

describe("ReadableStream concurrent stress tests", () => {
  // simulates LLM API streaming with many concurrent clients
  const CONCURRENT_CLIENTS = isCI ? 100 : 200;
  const CHUNKS_PER_STREAM = 20;

  test("should handle many concurrent streaming clients", async () => {
    let activeStreams = 0;
    let peakActiveStreams = 0;
    let completedStreams = 0;

    using server = Bun.serve({
      port: 0,
      async fetch(req) {
        activeStreams++;
        peakActiveStreams = Math.max(peakActiveStreams, activeStreams);

        const clientId = req.headers.get("x-client-id") || "unknown";
        const stream = new ReadableStream({
          async start(controller) {
            try {
              for (let i = 0; i < CHUNKS_PER_STREAM; i++) {
                const chunk = { clientId, chunk: i, data: Buffer.alloc(100, "x").toString() };
                controller.enqueue(new TextEncoder().encode(JSON.stringify(chunk) + "\n"));
                // small delay to simulate real streaming
                await Bun.sleep(1);
              }
              controller.close();
              completedStreams++;
            } catch {
              // client aborted
            } finally {
              activeStreams--;
            }
          },
        });

        return new Response(stream, {
          headers: { "Content-Type": "application/x-ndjson" },
        });
      },
    });

    const results = {
      completed: 0,
      aborted: 0,
      errors: 0,
      totalChunksRead: 0,
    };

    const clientPromises = Array.from({ length: CONCURRENT_CLIENTS }, async (_, clientId) => {
      const abortController = new AbortController();
      const shouldAbort = clientId % 5 === 0; // 20% of clients abort mid-stream
      const abortAfterChunks = Math.floor(CHUNKS_PER_STREAM / 2);

      try {
        const response = await fetch(server.url, {
          signal: abortController.signal,
          headers: { "x-client-id": String(clientId) },
        });

        const reader = response.body!.getReader();
        let chunksRead = 0;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          chunksRead++;
          results.totalChunksRead++;

          // abort after reading some chunks
          if (shouldAbort && chunksRead >= abortAfterChunks) {
            abortController.abort();
            throw new DOMException("Aborted", "AbortError");
          }
        }

        results.completed++;
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
          results.aborted++;
        } else {
          results.errors++;
        }
      }
    });

    await Promise.all(clientPromises);

    // verify results
    expect(results.completed + results.aborted).toBe(CONCURRENT_CLIENTS);
    expect(results.errors).toBe(0);
    expect(results.aborted).toBeGreaterThan(0);
    expect(results.completed).toBeGreaterThan(0);
    expect(peakActiveStreams).toBeGreaterThan(1); // should have concurrent streams
  });

  test("should handle rapid connect/disconnect cycle", async () => {
    let connectionCount = 0;
    let disconnectionCount = 0;

    using server = Bun.serve({
      port: 0,
      async fetch() {
        connectionCount++;
        const stream = new ReadableStream({
          async start(controller) {
            // send a few chunks then wait
            for (let i = 0; i < 5; i++) {
              controller.enqueue(new TextEncoder().encode(`chunk${i}\n`));
              await Bun.sleep(5);
            }
            controller.close();
          },
          cancel() {
            disconnectionCount++;
          },
        });

        return new Response(stream);
      },
    });

    const CYCLES = isCI ? 50 : 100;
    const results = { earlyAborts: 0, completed: 0 };

    // rapid fire requests, abort immediately after first chunk
    const promises = Array.from({ length: CYCLES }, async (_, i) => {
      const abortController = new AbortController();
      const abortEarly = i % 2 === 0;

      try {
        const response = await fetch(server.url, { signal: abortController.signal });
        const reader = response.body!.getReader();

        // read one chunk
        await reader.read();

        if (abortEarly) {
          abortController.abort();
          results.earlyAborts++;
          return;
        }

        // read rest
        while (true) {
          const { done } = await reader.read();
          if (done) break;
        }
        results.completed++;
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
          results.earlyAborts++;
        }
      }
    });

    await Promise.all(promises);

    expect(connectionCount).toBe(CYCLES);
    expect(results.earlyAborts + results.completed).toBe(CYCLES);
  });

  test("should handle mixed large and small stream payloads", async () => {
    using server = Bun.serve({
      port: 0,
      async fetch(req) {
        const size = req.headers.get("x-payload-size") || "small";
        const chunkSize = size === "large" ? 10_000 : 100;
        const chunkCount = size === "large" ? 50 : 10;

        const stream = new ReadableStream({
          async start(controller) {
            for (let i = 0; i < chunkCount; i++) {
              controller.enqueue(new Uint8Array(chunkSize).fill(65 + (i % 26)));
              await Bun.sleep(1);
            }
            controller.close();
          },
        });

        return new Response(stream);
      },
    });

    const CLIENTS = isCI ? 30 : 60;
    let totalBytesReceived = 0;
    let completedClients = 0;

    const promises = Array.from({ length: CLIENTS }, async (_, i) => {
      const isLarge = i % 3 === 0; // 1/3 large payloads
      const abortController = new AbortController();
      const shouldAbort = i % 7 === 0; // ~14% abort

      try {
        const response = await fetch(server.url, {
          signal: abortController.signal,
          headers: { "x-payload-size": isLarge ? "large" : "small" },
        });

        const reader = response.body!.getReader();
        let bytesRead = 0;
        let chunksRead = 0;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          bytesRead += value!.byteLength;
          chunksRead++;

          // abort large streams after a few chunks
          if (shouldAbort && isLarge && chunksRead > 5) {
            abortController.abort();
            throw new DOMException("Aborted", "AbortError");
          }
        }

        totalBytesReceived += bytesRead;
        completedClients++;
      } catch (error) {
        if (!(error instanceof Error && error.name === "AbortError")) {
          throw error;
        }
      }
    });

    await Promise.all(promises);

    expect(completedClients).toBeGreaterThan(0);
    expect(totalBytesReceived).toBeGreaterThan(0);
  });

  test("should handle concurrent readers with varying consumption speeds", async () => {
    let completedReaders = 0;
    let abortedReaders = 0;

    using server = Bun.serve({
      port: 0,
      async fetch(req) {
        const clientId = req.headers.get("x-client-id")!;
        const chunkCount = parseInt(req.headers.get("x-chunk-count") || "30");

        const stream = new ReadableStream({
          async start(controller) {
            for (let i = 0; i < chunkCount; i++) {
              controller.enqueue(new TextEncoder().encode(`${clientId}:chunk${i}\n`));
              await Bun.sleep(2);
            }
            controller.close();
          },
        });

        return new Response(stream);
      },
    });

    const CLIENTS = isCI ? 40 : 80;

    const promises = Array.from({ length: CLIENTS }, async (_, i) => {
      const speed = i % 4; // 0: fast, 1: medium, 2: slow, 3: abort early
      const readDelays = [0, 2, 5, 10];
      const readDelay = readDelays[speed];
      const abortController = new AbortController();

      try {
        const response = await fetch(server.url, {
          signal: abortController.signal,
          headers: {
            "x-client-id": String(i),
            "x-chunk-count": "20",
          },
        });

        const reader = response.body!.getReader();
        let chunksRead = 0;

        while (true) {
          const { done } = await reader.read();
          if (done) break;

          chunksRead++;
          if (readDelay > 0) {
            await Bun.sleep(readDelay);
          }

          // abort early after 5 chunks for every 4th client
          if (speed === 3 && chunksRead >= 5) {
            abortController.abort();
            throw new DOMException("Aborted", "AbortError");
          }
        }

        completedReaders++;
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
          abortedReaders++;
        } else {
          throw error;
        }
      }
    });

    await Promise.all(promises);

    // verify we have a mix of completed and aborted
    expect(completedReaders + abortedReaders).toBe(CLIENTS);
    expect(abortedReaders).toBeGreaterThan(0);
    expect(completedReaders).toBeGreaterThan(0);
  });

  test("should handle backpressure with slow consumers", async () => {
    let serverCompleted = false;
    const CHUNK_SIZE = 50_000;
    const CHUNK_COUNT = 100;

    using server = Bun.serve({
      port: 0,
      async fetch() {
        const stream = new ReadableStream({
          async start(controller) {
            for (let i = 0; i < CHUNK_COUNT; i++) {
              // large chunks to trigger backpressure
              controller.enqueue(new Uint8Array(CHUNK_SIZE).fill(65));
              await Bun.sleep(1);
            }
            serverCompleted = true;
            controller.close();
          },
        });

        return new Response(stream);
      },
    });

    const response = await fetch(server.url);
    const reader = response.body!.getReader();
    let bytesReceived = 0;
    let chunksReceived = 0;

    while (true) {
      // simulate slow consumer
      await Bun.sleep(10);
      const { done, value } = await reader.read();
      if (done) break;

      bytesReceived += value!.byteLength;
      chunksReceived++;
    }

    expect(serverCompleted).toBe(true);
    expect(bytesReceived).toBe(CHUNK_COUNT * CHUNK_SIZE);
    // chunks may be coalesced by the network layer, so we just verify we received some
    expect(chunksReceived).toBeGreaterThan(0);
    expect(chunksReceived).toBeLessThanOrEqual(CHUNK_COUNT);
  });
});
