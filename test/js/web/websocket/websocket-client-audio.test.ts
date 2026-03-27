import { describe, expect, test } from "bun:test";
import { join } from "path";

const AUDIO_FILE_PATH = join(import.meta.dir, "../fetch/audio-test.wav");

describe("WebSocket audio streaming patterns", () => {
  // tests the pattern where ws.data holds an entry point object (like phoner's PhoneEntryPoint)
  test("should handle WebSocket data context with lifecycle callbacks", async () => {
    interface EntryPoint {
      provider: string;
      destroyed: boolean;
      onWebSocketMessage(message: string | Buffer): Promise<void>;
      destroy(): Promise<void>;
    }

    const entryPoints: EntryPoint[] = [];

    using server = Bun.serve<EntryPoint>({
      port: 0,
      fetch(req, server) {
        if (req.headers.get("upgrade") === "websocket") {
          server.upgrade(req, {
            data: {
              provider: "some-company-idk",
              destroyed: false,
              async onWebSocketMessage(message) {
                // simulates entry point message handling
              },
              async destroy() {
                this.destroyed = true;
              },
            } satisfies EntryPoint,
          });
          return new Response(null, { status: 101 });
        }
        return new Response("Not found", { status: 404 });
      },
      websocket: {
        open(ws) {
          entryPoints.push(ws.data);
        },
        async message(ws, message) {
          await ws.data.onWebSocketMessage(typeof message === "string" ? message : Buffer.from(message));
        },
        async close(ws) {
          await ws.data.destroy();
        },
      },
    });

    const ws = new WebSocket(server.url.toString().replace("http", "ws"));
    await new Promise<void>(resolve => {
      ws.onopen = () => resolve();
    });

    expect(entryPoints.length).toBe(1);
    expect(entryPoints[0]!.provider).toBe("some-company-idk");
    expect(entryPoints[0]!.destroyed).toBe(false);

    ws.close();
    await Bun.sleep(50);
    expect(entryPoints[0]!.destroyed).toBe(true);
  });

  // tests error handling in websocket message handler
  test("should handle errors in message handler without crashing", async () => {
    let errorSent = false;
    let errorMessage = "";

    using server = Bun.serve({
      port: 0,
      fetch(req, server) {
        if (req.headers.get("upgrade") === "websocket") {
          server.upgrade(req);
          return new Response(null, { status: 101 });
        }
        return new Response("OK");
      },
      websocket: {
        async message(ws, message) {
          try {
            throw new Error("Processing error");
          } catch (error) {
            errorSent = true;
            errorMessage = (error as Error).message;
            ws.send(
              JSON.stringify({
                event: "error",
                error: errorMessage,
              }),
            );
          }
        },
      },
    });

    const ws = new WebSocket(server.url.toString().replace("http", "ws"));
    await new Promise<void>(resolve => {
      ws.onopen = () => resolve();
    });

    const errorPromise = new Promise<any>(resolve => {
      ws.onmessage = e => resolve(JSON.parse(e.data));
    });

    ws.send("trigger-error");
    const errorResponse = await errorPromise;

    expect(errorSent).toBe(true);
    expect(errorResponse.event).toBe("error");
    expect(errorResponse.error).toBe("Processing error");

    ws.close();
  });

  // tests binary audio chunk streaming
  test("should handle binary audio chunks", async () => {
    const receivedChunks: Buffer[] = [];
    const { promise: allChunksReceived, resolve: resolveChunks } = Promise.withResolvers<void>();

    using server = Bun.serve({
      port: 0,
      fetch(req, server) {
        if (req.headers.get("upgrade") === "websocket") {
          server.upgrade(req);
          return new Response(null, { status: 101 });
        }
        return new Response("OK");
      },
      websocket: {
        message(ws, message) {
          if (message instanceof Buffer || message instanceof ArrayBuffer) {
            const buffer = Buffer.from(message);
            receivedChunks.push(buffer);

            ws.send(JSON.stringify({ chunksReceived: receivedChunks.length }));

            if (receivedChunks.length === 3) {
              resolveChunks();
            }
          }
        },
      },
    });

    const ws = new WebSocket(server.url.toString().replace("http", "ws"));
    await new Promise<void>(resolve => {
      ws.onopen = () => resolve();
    });

    // send binary audio chunks (simulating 20ms of 8kHz audio each)
    const chunk1 = Buffer.alloc(160, 0x7f);
    const chunk2 = Buffer.alloc(160, 0x80);
    const chunk3 = Buffer.alloc(160, 0x81);

    ws.send(chunk1);
    ws.send(chunk2);
    ws.send(chunk3);

    await allChunksReceived;

    expect(receivedChunks.length).toBe(3);
    expect(receivedChunks[0]![0]).toBe(0x7f);
    expect(receivedChunks[1]![0]).toBe(0x80);
    expect(receivedChunks[2]![0]).toBe(0x81);

    ws.close();
  });

  // tests streaming real audio file over WebSocket
  test("should stream real audio file in chunks", async () => {
    const audioData = await Bun.file(AUDIO_FILE_PATH).arrayBuffer();
    const audioBuffer = Buffer.from(audioData);

    const receivedChunks: Buffer[] = [];
    const CHUNK_SIZE = 1024;
    const expectedChunkCount = Math.ceil(audioBuffer.length / CHUNK_SIZE);
    const { promise: allChunksReceived, resolve: resolveChunks } = Promise.withResolvers<void>();

    using server = Bun.serve({
      port: 0,
      fetch(req, server) {
        if (req.headers.get("upgrade") === "websocket") {
          server.upgrade(req);
          return new Response(null, { status: 101 });
        }
        return new Response("OK");
      },
      websocket: {
        message(ws, message) {
          if (message instanceof Buffer || message instanceof ArrayBuffer) {
            receivedChunks.push(Buffer.from(message));
            if (receivedChunks.length === expectedChunkCount) {
              resolveChunks();
            }
          }
        },
      },
    });

    const ws = new WebSocket(server.url.toString().replace("http", "ws"));
    await new Promise<void>(resolve => {
      ws.onopen = () => resolve();
    });

    // stream audio in chunks
    for (let offset = 0; offset < audioBuffer.length; offset += CHUNK_SIZE) {
      const chunk = audioBuffer.subarray(offset, Math.min(offset + CHUNK_SIZE, audioBuffer.length));
      ws.send(chunk);
    }

    await allChunksReceived;

    // reassemble and verify
    const reassembled = Buffer.concat(receivedChunks);
    expect(reassembled.length).toBe(audioBuffer.length);
    expect(reassembled.equals(audioBuffer)).toBe(true);

    ws.close();
  });

  // tests bidirectional audio streaming (server echoes back)
  test("should handle bidirectional audio streaming", async () => {
    const audioData = await Bun.file(AUDIO_FILE_PATH).arrayBuffer();
    const audioBuffer = Buffer.from(audioData);

    // use smaller portion for faster test
    const testBuffer = audioBuffer.subarray(0, 10 * 1024);
    const CHUNK_SIZE = 512;
    const expectedChunkCount = Math.ceil(testBuffer.length / CHUNK_SIZE);

    const receivedEchoes: Buffer[] = [];
    const { promise: allEchoesReceived, resolve: resolveEchoes } = Promise.withResolvers<void>();

    using server = Bun.serve({
      port: 0,
      fetch(req, server) {
        if (req.headers.get("upgrade") === "websocket") {
          server.upgrade(req);
          return new Response(null, { status: 101 });
        }
        return new Response("OK");
      },
      websocket: {
        message(ws, message) {
          // echo back the audio chunk
          ws.send(message);
        },
      },
    });

    const ws = new WebSocket(server.url.toString().replace("http", "ws"));
    ws.binaryType = "arraybuffer";

    await new Promise<void>(resolve => {
      ws.onopen = () => resolve();
    });

    ws.onmessage = e => {
      receivedEchoes.push(Buffer.from(e.data as ArrayBuffer));
      if (receivedEchoes.length === expectedChunkCount) {
        resolveEchoes();
      }
    };

    // stream chunks
    for (let offset = 0; offset < testBuffer.length; offset += CHUNK_SIZE) {
      const chunk = testBuffer.subarray(offset, Math.min(offset + CHUNK_SIZE, testBuffer.length));
      ws.send(chunk);
    }

    await allEchoesReceived;

    // verify echoed data matches original
    const reassembled = Buffer.concat(receivedEchoes);
    expect(reassembled.length).toBe(testBuffer.length);
    expect(reassembled.equals(testBuffer)).toBe(true);

    ws.close();
  });

  // tests backpressure handling with large messages
  test("should handle backpressure with large messages", async () => {
    let messagesReceived = 0;
    const { promise: done, resolve: resolveDone } = Promise.withResolvers<void>();

    using server = Bun.serve({
      port: 0,
      fetch(req, server) {
        if (req.headers.get("upgrade") === "websocket") {
          server.upgrade(req);
          return new Response(null, { status: 101 });
        }
        return new Response("OK");
      },
      websocket: {
        async message(ws, message) {
          messagesReceived++;
          // simulate slow processing
          await Bun.sleep(10);
          if (messagesReceived >= 10) {
            resolveDone();
          }
        },
        backpressureLimit: 1024 * 1024,
        closeOnBackpressureLimit: false,
      },
    });

    const ws = new WebSocket(server.url.toString().replace("http", "ws"));
    await new Promise<void>(resolve => {
      ws.onopen = () => resolve();
    });

    // send multiple large messages rapidly
    const largeChunk = Buffer.alloc(100 * 1024, "x");
    for (let i = 0; i < 10; i++) {
      ws.send(largeChunk);
    }

    await done;
    expect(messagesReceived).toBe(10);

    ws.close();
  });

  // tests concurrent WebSocket connections
  test("should handle multiple concurrent connections", async () => {
    const connections = new Map<number, { messages: string[] }>();
    let nextId = 0;

    using server = Bun.serve<{ id: number }>({
      port: 0,
      fetch(req, server) {
        if (req.headers.get("upgrade") === "websocket") {
          const id = nextId++;
          server.upgrade(req, { data: { id } });
          return new Response(null, { status: 101 });
        }
        return new Response("OK");
      },
      websocket: {
        open(ws) {
          connections.set(ws.data.id, { messages: [] });
        },
        message(ws, message) {
          const conn = connections.get(ws.data.id);
          if (conn) {
            conn.messages.push(typeof message === "string" ? message : "binary");
            ws.send(`ack:${ws.data.id}:${conn.messages.length}`);
          }
        },
        close(ws) {
          connections.delete(ws.data.id);
        },
      },
    });

    // create multiple WebSocket connections
    const ws1 = new WebSocket(server.url.toString().replace("http", "ws"));
    const ws2 = new WebSocket(server.url.toString().replace("http", "ws"));
    const ws3 = new WebSocket(server.url.toString().replace("http", "ws"));

    await Promise.all([
      new Promise<void>(resolve => {
        ws1.onopen = () => resolve();
      }),
      new Promise<void>(resolve => {
        ws2.onopen = () => resolve();
      }),
      new Promise<void>(resolve => {
        ws3.onopen = () => resolve();
      }),
    ]);

    expect(connections.size).toBe(3);

    // send messages to each connection
    const ackPromises = [
      new Promise<string>(resolve => {
        ws1.onmessage = e => resolve(e.data);
      }),
      new Promise<string>(resolve => {
        ws2.onmessage = e => resolve(e.data);
      }),
      new Promise<string>(resolve => {
        ws3.onmessage = e => resolve(e.data);
      }),
    ];

    ws1.send("msg1");
    ws2.send("msg2");
    ws3.send("msg3");

    const acks = await Promise.all(ackPromises);
    expect(acks).toContain("ack:0:1");
    expect(acks).toContain("ack:1:1");
    expect(acks).toContain("ack:2:1");

    // close connections
    ws1.close();
    ws2.close();
    ws3.close();

    await Bun.sleep(50);
    expect(connections.size).toBe(0);
  });

  // tests client disconnect during message processing
  test("should handle client disconnect during message processing", async () => {
    let messageProcessingStarted = false;
    let messageProcessingCompleted = false;
    let connectionClosed = false;
    const { promise: processingStarted, resolve: resolveProcessing } = Promise.withResolvers<void>();

    using server = Bun.serve({
      port: 0,
      fetch(req, server) {
        if (req.headers.get("upgrade") === "websocket") {
          server.upgrade(req);
          return new Response(null, { status: 101 });
        }
        return new Response("OK");
      },
      websocket: {
        async message(ws, message) {
          messageProcessingStarted = true;
          resolveProcessing();

          // simulate long processing (like LLM call)
          await Bun.sleep(500);

          messageProcessingCompleted = true;
          try {
            ws.send("done");
          } catch {
            // client may have disconnected
          }
        },
        close(ws) {
          connectionClosed = true;
        },
      },
    });

    const ws = new WebSocket(server.url.toString().replace("http", "ws"));
    await new Promise<void>(resolve => {
      ws.onopen = () => resolve();
    });

    ws.send("start-long-process");

    // wait for processing to start, then close connection
    await processingStarted;
    expect(messageProcessingStarted).toBe(true);

    ws.close();

    // wait for server to finish processing
    await Bun.sleep(600);

    expect(connectionClosed).toBe(true);
    // message processing should complete even after client disconnects
    expect(messageProcessingCompleted).toBe(true);
  });
});
