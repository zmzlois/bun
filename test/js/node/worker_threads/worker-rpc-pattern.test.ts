/**
 * Tests for common worker_threads RPC patterns:
 * - Same-file worker pattern (isMainThread check)
 * - Promise-based request/response with message IDs
 * - Buffer/Uint8Array transfer between threads
 * - Worker proxy class wrapping worker communication
 */

import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { isMainThread, parentPort, Worker } from "worker_threads";

/**
 * Worker code - runs when this file is loaded in a worker thread.
 * Implements a simple calculator with async methods.
 */
if (!isMainThread && parentPort) {
  let state = { value: 0 };

  parentPort.on("message", async (msg: { id: number; method: string; args: unknown[] }) => {
    const { id, method, args } = msg;

    try {
      let result: unknown;

      switch (method) {
        case "add":
          state.value += args[0] as number;
          result = state.value;
          break;

        case "getValue":
          result = state.value;
          break;

        case "reset":
          state.value = 0;
          result = true;
          break;

        case "processBuffer": {
          // receives a buffer, processes it, returns modified buffer
          const input = args[0] as Buffer;
          const output = Buffer.alloc(input.length);
          for (let i = 0; i < input.length; i++) {
            output[i] = input[i]! * 2;
          }
          result = output;
          break;
        }

        case "asyncOperation": {
          // simulates async work
          await Bun.sleep(args[0] as number);
          result = "done";
          break;
        }

        case "throwError":
          throw new Error(args[0] as string);

        default:
          throw new Error(`Unknown method: ${method}`);
      }

      parentPort!.postMessage({ id, result });
    } catch (err: unknown) {
      parentPort!.postMessage({
        id,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  });
}

/**
 * Proxy class that wraps worker communication with promises.
 * Similar to AudioTracksThreaded pattern.
 */
class WorkerProxy {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
  private destroyed = false;

  private constructor() {
    // spawn worker using this same file
    this.worker = new Worker(__filename);

    this.worker.on("message", (msg: { id: number; result?: unknown; error?: string }) => {
      const { id, result, error } = msg;
      const promise = this.pending.get(id);
      if (!promise) return;

      this.pending.delete(id);

      if (error) {
        promise.reject(new Error(error));
      } else {
        promise.resolve(result);
      }
    });

    this.worker.on("error", err => {
      // reject all pending promises on worker error
      for (const [id, promise] of this.pending) {
        promise.reject(err);
        this.pending.delete(id);
      }
    });
  }

  static async create(): Promise<WorkerProxy> {
    return new WorkerProxy();
  }

  private call(method: string, ...args: unknown[]): Promise<unknown> {
    if (this.destroyed) {
      return Promise.reject(new Error("Worker has been destroyed"));
    }

    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, method, args });
    });
  }

  async add(n: number): Promise<number> {
    return this.call("add", n) as Promise<number>;
  }

  async getValue(): Promise<number> {
    return this.call("getValue") as Promise<number>;
  }

  async reset(): Promise<boolean> {
    return this.call("reset") as Promise<boolean>;
  }

  async processBuffer(buf: Buffer): Promise<Buffer> {
    return this.call("processBuffer", buf) as Promise<Buffer>;
  }

  async asyncOperation(delayMs: number): Promise<string> {
    return this.call("asyncOperation", delayMs) as Promise<string>;
  }

  async throwError(message: string): Promise<never> {
    return this.call("throwError", message) as Promise<never>;
  }

  async destroy(): Promise<void> {
    if (this.destroyed) return;
    this.destroyed = true;
    await this.worker.terminate();
  }
}

// only run tests on main thread
if (isMainThread) {
  describe("worker_threads RPC pattern", () => {
    let proxy: WorkerProxy;

    beforeEach(async () => {
      proxy = await WorkerProxy.create();
    });

    afterEach(async () => {
      await proxy.destroy();
    });

    test("basic method call", async () => {
      const result = await proxy.getValue();
      expect(result).toBe(0);
    });

    test("stateful operations", async () => {
      await proxy.add(5);
      await proxy.add(3);
      const value = await proxy.getValue();
      expect(value).toBe(8);
    });

    test("reset state", async () => {
      await proxy.add(10);
      await proxy.reset();
      const value = await proxy.getValue();
      expect(value).toBe(0);
    });

    test("handles errors from worker", async () => {
      await expect(proxy.throwError("test error")).rejects.toThrow("test error");
    });

    test("async operations in worker", async () => {
      const result = await proxy.asyncOperation(10);
      expect(result).toBe("done");
    });

    test("concurrent calls", async () => {
      const results = await Promise.all([proxy.add(1), proxy.add(2), proxy.add(3)]);
      // all calls should complete, final value should be 6
      const value = await proxy.getValue();
      expect(value).toBe(6);
    });
  });

  describe("worker_threads Buffer transfer", () => {
    let proxy: WorkerProxy;

    beforeEach(async () => {
      proxy = await WorkerProxy.create();
    });

    afterEach(async () => {
      await proxy.destroy();
    });

    test("Buffer roundtrip", async () => {
      const input = Buffer.from([1, 2, 3, 4, 5]);
      const output = await proxy.processBuffer(input);

      expect(output).toBeInstanceOf(Uint8Array);
      expect(output.length).toBe(5);
      expect([...output]).toEqual([2, 4, 6, 8, 10]);
    });

    test("large Buffer transfer", async () => {
      const size = 1024 * 100; // 100KB
      const input = Buffer.alloc(size);
      for (let i = 0; i < size; i++) {
        input[i] = i % 128; // keep values small so doubling doesn't overflow
      }

      const output = await proxy.processBuffer(input);

      expect(output.length).toBe(size);
      for (let i = 0; i < size; i++) {
        expect(output[i]).toBe((i % 128) * 2);
      }
    });

    test("multiple Buffer transfers", async () => {
      const buffers = [Buffer.from([1]), Buffer.from([2, 3]), Buffer.from([4, 5, 6])];

      const results = await Promise.all(buffers.map(b => proxy.processBuffer(b)));

      expect([...results[0]!]).toEqual([2]);
      expect([...results[1]!]).toEqual([4, 6]);
      expect([...results[2]!]).toEqual([8, 10, 12]);
    });
  });

  describe("worker_threads proxy lifecycle", () => {
    test("calls after destroy should reject", async () => {
      const proxy = await WorkerProxy.create();
      await proxy.destroy();

      await expect(proxy.getValue()).rejects.toThrow("Worker has been destroyed");
    });

    test("multiple destroy calls are safe", async () => {
      const proxy = await WorkerProxy.create();
      await proxy.destroy();
      await proxy.destroy(); // should not throw
    });

    test("can create multiple proxies", async () => {
      const proxy1 = await WorkerProxy.create();
      const proxy2 = await WorkerProxy.create();

      await proxy1.add(5);
      await proxy2.add(10);

      expect(await proxy1.getValue()).toBe(5);
      expect(await proxy2.getValue()).toBe(10);

      await proxy1.destroy();
      await proxy2.destroy();
    });
  });
}
