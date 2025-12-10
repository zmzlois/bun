/**
 * Tests for onnxruntime-node compatibility with Bun.
 *
 * These tests verify:
 * - Tensor creation and disposal (memory management)
 * - InferenceSession lifecycle
 * - Concurrent session usage
 * - Memory leak detection under load
 *
 * Common issues found in production (VAD service):
 * - Tensor.dispose() not properly freeing native memory
 * - Session release/create cycles leaking memory
 * - Race conditions in concurrent inference
 * - State tensor updates causing memory growth
 */

import { describe, test, expect, beforeAll } from "bun:test";
import * as ort from "onnxruntime-node";
import { join } from "path";
import { existsSync, mkdirSync, writeFileSync } from "fs";

// directory to store test models
const MODELS_DIR = join(import.meta.dir, "models");
const SIMPLE_MODEL_PATH = join(MODELS_DIR, "simple_add.onnx");

/**
 * Creates a minimal ONNX model that performs: output = input + 1
 * This is created using raw protobuf bytes for a valid ONNX model.
 */
// function createSimpleAddModel(): Uint8Array {
//   // this is a minimal valid ONNX model (ir_version=8, opset=13)
//   // graph: output = Add(input, ones)
//   // input: float32[1, 4]
//   // output: float32[1, 4]
//   return new Uint8Array([
//     // ONNX file header
//     0x08,
//     0x08, // ir_version = 8
//     0x12,
//     0x08, // producer_name (8 bytes)
//     0x62,
//     0x75,
//     0x6e,
//     0x2d,
//     0x74,
//     0x65,
//     0x73,
//     0x74,
//     0x1a,
//     0x85,
//     0x01, // graph field (133 bytes)
//
//     // graph.node[0]: Add
//     0x0a,
//     0x1a, // node (26 bytes)
//     0x0a,
//     0x05,
//     0x69,
//     0x6e,
//     0x70,
//     0x75,
//     0x74, // input: "input"
//     0x0a,
//     0x04,
//     0x6f,
//     0x6e,
//     0x65,
//     0x73, // input: "ones"
//     0x12,
//     0x06,
//     0x6f,
//     0x75,
//     0x74,
//     0x70,
//     0x75,
//     0x74, // output: "output"
//     0x22,
//     0x03,
//     0x41,
//     0x64,
//     0x64, // op_type: "Add"
//
//     // graph.name
//     0x12,
//     0x0a, // name (10 bytes)
//     0x74,
//     0x65,
//     0x73,
//     0x74,
//     0x5f,
//     0x67,
//     0x72,
//     0x61,
//     0x70,
//     0x68, // "test_graph"
//
//     // graph.initializer[0]: ones (constant tensor)
//     0x22,
//     0x24, // initializer (36 bytes)
//     0x0a,
//     0x04,
//     0x6f,
//     0x6e,
//     0x65,
//     0x73, // name: "ones"
//     0x10,
//     0x01, // data_type: FLOAT (1)
//     0x1a,
//     0x02,
//     0x01,
//     0x04, // dims: [1, 4]
//     0x2a,
//     0x10, // float_data (16 bytes = 4 floats)
//     0x00,
//     0x00,
//     0x80,
//     0x3f, // 1.0f
//     0x00,
//     0x00,
//     0x80,
//     0x3f, // 1.0f
//     0x00,
//     0x00,
//     0x80,
//     0x3f, // 1.0f
//     0x00,
//     0x00,
//     0x80,
//     0x3f, // 1.0f
//
//     // graph.input[0]
//     0x32,
//     0x19, // input (25 bytes)
//     0x0a,
//     0x17, // ValueInfoProto
//     0x0a,
//     0x05,
//     0x69,
//     0x6e,
//     0x70,
//     0x75,
//     0x74, // name: "input"
//     0x12,
//     0x0e, // type
//     0x0a,
//     0x0c, // tensor_type
//     0x08,
//     0x01, // elem_type: FLOAT
//     0x12,
//     0x06, // shape
//     0x0a,
//     0x02,
//     0x08,
//     0x01, // dim: 1
//     0x0a,
//     0x02,
//     0x08,
//     0x04, // dim: 4
//
//     // graph.output[0]
//     0x3a,
//     0x1a, // output (26 bytes)
//     0x0a,
//     0x18, // ValueInfoProto
//     0x0a,
//     0x06,
//     0x6f,
//     0x75,
//     0x74,
//     0x70,
//     0x75,
//     0x74, // name: "output"
//     0x12,
//     0x0e, // type
//     0x0a,
//     0x0c, // tensor_type
//     0x08,
//     0x01, // elem_type: FLOAT
//     0x12,
//     0x06, // shape
//     0x0a,
//     0x02,
//     0x08,
//     0x01, // dim: 1
//     0x0a,
//     0x02,
//     0x08,
//     0x04, // dim: 4
//
//     // opset_import
//     0x40,
//     0x0d, // opset version 13
//   ]);
// }
//
/**
 * Downloads a real, small ONNX model for testing.
 * Using the MNIST model which is tiny (~26KB).
 */
async function downloadTestModel(): Promise<string> {
  const modelUrl = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx";
  const modelPath = join(MODELS_DIR, "mnist.onnx");

  if (existsSync(modelPath)) {
    return modelPath;
  }

  if (!existsSync(MODELS_DIR)) {
    mkdirSync(MODELS_DIR, { recursive: true });
  }

  console.log("Downloading MNIST test model...");
  const response = await fetch(modelUrl);
  if (!response.ok) {
    throw new Error(`Failed to download model: ${response.status}`);
  }

  const buffer = await response.arrayBuffer();
  writeFileSync(modelPath, Buffer.from(buffer));
  console.log(`Model saved to ${modelPath}`);

  return modelPath;
}

// ============================================================================
// TENSOR TESTS
// ============================================================================

describe("onnxruntime-node Tensor", () => {
  test("can create float32 tensor", () => {
    const data = new Float32Array([1, 2, 3, 4]);
    const tensor = new ort.Tensor("float32", data, [2, 2]);

    expect(tensor.type).toBe("float32");
    expect(tensor.dims).toEqual([2, 2]);
    expect(tensor.size).toBe(4);
    expect([...(tensor.data as Float32Array)]).toEqual([1, 2, 3, 4]);

    tensor.dispose();
  });

  test("can create int64 tensor", () => {
    const data = BigInt64Array.from([1n, 2n, 3n]);
    const tensor = new ort.Tensor("int64", data, [3]);

    expect(tensor.type).toBe("int64");
    expect(tensor.dims).toEqual([3]);

    tensor.dispose();
  });

  test("can create tensor with various shapes", () => {
    const shapes = [[1], [10], [2, 3], [2, 3, 4], [1, 128], [2, 1, 128]];

    for (const shape of shapes) {
      const size = shape.reduce((a, b) => a * b, 1);
      const data = new Float32Array(size);
      const tensor = new ort.Tensor("float32", data, shape);

      expect(tensor.dims).toEqual(shape);
      expect(tensor.size).toBe(size);

      tensor.dispose();
    }
  });

  test("dispose is idempotent", () => {
    const tensor = new ort.Tensor("float32", new Float32Array([1, 2, 3]), [3]);

    // multiple dispose calls should not throw
    tensor.dispose();
    tensor.dispose();
    tensor.dispose();
  });

  test("tensor creation/disposal cycle (memory leak check)", () => {
    // create and dispose many tensors to check for memory leaks
    const iterations = 1000;

    for (let i = 0; i < iterations; i++) {
      const data = new Float32Array(512); // 2KB per tensor
      const tensor = new ort.Tensor("float32", data, [1, 512]);
      tensor.dispose();
    }

    // if we get here without OOM, the test passes
    expect(true).toBe(true);
  });

  test("concurrent tensor creation", async () => {
    const createTensor = () => {
      const data = new Float32Array(128);
      for (let i = 0; i < 128; i++) data[i] = Math.random();
      return new ort.Tensor("float32", data, [1, 128]);
    };

    // create many tensors concurrently
    const tensors = await Promise.all(Array.from({ length: 100 }, () => Promise.resolve(createTensor())));

    expect(tensors.length).toBe(100);

    // dispose all
    for (const tensor of tensors) {
      tensor.dispose();
    }
  });
});

// ============================================================================
// INFERENCE SESSION TESTS
// ============================================================================

describe("onnxruntime-node InferenceSession", () => {
  let modelPath: string;

  beforeAll(async () => {
    // download a real model for testing
    modelPath = await downloadTestModel();
  });

  test("can create session from file path", async () => {
    const session = await ort.InferenceSession.create(modelPath);

    expect(session).toBeDefined();
    expect(session.inputNames).toBeDefined();
    expect(session.inputNames.length).toBeGreaterThan(0);
    expect(session.outputNames).toBeDefined();
    expect(session.outputNames.length).toBeGreaterThan(0);

    await session.release();
  });

  test("can create session with options (VAD-style)", async () => {
    // these are the options used in VAD's BaseOnnxModel
    const session = await ort.InferenceSession.create(modelPath, {
      graphOptimizationLevel: "all",
      enableCpuMemArena: false,
      enableMemPattern: false,
      executionMode: "sequential",
    });

    expect(session).toBeDefined();
    await session.release();
  });

  test("session double release throws", async () => {
    const session = await ort.InferenceSession.create(modelPath);

    // first release should succeed
    await session.release();

    // second release throws "Session already disposed"
    await expect(session.release()).rejects.toThrow("Session already disposed");
  });

  test("session creation/release cycle (memory leak check)", async () => {
    const iterations = 20;

    for (let i = 0; i < iterations; i++) {
      const session = await ort.InferenceSession.create(modelPath, {
        graphOptimizationLevel: "all",
        enableCpuMemArena: false,
        enableMemPattern: false,
        executionMode: "sequential",
      });
      await session.release();
    }

    expect(true).toBe(true);
  });

  test("can run inference", async () => {
    const session = await ort.InferenceSession.create(modelPath);

    // MNIST expects input shape [1, 1, 28, 28]
    const inputName = session.inputNames[0]!;
    const inputData = new Float32Array(1 * 1 * 28 * 28).fill(0);
    const inputTensor = new ort.Tensor("float32", inputData, [1, 1, 28, 28]);

    const feeds: Record<string, ort.Tensor> = {};
    feeds[inputName] = inputTensor;

    const results = await session.run(feeds);

    expect(results).toBeDefined();
    expect(Object.keys(results).length).toBeGreaterThan(0);

    // cleanup
    inputTensor.dispose();
    for (const key of Object.keys(results)) {
      results[key]!.dispose();
    }
    await session.release();
  });

  test("repeated inference cycles (memory check)", async () => {
    const session = await ort.InferenceSession.create(modelPath, {
      graphOptimizationLevel: "all",
      enableCpuMemArena: false,
      enableMemPattern: false,
      executionMode: "sequential",
    });

    const inputName = session.inputNames[0]!;
    const iterations = 100;

    for (let i = 0; i < iterations; i++) {
      const inputData = new Float32Array(1 * 1 * 28 * 28);
      // fill with some data
      for (let j = 0; j < inputData.length; j++) {
        inputData[j] = Math.random();
      }

      const inputTensor = new ort.Tensor("float32", inputData, [1, 1, 28, 28]);
      const feeds: Record<string, ort.Tensor> = {};
      feeds[inputName] = inputTensor;

      const results = await session.run(feeds);

      // dispose all tensors
      inputTensor.dispose();
      for (const key of Object.keys(results)) {
        results[key]!.dispose();
      }
    }

    await session.release();
    expect(true).toBe(true);
  });
});

// ============================================================================
// VAD-STYLE PATTERNS
// ============================================================================

describe("onnxruntime-node VAD-style patterns", () => {
  let modelPath: string;

  beforeAll(async () => {
    modelPath = await downloadTestModel();
  });

  test("state tensor update pattern (like Silero VAD)", async () => {
    // simulates the VAD pattern where state tensor is updated each frame
    // this is the exact pattern from silero.ts that can cause memory issues
    const batchSize = 1;
    const stateSize = 128;

    let state = new ort.Tensor("float32", new Float32Array(2 * batchSize * stateSize), [2, batchSize, stateSize]);

    const iterations = 100;

    for (let i = 0; i < iterations; i++) {
      // create new state (simulating inference output)
      const newState = new ort.Tensor("float32", new Float32Array(2 * batchSize * stateSize), [
        2,
        batchSize,
        stateSize,
      ]);

      // CRITICAL: dispose old state BEFORE replacing reference
      // this is the pattern from silero.ts lines 140-145
      const oldState = state;
      state = newState;
      oldState.dispose();
    }

    // cleanup final state
    state.dispose();

    expect(true).toBe(true);
  });

  test("full inference cycle with tensor lifecycle (VAD pattern)", async () => {
    const session = await ort.InferenceSession.create(modelPath, {
      graphOptimizationLevel: "all",
      enableCpuMemArena: false,
      enableMemPattern: false,
      executionMode: "sequential",
    });

    const inputName = session.inputNames[0]!;
    const iterations = 50;

    for (let i = 0; i < iterations; i++) {
      // create input tensor (like silero.ts line 112)
      const inputTensor = new ort.Tensor("float32", new Float32Array(1 * 1 * 28 * 28), [1, 1, 28, 28]);

      // create sample rate tensor (like silero.ts line 113)
      const srTensor = new ort.Tensor("int64", BigInt64Array.from([16000n]), [1]);

      const feeds: Record<string, ort.Tensor> = {};
      feeds[inputName] = inputTensor;

      // run inference (like silero.ts line 116)
      const results = await session.run(feeds);

      // dispose in finally block pattern (silero.ts lines 166-205)
      inputTensor.dispose();
      srTensor.dispose();
      for (const key of Object.keys(results)) {
        results[key]!.dispose();
      }
    }

    await session.release();
    expect(true).toBe(true);
  });

  test("concurrent inference streams (multiple VAD clients)", async () => {
    const session = await ort.InferenceSession.create(modelPath, {
      graphOptimizationLevel: "all",
      enableCpuMemArena: false,
      enableMemPattern: false,
      executionMode: "sequential",
    });

    const inputName = session.inputNames[0]!;
    const numStreams = 5;
    const framesPerStream = 20;

    // simulate semaphore with a simple lock
    let locked = false;
    const acquire = async () => {
      while (locked) {
        await Bun.sleep(1);
      }
      locked = true;
    };
    const release = () => {
      locked = false;
    };

    const runStream = async (streamId: number) => {
      // each stream has its own state tensor (like RealtimeVAD.silero)
      let state = new ort.Tensor("float32", new Float32Array(256), [2, 1, 128]);

      for (let frame = 0; frame < framesPerStream; frame++) {
        const inputTensor = new ort.Tensor("float32", new Float32Array(1 * 1 * 28 * 28), [1, 1, 28, 28]);

        // acquire semaphore before session.run (like silero.ts line 105)
        await acquire();

        try {
          const feeds: Record<string, ort.Tensor> = {};
          feeds[inputName] = inputTensor;

          const results = await session.run(feeds);

          // simulate state update
          const newState = new ort.Tensor("float32", new Float32Array(256), [2, 1, 128]);
          const oldState = state;
          state = newState;
          oldState.dispose();

          // dispose results
          for (const key of Object.keys(results)) {
            results[key]!.dispose();
          }
        } finally {
          // release semaphore AFTER tensor cleanup (silero.ts line 203)
          release();
        }

        inputTensor.dispose();
      }

      state.dispose();
    };

    // run all streams concurrently
    await Promise.all(Array.from({ length: numStreams }, (_, i) => runStream(i)));

    await session.release();
    expect(true).toBe(true);
  });

  test("session singleton pattern with re-initialization", async () => {
    // simulates BaseOnnxModel.getSession() pattern
    let session: ort.InferenceSession | null = null;
    let released = false;

    const getSession = async () => {
      if (session && !released) {
        return session;
      }
      session = await ort.InferenceSession.create(modelPath, {
        graphOptimizationLevel: "all",
        enableCpuMemArena: false,
        enableMemPattern: false,
        executionMode: "sequential",
      });
      released = false;
      return session;
    };

    const releaseSession = async () => {
      if (session && !released) {
        await session.release();
        released = true;
      }
    };

    // simulate multiple getSession calls
    const s1 = await getSession();
    const s2 = await getSession();
    expect(s1).toBe(s2);

    // release and re-get
    await releaseSession();
    const s3 = await getSession();
    expect(s3).not.toBe(s1);

    await releaseSession();
  });

  test("rapid destroy/recreate cycle (like WebSocket close/reopen)", async () => {
    // simulates rapid client connections/disconnections
    const cycles = 30;

    for (let i = 0; i < cycles; i++) {
      // create "client" with state
      let state = new ort.Tensor("float32", new Float32Array(256), [2, 1, 128]);

      // simulate a few frames
      for (let frame = 0; frame < 5; frame++) {
        const newState = new ort.Tensor("float32", new Float32Array(256), [2, 1, 128]);
        const oldState = state;
        state = newState;
        oldState.dispose();
      }

      // destroy "client"
      state.dispose();
    }

    expect(true).toBe(true);
  });
});

// ============================================================================
// MEMORY STRESS TESTS
// ============================================================================

describe("onnxruntime-node memory stress", () => {
  let modelPath: string;

  beforeAll(async () => {
    modelPath = await downloadTestModel();
  });

  test("rapid tensor allocation stress test", () => {
    const tensors: ort.Tensor[] = [];
    const maxTensors = 500;

    // allocate many tensors
    for (let i = 0; i < maxTensors; i++) {
      tensors.push(new ort.Tensor("float32", new Float32Array(1024), [1, 1024]));
    }

    // dispose in reverse order
    while (tensors.length > 0) {
      tensors.pop()!.dispose();
    }

    // allocate again to ensure memory was freed
    for (let i = 0; i < maxTensors; i++) {
      tensors.push(new ort.Tensor("float32", new Float32Array(1024), [1, 1024]));
    }

    // cleanup
    for (const t of tensors) {
      t.dispose();
    }

    expect(true).toBe(true);
  });

  test("large tensor allocation", () => {
    // test with larger tensors (closer to real audio buffers)
    const iterations = 50;

    for (let i = 0; i < iterations; i++) {
      // ~1MB tensor
      const tensor = new ort.Tensor("float32", new Float32Array(256 * 1024), [1, 256 * 1024]);
      tensor.dispose();
    }

    expect(true).toBe(true);
  });

  test("mixed tensor sizes", () => {
    const iterations = 100;
    const sizes = [64, 256, 1024, 4096, 16384];

    for (let i = 0; i < iterations; i++) {
      const size = sizes[i % sizes.length]!;
      const tensor = new ort.Tensor("float32", new Float32Array(size), [1, size]);
      tensor.dispose();
    }

    expect(true).toBe(true);
  });
});

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

describe("onnxruntime-node error handling", () => {
  test("invalid tensor type throws", () => {
    expect(() => {
      // @ts-expect-error testing invalid type
      new ort.Tensor("invalid_type", new Float32Array([1]), [1]);
    }).toThrow();
  });

  test("mismatched data size throws", () => {
    expect(() => {
      // data size (3) doesn't match shape (2x2 = 4)
      new ort.Tensor("float32", new Float32Array([1, 2, 3]), [2, 2]);
    }).toThrow();
  });

  test("negative dimensions throw", () => {
    expect(() => {
      new ort.Tensor("float32", new Float32Array([1]), [-1]);
    }).toThrow();
  });

  test("invalid model path throws", async () => {
    await expect(ort.InferenceSession.create("/nonexistent/model.onnx")).rejects.toThrow();
  });
});
