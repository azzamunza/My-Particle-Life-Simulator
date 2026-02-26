import constructSrc from "./construct.wgsl?raw";
import simSrc from "./sim.wgsl?raw";
import { COLOUR_COUNT, MATRIX_SIZE, PARTICLE_COUNT } from "../options";

// Bytes per particle: vec4<f32> = 4 floats × 4 bytes
const PARTICLE_STRIDE = 16;

export class LinkedListEngine {
  private device: GPUDevice;
  private simBuffer: GPUBuffer;
  private matrixBuffer: GPUBuffer;
  private colourBuffer: GPUBuffer;

  // Ping-pong particle buffers
  readonly particleBuffers: [GPUBuffer, GPUBuffer];

  // Spatial acceleration
  private headsBuffer!: GPUBuffer;
  private headsInitBuffer!: GPUBuffer; // filled with -1, used for fast reset
  private linkedBuffer!: GPUBuffer;

  // Pipelines
  private constructPipeline!: GPUComputePipeline;
  private simPipeline!: GPUComputePipeline;

  // Bind groups (two sets, one per ping-pong direction)
  private constructBindGroups!: [GPUBindGroup, GPUBindGroup];
  private simBindGroups!: [GPUBindGroup, GPUBindGroup];

  private step = 0; // current ping-pong index

  constructor(
    device: GPUDevice,
    simBuffer: GPUBuffer,
    matrixBuffer: GPUBuffer,
    colourBuffer: GPUBuffer,
    particleBufferA: GPUBuffer,
    particleBufferB: GPUBuffer
  ) {
    this.device = device;
    this.simBuffer = simBuffer;
    this.matrixBuffer = matrixBuffer;
    this.colourBuffer = colourBuffer;
    this.particleBuffers = [particleBufferA, particleBufferB];

    this.createAccelerationBuffers();
    this.createPipelines();
    this.createBindGroups();
  }

  // -------------------------------------------------------------------------
  // Buffer creation
  // -------------------------------------------------------------------------
  private createAccelerationBuffers(): void {
    // Grid dimensions: assume sim r=15, worldSize=6 for initial sizing.
    // We over-allocate to accommodate any runtime parameter changes.
    // grid cells = ceil(worldSize*2 / r) + 1 per axis; use worst case r=1
    const maxDim = Math.ceil(6 * 2 * 2) + 2; // r=0.5 worst case → generous
    const maxCells = maxDim * maxDim * 4; // extra headroom

    const headsBytes = maxCells * 4; // one i32 per cell
    const linkedBytes = PARTICLE_COUNT * 4; // one i32 per particle

    this.headsBuffer = this.device.createBuffer({
      label: "heads",
      size: headsBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Initialize all entries to -1 (empty sentinel)
    const initData = new Int32Array(maxCells).fill(-1);
    this.headsInitBuffer = this.device.createBuffer({
      label: "headsInit",
      size: headsBytes,
      usage: GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Int32Array(this.headsInitBuffer.getMappedRange()).set(initData);
    this.headsInitBuffer.unmap();

    this.linkedBuffer = this.device.createBuffer({
      label: "linked",
      size: linkedBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  private createPipelines(): void {
    const constructModule = this.device.createShaderModule({ code: constructSrc });
    const simModule = this.device.createShaderModule({ code: simSrc });

    this.constructPipeline = this.device.createComputePipeline({
      label: "construct",
      layout: "auto",
      compute: { module: constructModule, entryPoint: "main" },
    });

    this.simPipeline = this.device.createComputePipeline({
      label: "sim",
      layout: "auto",
      compute: { module: simModule, entryPoint: "main" },
    });
  }

  private createBindGroups(): void {
    const cLayout = this.constructPipeline.getBindGroupLayout(0);
    const sLayout = this.simPipeline.getBindGroupLayout(0);

    const makeConstructBG = (particlesIn: GPUBuffer): GPUBindGroup =>
      this.device.createBindGroup({
        layout: cLayout,
        entries: [
          { binding: 0, resource: { buffer: this.simBuffer } },
          { binding: 1, resource: { buffer: particlesIn } },
          { binding: 2, resource: { buffer: this.headsBuffer } },
          { binding: 3, resource: { buffer: this.linkedBuffer } },
        ],
      });

    const makeSimBG = (particlesIn: GPUBuffer, particlesOut: GPUBuffer): GPUBindGroup =>
      this.device.createBindGroup({
        layout: sLayout,
        entries: [
          { binding: 0, resource: { buffer: this.simBuffer } },
          { binding: 1, resource: { buffer: particlesIn } },
          { binding: 2, resource: { buffer: particlesOut } },
          { binding: 3, resource: { buffer: this.matrixBuffer } },
          { binding: 4, resource: { buffer: this.headsBuffer } },
          { binding: 5, resource: { buffer: this.linkedBuffer } },
          { binding: 6, resource: { buffer: this.colourBuffer } },
        ],
      });

    this.constructBindGroups = [
      makeConstructBG(this.particleBuffers[0]),
      makeConstructBG(this.particleBuffers[1]),
    ];

    this.simBindGroups = [
      makeSimBG(this.particleBuffers[0], this.particleBuffers[1]),
      makeSimBG(this.particleBuffers[1], this.particleBuffers[0]),
    ];
  }

  // -------------------------------------------------------------------------
  // Per-frame tick
  // -------------------------------------------------------------------------
  tick(encoder: GPUCommandEncoder): void {
    const s = this.step;

    // 1. Reset heads buffer to -1
    encoder.copyBufferToBuffer(
      this.headsInitBuffer,
      0,
      this.headsBuffer,
      0,
      this.headsInitBuffer.size
    );

    // 2. Construct pass – build spatial linked list from current particles
    {
      const pass = encoder.beginComputePass({ label: "construct" });
      pass.setPipeline(this.constructPipeline);
      pass.setBindGroup(0, this.constructBindGroups[s]);
      pass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / 64));
      pass.end();
    }

    // 3. Sim pass – integrate forces, write to the other buffer
    {
      const pass = encoder.beginComputePass({ label: "sim" });
      pass.setPipeline(this.simPipeline);
      pass.setBindGroup(0, this.simBindGroups[s]);
      pass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / 64));
      pass.end();
    }

    // Advance ping-pong
    this.step = 1 - s;
  }

  /** Index of the buffer that holds the most recently written particle data */
  get currentBufferIndex(): number {
    // After tick(), step was flipped; the "output" of the last tick is
    // the buffer that was written to, i.e. the one NOT currently at this.step.
    return 1 - this.step;
  }
}
