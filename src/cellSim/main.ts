import brownianSrc    from "./passes/brownian.wgsl?raw";
import bondsSrc       from "./passes/bonds.wgsl?raw";
import pressureSrc    from "./passes/pressure.wgsl?raw";
import xpbdSrc        from "./passes/xpbd.wgsl?raw";
import integrateSrc   from "./passes/integrate.wgsl?raw";
import gridSrc        from "./passes/grid.wgsl?raw";
import interactSrc    from "./passes/interact.wgsl?raw";
import transitionsSrc from "./passes/transitions.wgsl?raw";
import renderSrc      from "./render.wgsl?raw";

import { Pane } from "tweakpane";
import { initScene } from "./init";
import {
  MAX_PARTICLES, MAX_BONDS, XPBD_ITERS,
  DT, BROWNIAN_STR, DAMPING, PRESSURE_K,
  MEMBRANE_K, NUCLEUS_K, CILIA_WAVE_STRENGTH, FLAGELLUM_WAVE_STRENGTH,
  CHANNEL_RADIUS, CYTOPLASM_LIFE,
  PARTICLE_STRIDE, BOND_STRIDE,
  GRID_CELLS,
  FORCE_FP_SCALE,
  U_TICK, U_NUM_PARTICLES, U_NUM_BONDS,
  U_DT, U_BROWNIAN_STR, U_DAMPING, U_PRESSURE_K,
  U_MEMBRANE_K, U_NUCLEUS_K, U_CILIA_WAVE, U_FLAG_WAVE,
  U_CHANNEL_R, U_CYTOPLASM_LIFE,
  UNIFORM_SIZE,
} from "./buffers";

// ---- Mutable sim params (tweakpane binds to these) ----
interface CellSimParams {
  brownianStr:  number;
  damping:      number;
  xpbdIters:    number;
  membraneK:    number;
  ciliaWave:    number;
  cytoplLife:   number;
  channelR:     number;
  zoom:         number;
  particlePx:   number;
}

// ---------------------------------------------------------------------------
// WebGPU init helper
// ---------------------------------------------------------------------------
async function initGPU(canvas: HTMLCanvasElement): Promise<{
  device:  GPUDevice;
  context: GPUCanvasContext;
  format:  GPUTextureFormat;
}> {
  if (!navigator.gpu) throw new Error("WebGPU not supported.");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter)     throw new Error("No WebGPU adapter found.");
  const device  = await adapter.requestDevice();
  const context = canvas.getContext("webgpu") as GPUCanvasContext;
  const format  = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "premultiplied" });
  return { device, context, format };
}

// ---------------------------------------------------------------------------
// Exported entry point
// ---------------------------------------------------------------------------
export async function startCellSim(
  canvas:      HTMLCanvasElement,
  uiContainer: HTMLElement,
): Promise<() => void> {

  const { device, context, format } = await initGPU(canvas);

  // ---- Mutable params (Tweakpane writes here) ----
  const params: CellSimParams = {
    brownianStr: BROWNIAN_STR,
    damping:     DAMPING,
    xpbdIters:   XPBD_ITERS,
    membraneK:   MEMBRANE_K,
    ciliaWave:   CILIA_WAVE_STRENGTH,
    cytoplLife:  CYTOPLASM_LIFE,
    channelR:    CHANNEL_RADIUS,
    zoom:        Math.min(canvas.width, canvas.height) / (200 * 2),
    particlePx:  4.0,
  };

  // -----------------------------------------------------------------------
  // Scene initialisation
  // -----------------------------------------------------------------------
  const scene = initScene();

  // -----------------------------------------------------------------------
  // GPU Buffers
  // -----------------------------------------------------------------------
  const particleBuffer = device.createBuffer({
    label: "cell particles",
    size:  MAX_PARTICLES * PARTICLE_STRIDE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(particleBuffer, 0, scene.particleData);

  const bondBuffer = device.createBuffer({
    label: "cell bonds",
    size:  MAX_BONDS * BOND_STRIDE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(bondBuffer, 0, scene.bondData);

  // Force accumulation (fixed-point atomic i32, 2 per particle)
  const forceAccumBuffer = device.createBuffer({
    label: "forceAccum",
    size:  MAX_PARTICLES * 2 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // XPBD position-correction accumulation (atomic i32, 2 per particle)
  const xpbdDeltaBuffer = device.createBuffer({
    label: "xpbdDelta",
    size:  MAX_PARTICLES * 2 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Spatial grid: heads (atomic i32 per cell)
  const headsBuffer = device.createBuffer({
    label: "grid heads",
    size:  GRID_CELLS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  // heads init buffer: all -1
  const headsInitData = new Int32Array(GRID_CELLS).fill(-1);
  const headsInitBuffer = device.createBuffer({
    label: "grid headsInit",
    size:  GRID_CELLS * 4,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Int32Array(headsInitBuffer.getMappedRange()).set(headsInitData);
  headsInitBuffer.unmap();

  // Spatial grid: linked list (i32 per particle)
  const linkedBuffer = device.createBuffer({
    label: "grid linked",
    size:  MAX_PARTICLES * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // Free list ring buffer
  const freeListBuffer = device.createBuffer({
    label: "freeList",
    size:  MAX_PARTICLES * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(freeListBuffer, 0, scene.freeList);

  // Free list control: [head, tail] as atomic<u32>[2]
  const freeCtrlBuffer = device.createBuffer({
    label: "freeCtrl",
    size:  8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  // head=0, tail=freeCount
  const freeCtrlData = new Uint32Array([0, scene.freeCount]);
  device.queue.writeBuffer(freeCtrlBuffer, 0, freeCtrlData);

  // Uniform buffer (64 bytes)
  const uniformBuffer = device.createBuffer({
    label: "cell uniforms",
    size:  UNIFORM_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Camera uniform
  const cameraBuffer = device.createBuffer({
    label: "cell camera",
    size:  16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // -----------------------------------------------------------------------
  // Uniform helpers
  // -----------------------------------------------------------------------
  let tick = 0;
  const uniformRaw = new ArrayBuffer(UNIFORM_SIZE);
  const uU32 = new Uint32Array(uniformRaw);
  const uF32 = new Float32Array(uniformRaw);

  function writeUniforms(): void {
    uU32[U_TICK]          = tick;
    uU32[U_NUM_PARTICLES] = MAX_PARTICLES;
    uU32[U_NUM_BONDS]     = scene.bondCount;
    uF32[U_DT]            = DT;
    uF32[U_BROWNIAN_STR]  = params.brownianStr;
    uF32[U_DAMPING]       = params.damping;
    uF32[U_PRESSURE_K]    = PRESSURE_K;
    uF32[U_MEMBRANE_K]    = params.membraneK;
    uF32[U_NUCLEUS_K]     = NUCLEUS_K;
    uF32[U_CILIA_WAVE]    = params.ciliaWave;
    uF32[U_FLAG_WAVE]     = FLAGELLUM_WAVE_STRENGTH;
    uF32[U_CHANNEL_R]     = params.channelR;
    uU32[U_CYTOPLASM_LIFE]= params.cytoplLife;
    device.queue.writeBuffer(uniformBuffer, 0, uniformRaw);
  }
  writeUniforms();

  function writeCamera(): void {
    const dpr   = window.devicePixelRatio || 1;
    const physR = params.particlePx * dpr;
    const clipR = physR * 2.0 / canvas.height;
    device.queue.writeBuffer(cameraBuffer, 0, new Float32Array([
      0, 0,
      params.zoom * 2 / canvas.height,
      clipR,
    ]));
  }
  writeCamera();

  // -----------------------------------------------------------------------
  // Compute pipelines
  // -----------------------------------------------------------------------
  const brownianPipeline = device.createComputePipeline({
    label:   "brownian",
    layout:  "auto",
    compute: { module: device.createShaderModule({ code: brownianSrc }),    entryPoint: "main" },
  });
  const bondsPipeline = device.createComputePipeline({
    label:   "bonds",
    layout:  "auto",
    compute: { module: device.createShaderModule({ code: bondsSrc }),       entryPoint: "main" },
  });
  const pressurePipeline = device.createComputePipeline({
    label:   "pressure",
    layout:  "auto",
    compute: { module: device.createShaderModule({ code: pressureSrc }),    entryPoint: "main" },
  });
  const xpbdModule = device.createShaderModule({ code: xpbdSrc });
  const xpbdAccumPipeline = device.createComputePipeline({
    label:   "xpbd-accum",
    layout:  "auto",
    compute: { module: xpbdModule, entryPoint: "accumulate" },
  });
  const xpbdApplyPipeline = device.createComputePipeline({
    label:   "xpbd-apply",
    layout:  "auto",
    compute: { module: xpbdModule, entryPoint: "apply_corrections" },
  });
  const integratePipeline = device.createComputePipeline({
    label:   "integrate",
    layout:  "auto",
    compute: { module: device.createShaderModule({ code: integrateSrc }),   entryPoint: "main" },
  });
  const gridPipeline = device.createComputePipeline({
    label:   "grid",
    layout:  "auto",
    compute: { module: device.createShaderModule({ code: gridSrc }),        entryPoint: "main" },
  });
  const interactPipeline = device.createComputePipeline({
    label:   "interact",
    layout:  "auto",
    compute: { module: device.createShaderModule({ code: interactSrc }),    entryPoint: "main" },
  });
  const transitionsPipeline = device.createComputePipeline({
    label:   "transitions",
    layout:  "auto",
    compute: { module: device.createShaderModule({ code: transitionsSrc }), entryPoint: "main" },
  });

  // -----------------------------------------------------------------------
  // Bind groups
  // -----------------------------------------------------------------------
  const brownianBG = device.createBindGroup({
    layout: brownianPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
      { binding: 2, resource: { buffer: forceAccumBuffer } },
      { binding: 3, resource: { buffer: headsBuffer } },
      { binding: 4, resource: { buffer: linkedBuffer } },
    ],
  });

  const bondsBG = device.createBindGroup({
    layout: bondsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
      { binding: 2, resource: { buffer: bondBuffer } },
      { binding: 3, resource: { buffer: forceAccumBuffer } },
    ],
  });

  const pressureBG = device.createBindGroup({
    layout: pressurePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
      { binding: 2, resource: { buffer: forceAccumBuffer } },
    ],
  });

  const xpbdAccumBG = device.createBindGroup({
    layout: xpbdAccumPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
      { binding: 2, resource: { buffer: bondBuffer } },
      { binding: 3, resource: { buffer: xpbdDeltaBuffer } },
    ],
  });

  const xpbdApplyBG = device.createBindGroup({
    layout: xpbdApplyPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
      { binding: 3, resource: { buffer: xpbdDeltaBuffer } },
    ],
  });

  const integrateBG = device.createBindGroup({
    layout: integratePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
      { binding: 2, resource: { buffer: forceAccumBuffer } },
    ],
  });

  const gridBG = device.createBindGroup({
    layout: gridPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
      { binding: 2, resource: { buffer: headsBuffer } },
      { binding: 3, resource: { buffer: linkedBuffer } },
    ],
  });

  const interactBG = device.createBindGroup({
    layout: interactPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
      { binding: 2, resource: { buffer: headsBuffer } },
      { binding: 3, resource: { buffer: linkedBuffer } },
      { binding: 4, resource: { buffer: forceAccumBuffer } },
      { binding: 5, resource: { buffer: freeListBuffer } },
      { binding: 6, resource: { buffer: freeCtrlBuffer } },
    ],
  });

  const transitionsBG = device.createBindGroup({
    layout: transitionsPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: particleBuffer } },
      { binding: 2, resource: { buffer: freeListBuffer } },
      { binding: 3, resource: { buffer: freeCtrlBuffer } },
    ],
  });

  // -----------------------------------------------------------------------
  // Render pipeline
  // -----------------------------------------------------------------------
  const renderModule = device.createShaderModule({ code: renderSrc });

  const renderBGL = device.createBindGroupLayout({
    label: "cell render BGL",
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
    ],
  });

  const renderPipeline = device.createRenderPipeline({
    label:  "cell render",
    layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
    vertex:   { module: renderModule, entryPoint: "vs_main" },
    fragment: {
      module: renderModule, entryPoint: "fs_main",
      targets: [{
        format,
        blend: {
          color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
        },
      }],
    },
    primitive: { topology: "triangle-list" },
  });

  const renderBG = device.createBindGroup({
    layout: renderBGL,
    entries: [
      { binding: 0, resource: { buffer: particleBuffer } },
      { binding: 1, resource: { buffer: cameraBuffer } },
    ],
  });

  // -----------------------------------------------------------------------
  // Tweakpane UI
  // -----------------------------------------------------------------------
  const pane = new Pane({ container: uiContainer });

  const simFolder = pane.addFolder({ title: "Simulation" });
  simFolder.addBinding(params, "brownianStr", { min: 0, max: 2,    label: "Brownian Strength" });
  simFolder.addBinding(params, "damping",     { min: 0.9, max: 1.0, label: "Damping" });
  simFolder.addBinding(params, "xpbdIters",   { min: 1, max: 10, step: 1, label: "XPBD Iterations" });
  simFolder.addBinding(params, "membraneK",   { min: 100, max: 2000, label: "Membrane Stiffness" });
  simFolder.addBinding(params, "ciliaWave",   { min: 0, max: 3,    label: "Cilia Wave Strength" });

  const cellFolder = pane.addFolder({ title: "Cell" });
  cellFolder.addBinding(params, "cytoplLife", { min: 100, max: 2000, step: 10, label: "Cytoplasm Lifetime" });
  cellFolder.addBinding(params, "channelR",   { min: 5, max: 50,   label: "Channel Radius" });

  const camFolder = pane.addFolder({ title: "Camera" });
  camFolder.addBinding(params, "zoom",       { min: 0.001, max: 0.05, label: "Zoom" }).on("change", writeCamera);
  camFolder.addBinding(params, "particlePx", { min: 1.0,  max: 20.0, label: "Particle size (px)" }).on("change", writeCamera);

  const btnRandomise = pane.addButton({ title: "Randomise" });
  btnRandomise.on("click", () => {
    const s2 = initScene();
    device.queue.writeBuffer(particleBuffer, 0, s2.particleData);
    device.queue.writeBuffer(bondBuffer,     0, s2.bondData);
    device.queue.writeBuffer(freeListBuffer, 0, s2.freeList);
    device.queue.writeBuffer(freeCtrlBuffer, 0, new Uint32Array([0, s2.freeCount]));
  });

  // -----------------------------------------------------------------------
  // Resize handler
  // -----------------------------------------------------------------------
  const onResize = () => writeCamera();
  window.addEventListener("resize", onResize);

  // -----------------------------------------------------------------------
  // Workgroup counts
  // -----------------------------------------------------------------------
  const particleWG = Math.ceil(MAX_PARTICLES / 64);
  const bondWG     = Math.ceil(MAX_BONDS     / 64);

  // -----------------------------------------------------------------------
  // Frame loop
  // -----------------------------------------------------------------------
  let rafId = 0;

  function frame(): void {
    tick++;
    writeUniforms();

    const encoder = device.createCommandEncoder();

    // 1. Clear forceAccumBuffer to zero
    encoder.clearBuffer(forceAccumBuffer);

    // 2. Brownian motion + wave forces
    {
      const pass = encoder.beginComputePass({ label: "brownian" });
      pass.setPipeline(brownianPipeline);
      pass.setBindGroup(0, brownianBG);
      pass.dispatchWorkgroups(particleWG);
      pass.end();
    }

    // 3. Bond spring forces
    {
      const pass = encoder.beginComputePass({ label: "bonds" });
      pass.setPipeline(bondsPipeline);
      pass.setBindGroup(0, bondsBG);
      pass.dispatchWorkgroups(bondWG);
      pass.end();
    }

    // 4. Radial pressure
    {
      const pass = encoder.beginComputePass({ label: "pressure" });
      pass.setPipeline(pressurePipeline);
      pass.setBindGroup(0, pressureBG);
      pass.dispatchWorkgroups(particleWG);
      pass.end();
    }

    // 5. XPBD constraint solve (N iterations)
    const iters = Math.max(1, Math.round(params.xpbdIters));
    for (let i = 0; i < iters; i++) {
      {
        const pass = encoder.beginComputePass({ label: `xpbd-accum-${i}` });
        pass.setPipeline(xpbdAccumPipeline);
        pass.setBindGroup(0, xpbdAccumBG);
        pass.dispatchWorkgroups(bondWG);
        pass.end();
      }
      {
        const pass = encoder.beginComputePass({ label: `xpbd-apply-${i}` });
        pass.setPipeline(xpbdApplyPipeline);
        pass.setBindGroup(0, xpbdApplyBG);
        pass.dispatchWorkgroups(particleWG);
        pass.end();
      }
    }

    // 6. Velocity + position integration (reads forceAccum, clears it)
    {
      const pass = encoder.beginComputePass({ label: "integrate" });
      pass.setPipeline(integratePipeline);
      pass.setBindGroup(0, integrateBG);
      pass.dispatchWorkgroups(particleWG);
      pass.end();
    }

    // 7. Spatial grid rebuild (reset heads first)
    encoder.copyBufferToBuffer(headsInitBuffer, 0, headsBuffer, 0, GRID_CELLS * 4);
    {
      const pass = encoder.beginComputePass({ label: "grid" });
      pass.setPipeline(gridPipeline);
      pass.setBindGroup(0, gridBG);
      pass.dispatchWorkgroups(particleWG);
      pass.end();
    }

    // 8. Nutrient absorption + proximity forces
    {
      const pass = encoder.beginComputePass({ label: "interact" });
      pass.setPipeline(interactPipeline);
      pass.setBindGroup(0, interactBG);
      pass.dispatchWorkgroups(particleWG);
      pass.end();
    }

    // 9. Particle type transitions
    {
      const pass = encoder.beginComputePass({ label: "transitions" });
      pass.setPipeline(transitionsPipeline);
      pass.setBindGroup(0, transitionsBG);
      pass.dispatchWorkgroups(particleWG);
      pass.end();
    }

    // 10. Render pass
    {
      const renderPass = encoder.beginRenderPass({
        colorAttachments: [{
          view:       context.getCurrentTexture().createView(),
          clearValue: { r: 0.0, g: 0.02, b: 0.05, a: 1.0 },
          loadOp:  "clear",
          storeOp: "store",
        }],
      });
      renderPass.setPipeline(renderPipeline);
      renderPass.setBindGroup(0, renderBG);
      renderPass.draw(6, MAX_PARTICLES, 0, 0);
      renderPass.end();
    }

    device.queue.submit([encoder.finish()]);
    rafId = requestAnimationFrame(frame);
  }

  rafId = requestAnimationFrame(frame);

  // -----------------------------------------------------------------------
  // Cleanup / stop function
  // -----------------------------------------------------------------------
  return () => {
    cancelAnimationFrame(rafId);
    window.removeEventListener("resize", onResize);
    pane.dispose();
    particleBuffer.destroy();
    bondBuffer.destroy();
    forceAccumBuffer.destroy();
    xpbdDeltaBuffer.destroy();
    headsBuffer.destroy();
    headsInitBuffer.destroy();
    linkedBuffer.destroy();
    freeListBuffer.destroy();
    freeCtrlBuffer.destroy();
    uniformBuffer.destroy();
    cameraBuffer.destroy();
  };
}
