import renderSrc from "./render.wgsl?raw";
import {
  COLOUR_COUNT,
  MATRIX_SIZE,
  PARTICLE_COUNT,
  defaultOptions,
  setSim,
  type SimOptions,
} from "./options";
import { LinkedListEngine } from "./linkedList/main";
import { Pane } from "tweakpane";

// ---------------------------------------------------------------------------
// WebGPU initialisation
// ---------------------------------------------------------------------------
async function initWebGPU(canvas: HTMLCanvasElement): Promise<{
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
}> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not supported in this browser.");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter found.");
  }
  const device = await adapter.requestDevice();

  const context = canvas.getContext("webgpu") as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "premultiplied" });

  return { device, context, format };
}

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------
/** Convert HSL (all in [0,1]) to linear RGB */
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h * 6) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if      (h < 1/6) { r = c; g = x; b = 0; }
  else if (h < 2/6) { r = x; g = c; b = 0; }
  else if (h < 3/6) { r = 0; g = c; b = x; }
  else if (h < 4/6) { r = 0; g = x; b = c; }
  else if (h < 5/6) { r = x; g = 0; b = c; }
  else              { r = c; g = 0; b = x; }
  return [r + m, g + m, b + m];
}

/** Build a flat Float32Array palette of size COLOUR_COUNT × 4 (rgb + padding) */
function buildPalette(): Float32Array<ArrayBuffer> {
  const data = new Float32Array(COLOUR_COUNT * 4) as Float32Array<ArrayBuffer>;
  for (let i = 0; i < COLOUR_COUNT; i++) {
    const h = i / COLOUR_COUNT;
    const [r, g, b] = hslToRgb(h, 0.9, 0.6);
    data[i * 4 + 0] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 1.0;
  }
  return data;
}

// ---------------------------------------------------------------------------
// Particle initialisation
// ---------------------------------------------------------------------------
/** Assign random colour indices in [0, COLOUR_COUNT) */
function buildColourIndices(): Uint32Array<ArrayBuffer> {
  const data = new Uint32Array(PARTICLE_COUNT) as Uint32Array<ArrayBuffer>;
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    data[i] = Math.floor(Math.random() * COLOUR_COUNT);
  }
  return data;
}

/** Initialise particles with random positions in [-worldSize, worldSize] */
function buildParticles(worldSize: number): Float32Array<ArrayBuffer> {
  // Each particle: vec4<f32> = (x, y, vx, vy)
  const data = new Float32Array(PARTICLE_COUNT * 4) as Float32Array<ArrayBuffer>;
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    data[i * 4 + 0] = (Math.random() * 2 - 1) * worldSize;
    data[i * 4 + 1] = (Math.random() * 2 - 1) * worldSize;
    data[i * 4 + 2] = 0;
    data[i * 4 + 3] = 0;
  }
  return data;
}

/** Build a random interaction matrix of size COLOUR_COUNT × COLOUR_COUNT */
function buildMatrix(): Float32Array<ArrayBuffer> {
  const data = new Float32Array(MATRIX_SIZE) as Float32Array<ArrayBuffer>;
  for (let i = 0; i < MATRIX_SIZE; i++) {
    data[i] = Math.random() * 2 - 1; // range [-1, 1]
  }
  return data;
}

// ---------------------------------------------------------------------------
// Camera state (pan + zoom)
// ---------------------------------------------------------------------------
interface Camera {
  x: number;
  y: number;
  zoom: number;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
async function main(): Promise<void> {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
  window.addEventListener("resize", () => {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  });

  const { device, context, format } = await initWebGPU(canvas);

  // -----------------------------------------------------------------------
  // Simulation options & buffers
  // -----------------------------------------------------------------------
  const opts: SimOptions = { ...defaultOptions };

  const simBuffer = device.createBuffer({
    label: "sim uniforms",
    size: 40,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  setSim(device, simBuffer, opts);

  const matrixData = buildMatrix();
  const matrixBuffer = device.createBuffer({
    label: "interaction matrix",
    size: matrixData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(matrixBuffer, 0, matrixData);

  const colourData = buildColourIndices();
  const colourBuffer = device.createBuffer({
    label: "colour indices",
    size: colourData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(colourBuffer, 0, colourData);

  // -----------------------------------------------------------------------
  // Palette buffer
  // -----------------------------------------------------------------------
  const paletteData = buildPalette();
  const paletteBuffer = device.createBuffer({
    label: "palette",
    size: paletteData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paletteBuffer, 0, paletteData);

  // -----------------------------------------------------------------------
  // Particle buffers (ping-pong)
  // -----------------------------------------------------------------------
  const particleBytes = PARTICLE_COUNT * 16; // 4 floats × 4 bytes
  const particleBufferA = device.createBuffer({
    label: "particles A",
    size: particleBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  const particleBufferB = device.createBuffer({
    label: "particles B",
    size: particleBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  const particleData = buildParticles(opts.worldSize);
  device.queue.writeBuffer(particleBufferA, 0, particleData);
  device.queue.writeBuffer(particleBufferB, 0, particleData);

  // -----------------------------------------------------------------------
  // Camera uniform buffer
  // -----------------------------------------------------------------------
  const cameraBuffer = device.createBuffer({
    label: "camera",
    size: 16, // vec2 pos + f32 zoom + f32 pad
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const camera: Camera = { x: 0, y: 0, zoom: Math.min(canvas.width, canvas.height) / (opts.worldSize * 2) };

  function updateCamera(): void {
    const aspect = canvas.width / canvas.height;
    const data = new Float32Array([camera.x, camera.y, camera.zoom / aspect, 0]);
    device.queue.writeBuffer(cameraBuffer, 0, data);
  }
  updateCamera();

  // -----------------------------------------------------------------------
  // Render pipeline
  // -----------------------------------------------------------------------
  const renderModule = device.createShaderModule({ code: renderSrc });

  const renderBindGroupLayout = device.createBindGroupLayout({
    label: "render BGL",
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
    ],
  });

  const renderPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [renderBindGroupLayout],
  });

  const renderPipeline = device.createRenderPipeline({
    label: "render pipeline",
    layout: renderPipelineLayout,
    vertex: {
      module: renderModule,
      entryPoint: "vs_main",
    },
    fragment: {
      module: renderModule,
      entryPoint: "fs_main",
      targets: [
        {
          format,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
            alpha: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
          },
        },
      ],
    },
    primitive: { topology: "triangle-list" },
  });

  // One render bind group per particle buffer
  function makeRenderBG(particleBuffer: GPUBuffer): GPUBindGroup {
    return device.createBindGroup({
      layout: renderBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: particleBuffer } },
        { binding: 1, resource: { buffer: colourBuffer } },
        { binding: 2, resource: { buffer: paletteBuffer } },
        { binding: 3, resource: { buffer: cameraBuffer } },
      ],
    });
  }

  const renderBindGroups: [GPUBindGroup, GPUBindGroup] = [
    makeRenderBG(particleBufferA),
    makeRenderBG(particleBufferB),
  ];

  // -----------------------------------------------------------------------
  // Linked list engine
  // -----------------------------------------------------------------------
  const engine = new LinkedListEngine(
    device,
    simBuffer,
    matrixBuffer,
    colourBuffer,
    particleBufferA,
    particleBufferB
  );

  // -----------------------------------------------------------------------
  // Tweakpane UI
  // -----------------------------------------------------------------------
  const pane = new Pane({ container: document.getElementById("ui") ?? undefined });
  const simFolder = pane.addFolder({ title: "Simulation" });

  simFolder.addBinding(opts, "r",         { min: 1,    max: 50,  label: "Radius" });
  simFolder.addBinding(opts, "force",     { min: 0,    max: 5,   label: "Force" });
  simFolder.addBinding(opts, "friction",  { min: 0,    max: 0.5, label: "Friction" });
  simFolder.addBinding(opts, "beta",      { min: 0.01, max: 0.9, label: "Beta" });
  simFolder.addBinding(opts, "delta",     { min: 0.001,max: 0.1, label: "Delta" });
  simFolder.addBinding(opts, "avoidance", { min: 0,    max: 20,  label: "Avoidance" });
  simFolder.addBinding(opts, "border",    { label: "Wrap border" });
  simFolder.addBinding(opts, "vortex",    { label: "Vortex" });

  simFolder.on("change", () => setSim(device, simBuffer, opts));

  const btnRandomise = simFolder.addButton({ title: "Randomise matrix" });
  btnRandomise.on("click", () => {
    const m = buildMatrix();
    device.queue.writeBuffer(matrixBuffer, 0, m);
  });

  const camFolder = pane.addFolder({ title: "Camera" });
  camFolder.addBinding(camera, "zoom", { min: 1, max: 500, label: "Zoom" }).on("change", updateCamera);

  // -----------------------------------------------------------------------
  // Mouse interaction (pan)
  // -----------------------------------------------------------------------
  let dragging = false;
  let lastX = 0, lastY = 0;
  canvas.addEventListener("mousedown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
  canvas.addEventListener("mouseup",   () => { dragging = false; });
  canvas.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = (e.clientX - lastX) / camera.zoom;
    const dy = (e.clientY - lastY) / camera.zoom;
    camera.x -= dx;
    camera.y -= dy;
    lastX = e.clientX;
    lastY = e.clientY;
    updateCamera();
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    camera.zoom *= e.deltaY < 0 ? 1.1 : 0.9;
    updateCamera();
  }, { passive: false });

  // -----------------------------------------------------------------------
  // Frame loop
  // -----------------------------------------------------------------------
  function frame(): void {
    const encoder = device.createCommandEncoder();

    // Compute passes (tick physics)
    engine.tick(encoder);

    // Render pass
    const currentBuf = engine.currentBufferIndex;
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroups[currentBuf]);
    // 6 vertices per particle quad
    renderPass.draw(6, PARTICLE_COUNT, 0, 0);
    renderPass.end();

    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
  document.body.innerHTML = `<pre style="color:red;padding:20px">${err}</pre>`;
});
