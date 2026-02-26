// Simulation parameters

/** Total number of particles */
export const PARTICLE_COUNT = 50000;

/** Number of distinct particle colour types */
export const COLOUR_COUNT = 200;

/** Squared side length of interaction matrix (COLOUR_COUNT * COLOUR_COUNT) */
export const MATRIX_SIZE = COLOUR_COUNT * COLOUR_COUNT;

// ---------------------------------------------------------------------------
// GPU buffer layout (must match the Sim struct in WGSL shaders)
// Fields (all f32 or u32, 4 bytes each):
//   0  colours   u32
//   1  r         f32  maximum interaction radius
//   2  force     f32  force magnitude multiplier
//   3  friction  f32  velocity damping per tick
//   4  beta      f32  attraction / repulsion threshold
//   5  delta     f32  time step
//   6  avoidance f32  close-range repulsion radius
//   7  worldSize f32  half-size of world boundary
//   8  border    u32  1 = wrap borders, 0 = reflect
//   9  vortex    u32  1 = add rotational force
// ---------------------------------------------------------------------------
export interface SimOptions {
  colours: number;
  r: number;
  force: number;
  friction: number;
  beta: number;
  delta: number;
  avoidance: number;
  worldSize: number;
  border: boolean;
  vortex: boolean;
}

/** Default simulation parameters */
export const defaultOptions: SimOptions = {
  colours: COLOUR_COUNT,
  r: 15,
  force: 1,
  friction: 0.04,
  beta: 0.3,
  delta: 0.02,
  avoidance: 4,
  worldSize: 6,
  border: true,
  vortex: false,
};

/**
 * Write the current SimOptions into a GPU-visible uniform buffer.
 * The buffer must be at least 40 bytes (10 × 4-byte fields).
 */
export function setSim(device: GPUDevice, buffer: GPUBuffer, opts: SimOptions): void {
  const data = new ArrayBuffer(40);
  const u32 = new Uint32Array(data);
  const f32 = new Float32Array(data);
  u32[0] = opts.colours;
  f32[1] = opts.r;
  f32[2] = opts.force;
  f32[3] = opts.friction;
  f32[4] = opts.beta;
  f32[5] = opts.delta;
  f32[6] = opts.avoidance;
  f32[7] = opts.worldSize;
  u32[8] = opts.border ? 1 : 0;
  u32[9] = opts.vortex ? 1 : 0;
  device.queue.writeBuffer(buffer, 0, data);
}
