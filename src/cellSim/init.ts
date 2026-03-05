import {
  MAX_PARTICLES, MAX_BONDS,
  CELL_RADIUS, NUCLEUS_RADIUS,
  MEMBRANE_NODES, NUCLEUS_NODES,
  DNA_RUNGS,
  RIBOSOME_COUNT,
  CILIA_COUNT, FLAGELLUM_LEN, PSEUDOPOD_COUNT,
  NUTRIENT_COUNT, SOLUTION_COUNT,
  MEMBRANE_K, NUCLEUS_K,
  DNA_BACKBONE_K, DNA_RUNG_K,
  RIBOSOME_K, CILIA_K,
  PARTICLE_FLOATS, BOND_UINTS,
  PTYPE_MEMBRANE, PTYPE_CHANNEL, PTYPE_PUMP,
  PTYPE_NUCLEUS, PTYPE_DNA_A, PTYPE_DNA_B,
  PTYPE_RIBOSOME, PTYPE_CYTOPLASM,
  PTYPE_CILIA, PTYPE_FLAGELLUM, PTYPE_PSEUDOPOD,
  PTYPE_NUTRIENT_1, PTYPE_SOLUTION, PTYPE_INACTIVE,
  WORLD_SIZE,
} from "./buffers";

// ============================================================
// Scene initialisation for the Cell Biology Simulator
// ============================================================

export interface SceneData {
  particleData: ArrayBuffer;
  bondData:     ArrayBuffer;
  bondCount:    number;
  freeList:     Uint32Array<ArrayBuffer>;
  freeCount:    number;
}

export function initScene(offsetX = 0, offsetY = 0): SceneData {
  const particleData = new ArrayBuffer(MAX_PARTICLES * 12 * 4); // 12 × f32/u32
  const pf32 = new Float32Array(particleData);
  const pu32 = new Uint32Array(particleData);

  const bondData = new ArrayBuffer(MAX_BONDS * 4 * 4); // 4 × f32/u32
  const bf32 = new Float32Array(bondData);
  const bu32 = new Uint32Array(bondData);

  // Mark all slots inactive initially
  for (let i = 0; i < MAX_PARTICLES; i++) {
    pu32[i * PARTICLE_FLOATS + 6] = PTYPE_INACTIVE;
  }

  let pCount = 0;
  let bCount = 0;

  // ------ Helper: add one particle (coordinates are offset by the cell origin) ------
  function addParticle(
    px: number, py: number,
    vx: number, vy: number,
    ptype: number,
    flags  = 1,
    age    = 0.0,
    phase  = 0.0,
    chainIdx = 0,
  ): number {
    const i = pCount++;
    const b = i * PARTICLE_FLOATS;
    pf32[b + 0] = px + offsetX;
    pf32[b + 1] = py + offsetY;
    pf32[b + 2] = vx;
    pf32[b + 3] = vy;
    pf32[b + 4] = 0; // force.x (cleared)
    pf32[b + 5] = 0; // force.y
    pu32[b + 6] = ptype;
    pu32[b + 7] = flags;
    pf32[b + 8] = age;
    pf32[b + 9] = phase;
    pu32[b + 10] = chainIdx;
    pu32[b + 11] = 0; // padding
    return i;
  }

  // ------ Helper: add one bond ------
  function addBond(a: number, b: number, restLen: number, stiffness: number): void {
    const base = bCount * BOND_UINTS;
    bu32[base + 0] = a;
    bu32[base + 1] = b;
    bf32[base + 2] = restLen;
    bf32[base + 3] = stiffness;
    bCount++;
  }

  // ============================================================
  // 1. Membrane ring (MEMBRANE_NODES = 60 particles)
  // ============================================================
  const membraneStart = pCount;
  // Use chord length (actual distance between adjacent particles) so bonds start
  // at their natural rest length and produce zero initial force.
  const memRestLen = 2 * Math.sin(Math.PI / MEMBRANE_NODES) * CELL_RADIUS;

  for (let i = 0; i < MEMBRANE_NODES; i++) {
    const angle = (i / MEMBRANE_NODES) * 2 * Math.PI;
    const px = Math.cos(angle) * CELL_RADIUS;
    const py = Math.sin(angle) * CELL_RADIUS;
    let ptype = PTYPE_MEMBRANE;
    if      (i % 8  === 0) ptype = PTYPE_CHANNEL;
    else if (i % 12 === 0) ptype = PTYPE_PUMP;
    addParticle(px, py, 0, 0, ptype);
  }
  // Bond neighbours + close the ring
  for (let i = 0; i < MEMBRANE_NODES; i++) {
    const a = membraneStart + i;
    const b = membraneStart + (i + 1) % MEMBRANE_NODES;
    addBond(a, b, memRestLen, MEMBRANE_K);
  }

  // ============================================================
  // 2. Nucleus ring (NUCLEUS_NODES = 24 particles)
  // ============================================================
  const nucleusStart = pCount;
  // Use chord length (actual distance between adjacent particles) so bonds start
  // at their natural rest length and produce zero initial force.
  const nucRestLen = 2 * Math.sin(Math.PI / NUCLEUS_NODES) * NUCLEUS_RADIUS;

  for (let i = 0; i < NUCLEUS_NODES; i++) {
    const angle = (i / NUCLEUS_NODES) * 2 * Math.PI;
    addParticle(
      Math.cos(angle) * NUCLEUS_RADIUS,
      Math.sin(angle) * NUCLEUS_RADIUS,
      0, 0, PTYPE_NUCLEUS,
    );
  }
  for (let i = 0; i < NUCLEUS_NODES; i++) {
    const a = nucleusStart + i;
    const b = nucleusStart + (i + 1) % NUCLEUS_NODES;
    addBond(a, b, nucRestLen, NUCLEUS_K);
  }

  // ============================================================
  // 3. DNA (2 strands × 20 rungs = 80 particles)
  // ============================================================
  // Scale DNA to fit proportionally inside the nucleus
  const dnaHalfSpan = NUCLEUS_RADIUS * 0.75; // X half-span
  const dnaYOffset  = NUCLEUS_RADIUS * 0.36; // Y offset between strands
  const dnaSpacing  = (dnaHalfSpan * 2) / (DNA_RUNGS - 1);
  const dnaRungLen  = dnaYOffset * 2;        // rest length of rung bonds

  const strandAIndices: number[] = [];
  const strandBIndices: number[] = [];

  for (let i = 0; i < DNA_RUNGS; i++) {
    const x = -dnaHalfSpan + i * dnaSpacing;
    strandAIndices.push(addParticle(x, -dnaYOffset, 0, 0, PTYPE_DNA_A));
    strandBIndices.push(addParticle(x,  dnaYOffset, 0, 0, PTYPE_DNA_B));
  }
  // Backbone bonds (strand A and strand B)
  for (let i = 0; i < DNA_RUNGS - 1; i++) {
    addBond(strandAIndices[i], strandAIndices[i + 1], dnaSpacing, DNA_BACKBONE_K);
    addBond(strandBIndices[i], strandBIndices[i + 1], dnaSpacing, DNA_BACKBONE_K);
  }
  // Rung bonds
  for (let i = 0; i < DNA_RUNGS; i++) {
    addBond(strandAIndices[i], strandBIndices[i], dnaRungLen, DNA_RUNG_K);
  }

  // ============================================================
  // 4. Ribosomes (5 clusters of 4 particles = 20 particles)
  // ============================================================
  const ribosomeRadius = (NUCLEUS_RADIUS - 5) * 0.7; // inside nucleus
  for (let r = 0; r < RIBOSOME_COUNT; r++) {
    const angle = (r / RIBOSOME_COUNT) * 2 * Math.PI;
    const cx = Math.cos(angle) * ribosomeRadius;
    const cy = Math.sin(angle) * ribosomeRadius;
    const half = 2.0; // half side of the square
    const corners = [
      addParticle(cx - half, cy - half, 0, 0, PTYPE_RIBOSOME),
      addParticle(cx + half, cy - half, 0, 0, PTYPE_RIBOSOME),
      addParticle(cx + half, cy + half, 0, 0, PTYPE_RIBOSOME),
      addParticle(cx - half, cy + half, 0, 0, PTYPE_RIBOSOME),
    ];
    const side = half * 2;       // 4
    const diag = side * Math.SQRT2; // ~5.66
    // 4 edges + 2 diagonals = 6 pairs
    addBond(corners[0], corners[1], side, RIBOSOME_K);
    addBond(corners[1], corners[2], side, RIBOSOME_K);
    addBond(corners[2], corners[3], side, RIBOSOME_K);
    addBond(corners[3], corners[0], side, RIBOSOME_K);
    addBond(corners[0], corners[2], diag, RIBOSOME_K);
    addBond(corners[1], corners[3], diag, RIBOSOME_K);
  }

  // ============================================================
  // 5. Cilia (8 chains × 6 nodes = 48 particles)
  // ============================================================
  // Start from every 8th membrane particle (not channel/pump)
  const CILIA_NODES = 6;
  let ciliaChain = 0;
  for (let i = 0; i < MEMBRANE_NODES && ciliaChain < CILIA_COUNT; i += 8) {
    const memIdx = membraneStart + i;
    const memPtype = pu32[memIdx * PARTICLE_FLOATS + 6];
    if (memPtype === PTYPE_CHANNEL || memPtype === PTYPE_PUMP) continue;

    const angle = (i / MEMBRANE_NODES) * 2 * Math.PI;
    const outX = Math.cos(angle);
    const outY = Math.sin(angle);
    const rootX = Math.cos(angle) * CELL_RADIUS;
    const rootY = Math.sin(angle) * CELL_RADIUS;

    let prevIdx = memIdx;
    for (let n = 0; n < CILIA_NODES; n++) {
      const px = rootX + outX * (n + 1) * 5;
      const py = rootY + outY * (n + 1) * 5;
      const phase = ciliaChain * 0.5;
      const idx = addParticle(px, py, 0, 0, PTYPE_CILIA, 1, 0, phase, n);
      addBond(prevIdx, idx, 5.0, CILIA_K);
      prevIdx = idx;
    }
    ciliaChain++;
  }

  // ============================================================
  // 6. Flagellum (1 chain, 12 nodes)
  // ============================================================
  {
    // Extends from membrane particle at angle 180° (index = MEMBRANE_NODES/2)
    const flagAngleIdx = Math.round(MEMBRANE_NODES / 2) % MEMBRANE_NODES;
    const flagAngle = (flagAngleIdx / MEMBRANE_NODES) * 2 * Math.PI;
    const outX = Math.cos(flagAngle);
    const outY = Math.sin(flagAngle);
    const rootX = outX * CELL_RADIUS;
    const rootY = outY * CELL_RADIUS;

    let prevIdx = membraneStart + flagAngleIdx;
    for (let n = 0; n < FLAGELLUM_LEN; n++) {
      const px = rootX + outX * (n + 1) * 6;
      const py = rootY + outY * (n + 1) * 6;
      const idx = addParticle(px, py, 0, 0, PTYPE_FLAGELLUM, 1, 0, 0, n);
      addBond(prevIdx, idx, 6.0, CILIA_K * 0.7);
      prevIdx = idx;
    }
  }

  // ============================================================
  // 7. Pseudopods (3 chains × 8 nodes = 24 particles)
  // ============================================================
  const PSEUDOPOD_NODES = 8;
  for (let p = 0; p < PSEUDOPOD_COUNT; p++) {
    const podAngle = (p / PSEUDOPOD_COUNT) * 2 * Math.PI; // 0°, 120°, 240°
    const podAngleIdx = Math.round((podAngle / (2 * Math.PI)) * MEMBRANE_NODES) % MEMBRANE_NODES;
    const outX = Math.cos(podAngle);
    const outY = Math.sin(podAngle);
    const rootX = outX * CELL_RADIUS;
    const rootY = outY * CELL_RADIUS;

    let prevIdx = membraneStart + podAngleIdx;
    for (let n = 0; n < PSEUDOPOD_NODES; n++) {
      const px = rootX + outX * (n + 1) * 7;
      const py = rootY + outY * (n + 1) * 7;
      const idx = addParticle(px, py, 0, 0, PTYPE_PSEUDOPOD, 1, 0, 0, n);
      addBond(prevIdx, idx, 7.0, CILIA_K * 0.5);
      prevIdx = idx;
    }
  }

  // ============================================================
  // 8. Cytoplasm (100 initial particles)
  // ============================================================
  const CYTOPLASM_INITIAL = 100;
  let cytoCount = 0;
  while (cytoCount < CYTOPLASM_INITIAL && pCount < MAX_PARTICLES) {
    const r     = Math.random() * CELL_RADIUS * 0.85;
    const angle = Math.random() * 2 * Math.PI;
    const px = Math.cos(angle) * r;
    const py = Math.sin(angle) * r;
    const vx = (Math.random() - 0.5);
    const vy = (Math.random() - 0.5);
    addParticle(px, py, vx, vy, PTYPE_CYTOPLASM);
    cytoCount++;
  }

  // ============================================================
  // 9. Nutrients (500 particles, PTYPE_NUTRIENT_1 … +6)
  // ============================================================
  const NUTRIENT_TYPES = 7;
  for (let n = 0; n < NUTRIENT_COUNT && pCount < MAX_PARTICLES; n++) {
    let px = 0, py = 0;
    // Place outside membrane (r > CELL_RADIUS + 10), within world bounds
    do {
      px = (Math.random() * 2 - 1) * WORLD_SIZE;
      py = (Math.random() * 2 - 1) * WORLD_SIZE;
    } while (Math.sqrt(px * px + py * py) < CELL_RADIUS + 10);

    const ptype = PTYPE_NUTRIENT_1 + (n % NUTRIENT_TYPES);
    const vx = (Math.random() - 0.5) * 0.6;
    const vy = (Math.random() - 0.5) * 0.6;
    addParticle(px, py, vx, vy, ptype);
  }

  // ============================================================
  // 10. Solution / extracellular fluid particles
  //     Fill the world outside (and loosely around) the cell.
  //     These act as a liquid medium that other particles push through.
  // ============================================================
  let solCount = 0;
  while (solCount < SOLUTION_COUNT && pCount < MAX_PARTICLES) {
    const px = (Math.random() * 2 - 1) * WORLD_SIZE;
    const py = (Math.random() * 2 - 1) * WORLD_SIZE;
    // Keep solution particles outside the cell membrane (squared-distance check)
    if (px * px + py * py < (CELL_RADIUS + 5) ** 2) continue;
    const vx = (Math.random() - 0.5) * 0.3;
    const vy = (Math.random() - 0.5) * 0.3;
    addParticle(px, py, vx, vy, PTYPE_SOLUTION);
    solCount++;
  }

  // ============================================================
  // 11. Free list: remaining inactive slots
  // ============================================================
  const freeList  = new Uint32Array(MAX_PARTICLES) as Uint32Array<ArrayBuffer>;
  let   freeCount = 0;
  for (let i = pCount; i < MAX_PARTICLES; i++) {
    pu32[i * PARTICLE_FLOATS + 6] = PTYPE_INACTIVE;
    pu32[i * PARTICLE_FLOATS + 7] = 0; // flags = 0 (inactive)
    freeList[freeCount++] = i;
  }

  return { particleData, bondData, bondCount: bCount, freeList, freeCount };
}
