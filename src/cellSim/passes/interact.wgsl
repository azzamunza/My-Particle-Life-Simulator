// Pass 7: Nutrient absorption via channel proteins + cytoplasm aging + pump ejection

struct Uniforms {
    tick         : u32,
    numParticles : u32,
    numBonds     : u32,
    pad0         : u32,
    dt           : f32,
    brownianStr  : f32,
    damping      : f32,
    pressureK    : f32,
    membraneK    : f32,
    nucleusK     : f32,
    ciliaWave    : f32,
    flagWave     : f32,
    channelR     : f32,
    cytoplLife   : u32,
    pad1         : u32,
    pad2         : u32,
};

struct Particle {
    pos      : vec2<f32>,
    vel      : vec2<f32>,
    force    : vec2<f32>,
    ptype    : u32,
    flags    : u32,
    age      : f32,
    phase    : f32,
    chainIdx : u32,
    pad      : u32,
};

@group(0) @binding(0) var<uniform>             uniforms   : Uniforms;
@group(0) @binding(1) var<storage, read_write>  particles  : array<Particle>;
@group(0) @binding(2) var<storage, read>        heads      : array<i32>;
@group(0) @binding(3) var<storage, read>        linked     : array<i32>;
@group(0) @binding(4) var<storage, read_write>  forceAccum : array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write>  freeList   : array<u32>;
@group(0) @binding(6) var<storage, read_write>  freeCtrl   : array<atomic<u32>>;

const WORLD_SIZE     : f32 = 200.0;
const GRID_CELL_SIZE : f32 = 22.0;
const GRID_DIM       : u32 = 20u;
const CELL_RADIUS    : f32 = 60.0;
const FORCE_FP_SCALE : f32 = 1024.0;
const MIN_LENGTH     : f32 = 0.001;

const PTYPE_CHANNEL   : u32 = 1u;
const PTYPE_PUMP      : u32 = 2u;
const PTYPE_CYTOPLASM : u32 = 7u;
const PTYPE_WASTE     : u32 = 18u;
const PTYPE_INACTIVE  : u32 = 255u;
const PTYPE_NUTRIENT_1: u32 = 11u;
const PTYPE_NUTRIENT_7: u32 = 17u;

const MAX_PARTICLES_U : u32 = 2048u;

fn cellCoord(v: f32) -> i32 {
    let shifted = v + WORLD_SIZE;
    return clamp(i32(floor(shifted / GRID_CELL_SIZE)), 0, i32(GRID_DIM) - 1);
}

fn gridCellIdx(cx: i32, cy: i32) -> i32 {
    return cy * i32(GRID_DIM) + cx;
}

// Pop a free slot from the ring buffer; returns MAX_PARTICLES_U if empty
fn popFreeSlot() -> u32 {
    let head = atomicAdd(&freeCtrl[0], 1u);
    let tail = atomicLoad(&freeCtrl[1]);
    if head >= tail {
        atomicSub(&freeCtrl[0], 1u); // undo
        return MAX_PARTICLES_U;
    }
    return freeList[head % MAX_PARTICLES_U];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.numParticles { return; }

    var p = particles[idx];
    if (p.flags & 1u) == 0u || p.ptype == PTYPE_INACTIVE { return; }

    // ---- Increment age for cytoplasm particles ----
    if p.ptype == PTYPE_CYTOPLASM {
        particles[idx].age = p.age + 1.0;
        return; // handled by transitions pass
    }

    // ---- Channel protein: scan for nearby nutrients ----
    if p.ptype == PTYPE_CHANNEL {
        let cx = cellCoord(p.pos.x);
        let cy = cellCoord(p.pos.y);
        let scanCells = min(i32(ceil(uniforms.channelR / GRID_CELL_SIZE)) + 1, i32(GRID_DIM));

        for (var dy2: i32 = -scanCells; dy2 <= scanCells; dy2++) {
            for (var dx2: i32 = -scanCells; dx2 <= scanCells; dx2++) {
                let nx = cx + dx2;
                let ny = cy + dy2;
                if nx < 0 || ny < 0 || nx >= i32(GRID_DIM) || ny >= i32(GRID_DIM) { continue; }
                var j = heads[gridCellIdx(nx, ny)];
                while j >= 0 {
                    let uj = u32(j);
                    if uj >= uniforms.numParticles { break; }
                    let q  = particles[uj];
                    if q.ptype >= PTYPE_NUTRIENT_1 && q.ptype <= PTYPE_NUTRIENT_7
                       && (q.flags & 1u) != 0u {
                        let diff = q.pos - p.pos;
                        let dist = length(diff);
                        if dist < 3.0 {
                            // Absorb: convert nutrient to cytoplasm inside the cell
                            let innerR = CELL_RADIUS * 0.5;
                            let qLen   = length(q.pos);
                            let inDir  = select(vec2<f32>(1.0, 0.0), q.pos / qLen, qLen > MIN_LENGTH); // fallback: place on +x axis when pos is at origin
                            particles[uj].pos      = inDir * innerR;
                            particles[uj].vel      = vec2<f32>(0.0);
                            particles[uj].force    = vec2<f32>(0.0);
                            particles[uj].ptype    = PTYPE_CYTOPLASM;
                            particles[uj].age      = 0.0;
                        } else if dist < uniforms.channelR {
                            // Attract toward channel
                            let attraction = (1.0 - dist / uniforms.channelR) * 0.3;
                            let dir = -diff / max(dist, MIN_LENGTH);
                            atomicAdd(&forceAccum[uj * 2u],      i32(dir.x * attraction * FORCE_FP_SCALE));
                            atomicAdd(&forceAccum[uj * 2u + 1u], i32(dir.y * attraction * FORCE_FP_SCALE));
                        }
                    }
                    j = linked[j];
                }
            }
        }
    }

    // ---- Respawn nutrients that leave world bounds ----
    let isNutrient = (p.ptype >= PTYPE_NUTRIENT_1 && p.ptype <= PTYPE_NUTRIENT_7);
    if isNutrient {
        let halfW = WORLD_SIZE;
        if abs(p.pos.x) > halfW || abs(p.pos.y) > halfW {
            // Respawn on a random world-boundary edge
            let tick = uniforms.tick;
            // Simple deterministic respawn using tick+idx as seed
            let seed = idx * 1664525u + tick * 1013904223u;
            let edge = seed % 4u;
            var nx2: f32; var ny2: f32;
            let t = f32((seed >> 8u) % 10000u) / 10000.0 * 2.0 - 1.0;
            if edge == 0u       { nx2 = -halfW; ny2 = t * halfW; }
            else if edge == 1u  { nx2 =  halfW; ny2 = t * halfW; }
            else if edge == 2u  { nx2 = t * halfW; ny2 = -halfW; }
            else                { nx2 = t * halfW; ny2 =  halfW; }
            particles[idx].pos = vec2<f32>(nx2, ny2);
            particles[idx].vel = vec2<f32>(0.0);
        }
    }
}
