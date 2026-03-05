// Pass 6: Spatial grid rebuild (linked-list)
// Inserts each active particle into the appropriate grid cell.

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

@group(0) @binding(0) var<uniform>            uniforms  : Uniforms;
@group(0) @binding(1) var<storage, read>      particles : array<Particle>;
@group(0) @binding(2) var<storage, read_write> heads    : array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> linked   : array<i32>;

const WORLD_SIZE     : f32 = 800.0;
const GRID_CELL_SIZE : f32 = 22.0;
const GRID_DIM       : u32 = 74u;
const PTYPE_INACTIVE : u32 = 255u;

fn cellCoord(v: f32) -> u32 {
    let shifted = v + WORLD_SIZE;
    return clamp(u32(floor(shifted / GRID_CELL_SIZE)), 0u, GRID_DIM - 1u);
}

fn cellIndex(cx: u32, cy: u32) -> u32 {
    return cy * GRID_DIM + cx;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.numParticles { return; }

    let p = particles[idx];
    if p.ptype == PTYPE_INACTIVE || (p.flags & 1u) == 0u { return; }

    let cx   = cellCoord(p.pos.x);
    let cy   = cellCoord(p.pos.y);
    let cell = cellIndex(cx, cy);

    let prev = atomicExchange(&heads[cell], i32(idx));
    linked[idx] = prev;
}
