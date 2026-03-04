// Pass 1: Brownian motion + wave forces for cilia / flagellum / pseudopods

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

@group(0) @binding(0) var<uniform>            uniforms   : Uniforms;
@group(0) @binding(1) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(2) var<storage, read_write> forceAccum: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read>       heads     : array<i32>;
@group(0) @binding(4) var<storage, read>       linked    : array<i32>;

const FORCE_FP_SCALE  : f32 = 1024.0;
const WORLD_SIZE      : f32 = 200.0;
const GRID_CELL_SIZE  : f32 = 22.0;
const GRID_DIM        : u32 = 20u;

const PTYPE_MEMBRANE  : u32 = 0u;
const PTYPE_CILIA     : u32 = 8u;
const PTYPE_FLAGELLUM : u32 = 9u;
const PTYPE_PSEUDOPOD : u32 = 10u;
const PTYPE_INACTIVE  : u32 = 255u;
const PTYPE_NUTRIENT_1: u32 = 11u;
const PTYPE_NUTRIENT_7: u32 = 17u;

// PCG pseudo-random hash
fn pcg(v: u32) -> u32 {
    var s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn pcgFloat(v: u32) -> f32 {
    return f32(pcg(v)) / f32(0xffffffffu);
}

// Grid helpers
fn gridCell(pos: vec2<f32>) -> vec2<i32> {
    let shifted = pos + vec2<f32>(WORLD_SIZE);
    return vec2<i32>(
        clamp(i32(floor(shifted.x / GRID_CELL_SIZE)), 0, i32(GRID_DIM) - 1),
        clamp(i32(floor(shifted.y / GRID_CELL_SIZE)), 0, i32(GRID_DIM) - 1),
    );
}

fn cellIndex(cx: i32, cy: i32) -> i32 {
    return cy * i32(GRID_DIM) + cx;
}

fn addForce(idx: u32, fx: f32, fy: f32) {
    atomicAdd(&forceAccum[idx * 2u],     i32(fx * FORCE_FP_SCALE));
    atomicAdd(&forceAccum[idx * 2u + 1u], i32(fy * FORCE_FP_SCALE));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.numParticles { return; }

    let p = particles[idx];
    if p.ptype == PTYPE_INACTIVE || (p.flags & 1u) == 0u { return; }

    let tick = uniforms.tick;

    var fx: f32 = 0.0;
    var fy: f32 = 0.0;

    // ---- Cilia: lateral sine-wave perpendicular to outward direction ----
    if p.ptype == PTYPE_CILIA {
        // Outward direction: normalise position (membrane is at CELL_RADIUS)
        let r = length(p.pos);
        if r > 0.001 {
            let outward = p.pos / r;
            // Perpendicular (rotate 90°)
            let perp = vec2<f32>(-outward.y, outward.x);
            let wave = sin(p.phase + f32(p.chainIdx) * 0.8 + f32(tick) * 0.12);
            fx += perp.x * wave * uniforms.ciliaWave;
            fy += perp.y * wave * uniforms.ciliaWave;
        }
    }
    // ---- Flagellum: travelling sine wave along chain ----
    else if p.ptype == PTYPE_FLAGELLUM {
        let r = length(p.pos);
        if r > 0.001 {
            let outward = p.pos / r;
            let perp = vec2<f32>(-outward.y, outward.x);
            let wave = sin(f32(p.chainIdx) * 1.2 - f32(tick) * 0.18);
            fx += perp.x * wave * uniforms.flagWave;
            fy += perp.y * wave * uniforms.flagWave;
        }
    }
    // ---- Pseudopod tip: attract to nearest nutrient within 40 units ----
    else if p.ptype == PTYPE_PSEUDOPOD && p.chainIdx == 7u {
        let searchRadius = 40.0;
        let gc = gridCell(p.pos);
        let cells = min(i32(ceil(searchRadius / GRID_CELL_SIZE)) + 1, i32(GRID_DIM));
        var bestDist = searchRadius;
        var bestDir  = vec2<f32>(0.0);
        for (var dy2: i32 = -cells; dy2 <= cells; dy2++) {
            for (var dx2: i32 = -cells; dx2 <= cells; dx2++) {
                let nx = gc.x + dx2;
                let ny = gc.y + dy2;
                if nx < 0 || ny < 0 || nx >= i32(GRID_DIM) || ny >= i32(GRID_DIM) { continue; }
                var j = heads[cellIndex(nx, ny)];
                while j >= 0 {
                    let uj = u32(j);
                    if uj >= uniforms.numParticles { break; }
                    let q = particles[uj];
                    if q.ptype >= PTYPE_NUTRIENT_1 && q.ptype <= PTYPE_NUTRIENT_7 {
                        let diff = q.pos - p.pos;
                        let dist = length(diff);
                        if dist < bestDist && dist > 0.001 {
                            bestDist = dist;
                            bestDir  = diff / dist;
                        }
                    }
                    j = linked[j];
                }
            }
        }
        if bestDist < searchRadius {
            let strength = 0.5 * (1.0 - bestDist / searchRadius);
            fx += bestDir.x * strength;
            fy += bestDir.y * strength;
        }
    }
    // ---- Brownian motion for free (non-structural) particles ----
    else {
        let isStructural = (p.ptype <= 6u); // membrane...ribosome are structural
        if !isStructural {
            let seed1 = idx * 2u + tick * 7919u;
            let seed2 = idx * 2u + 1u + tick * 7919u;
            let rx = pcgFloat(seed1) * 2.0 - 1.0;
            let ry = pcgFloat(seed2) * 2.0 - 1.0;
            fx += rx * uniforms.brownianStr;
            fy += ry * uniforms.brownianStr;
        }
    }

    addForce(idx, fx, fy);
}
