// Pass 3: Radial pressure for membrane and nucleus rings

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

@group(0) @binding(0) var<uniform>              uniforms  : Uniforms;
@group(0) @binding(1) var<storage, read>         particles : array<Particle>;
@group(0) @binding(2) var<storage, read_write>   forceAccum: array<atomic<i32>>;

const FORCE_FP_SCALE  : f32 = 1024.0;
const CELL_RADIUS     : f32 = 80.0;
const NUCLEUS_RADIUS  : f32 = 28.0;
const PTYPE_MEMBRANE  : u32 = 0u;
const PTYPE_CHANNEL   : u32 = 1u;
const PTYPE_PUMP      : u32 = 2u;
const PTYPE_NUCLEUS   : u32 = 3u;
const PTYPE_INACTIVE  : u32 = 255u;

fn addForceA(idx: u32, fx: f32, fy: f32) {
    atomicAdd(&forceAccum[idx * 2u],     i32(fx * FORCE_FP_SCALE));
    atomicAdd(&forceAccum[idx * 2u + 1u], i32(fy * FORCE_FP_SCALE));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.numParticles { return; }

    let p = particles[idx];
    if p.ptype == PTYPE_INACTIVE || (p.flags & 1u) == 0u { return; }

    let isMembrane = (p.ptype == PTYPE_MEMBRANE || p.ptype == PTYPE_CHANNEL || p.ptype == PTYPE_PUMP);
    let isNucleus  = (p.ptype == PTYPE_NUCLEUS);

    if !isMembrane && !isNucleus { return; }

    let dist   = length(p.pos);
    if dist < 0.0001 { return; }
    let dir    = p.pos / dist;

    var targetR: f32;
    var k:       f32;
    if isMembrane {
        targetR = CELL_RADIUS;
        k = uniforms.pressureK;
    } else {
        targetR = NUCLEUS_RADIUS;
        k = uniforms.pressureK;
    }

    // Push inward if outside target radius, outward if inside 90% of target radius
    let error = targetR - dist;
    var fmag: f32 = 0.0;
    if dist > targetR {
        fmag = error * k; // negative error → force inward (toward origin)
    } else if dist < targetR * 0.9 {
        fmag = error * k; // positive error → force outward (away from origin)
    }

    // dir points outward; force = dir * (-fmag) if dist > targetR (push inward)
    // Using error = targetR - dist: positive means particle is too close
    addForceA(idx, dir.x * fmag, dir.y * fmag);
}
