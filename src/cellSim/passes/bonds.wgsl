// Pass 2: Bond / spring force accumulation

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

struct Bond {
    a         : u32,
    b         : u32,
    restLen   : f32,
    stiffness : f32,
};

@group(0) @binding(0) var<uniform>           uniforms  : Uniforms;
@group(0) @binding(1) var<storage, read>     particles : array<Particle>;
@group(0) @binding(2) var<storage, read>     bonds     : array<Bond>;
@group(0) @binding(3) var<storage, read_write> forceAccum: array<atomic<i32>>;

const FORCE_FP_SCALE : f32 = 1024.0;
const PTYPE_INACTIVE : u32 = 255u;

fn addForceA(idx: u32, fx: f32, fy: f32) {
    atomicAdd(&forceAccum[idx * 2u],     i32(fx * FORCE_FP_SCALE));
    atomicAdd(&forceAccum[idx * 2u + 1u], i32(fy * FORCE_FP_SCALE));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bondIdx = gid.x;
    if bondIdx >= uniforms.numBonds { return; }

    let bond = bonds[bondIdx];
    let pa   = particles[bond.a];
    let pb   = particles[bond.b];

    // Skip bonds involving inactive particles
    if pa.ptype == PTYPE_INACTIVE || pb.ptype == PTYPE_INACTIVE { return; }
    if (pa.flags & 1u) == 0u || (pb.flags & 1u) == 0u { return; }

    let delta   = pb.pos - pa.pos;
    let dist    = length(delta);
    if dist < 0.0001 { return; }

    let dir     = delta / dist;
    let stretch = dist - bond.restLen;
    let fmag    = stretch * bond.stiffness * 0.5;

    addForceA(bond.a, dir.x * fmag,  dir.y * fmag);
    addForceA(bond.b, -dir.x * fmag, -dir.y * fmag);
}
