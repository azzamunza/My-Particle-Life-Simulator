// Pass 4: XPBD constraint solve
// Two entry points sharing the same module-level bindings:
//   "accumulate"        — per-bond: compute corrections → xpbdDelta (atomic)
//   "apply_corrections" — per-particle: apply xpbdDelta and clear it

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

// All four bindings are declared at module scope.
// Each entry point only "statically uses" a subset; WebGPU auto-layout
// will only include the statically-used bindings in each pipeline's BGL.
@group(0) @binding(0) var<uniform>             uniforms  : Uniforms;
@group(0) @binding(1) var<storage, read_write>  particles : array<Particle>;
@group(0) @binding(2) var<storage, read>        bonds     : array<Bond>;
@group(0) @binding(3) var<storage, read_write>  xpbdDelta : array<atomic<i32>>;

const XPBD_FP_SCALE  : f32 = 65536.0;
const PTYPE_INACTIVE : u32 = 255u;

// ---- Entry point 1: accumulate corrections from bonds ----
@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bondIdx = gid.x;
    if bondIdx >= uniforms.numBonds { return; }

    let bond = bonds[bondIdx];
    let pa   = particles[bond.a];
    let pb   = particles[bond.b];

    if pa.ptype == PTYPE_INACTIVE || pb.ptype == PTYPE_INACTIVE { return; }
    if (pa.flags & 1u) == 0u || (pb.flags & 1u) == 0u { return; }

    let delta = pb.pos - pa.pos;
    let dist  = length(delta);
    if dist < 0.0001 { return; }
    let dir = delta / dist;

    let C          = dist - bond.restLen;
    let compliance = 1.0 / (bond.stiffness * uniforms.dt * uniforms.dt);
    let lambda     = -C / (2.0 + compliance);   // w_a = w_b = 1

    // dx_a = +lambda * dir,  dx_b = -lambda * dir
    atomicAdd(&xpbdDelta[bond.a * 2u],      i32(dir.x *  lambda * XPBD_FP_SCALE));
    atomicAdd(&xpbdDelta[bond.a * 2u + 1u], i32(dir.y *  lambda * XPBD_FP_SCALE));
    atomicAdd(&xpbdDelta[bond.b * 2u],      i32(dir.x * -lambda * XPBD_FP_SCALE));
    atomicAdd(&xpbdDelta[bond.b * 2u + 1u], i32(dir.y * -lambda * XPBD_FP_SCALE));
}

// ---- Entry point 2: apply corrections and clear xpbdDelta ----
@compute @workgroup_size(64)
fn apply_corrections(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.numParticles { return; }

    // Always clear to avoid stale values, even for inactive particles
    let raw_dx = atomicExchange(&xpbdDelta[idx * 2u],      0);
    let raw_dy = atomicExchange(&xpbdDelta[idx * 2u + 1u], 0);

    let p = particles[idx];
    if p.ptype == PTYPE_INACTIVE || (p.flags & 1u) == 0u { return; }

    let dx = f32(raw_dx) / XPBD_FP_SCALE;
    let dy = f32(raw_dy) / XPBD_FP_SCALE;
    particles[idx].pos = p.pos + vec2<f32>(dx, dy);
}
