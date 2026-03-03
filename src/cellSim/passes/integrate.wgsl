// Pass 5: Verlet velocity + position integration
// Reads accumulated forces from forceAccumBuffer (fixed-point i32),
// updates velocity and position, then clears the force accumulator.

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

@group(0) @binding(0) var<uniform>             uniforms  : Uniforms;
@group(0) @binding(1) var<storage, read_write>  particles : array<Particle>;
@group(0) @binding(2) var<storage, read_write>  forceAccum: array<atomic<i32>>;

const FORCE_FP_SCALE  : f32 = 1024.0;
const WORLD_SIZE      : f32 = 200.0;
const PTYPE_INACTIVE  : u32 = 255u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.numParticles { return; }

    let p = particles[idx];

    // Clear force accumulator regardless of active state
    let raw_fx = atomicExchange(&forceAccum[idx * 2u],     0);
    let raw_fy = atomicExchange(&forceAccum[idx * 2u + 1u], 0);

    if p.ptype == PTYPE_INACTIVE || (p.flags & 1u) == 0u { return; }

    let force = vec2<f32>(f32(raw_fx), f32(raw_fy)) / FORCE_FP_SCALE;
    let dt     = uniforms.dt;
    let damp   = uniforms.damping;

    var vel = (p.vel + force * dt) * damp;
    var pos = p.pos + vel * dt;

    // World boundary wrap
    let half = WORLD_SIZE;
    if pos.x >  half { pos.x -= half * 2.0; }
    if pos.x < -half { pos.x += half * 2.0; }
    if pos.y >  half { pos.y -= half * 2.0; }
    if pos.y < -half { pos.y += half * 2.0; }

    particles[idx].vel = vel;
    particles[idx].pos = pos;
}
