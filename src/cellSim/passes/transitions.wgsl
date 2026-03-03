// Pass 8: Particle type transitions
// - Cytoplasm with age > cytoplLife → WASTE
// - WASTE that wanders outside cell → INACTIVE (pushed to free list)

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
@group(0) @binding(2) var<storage, read_write>  freeList  : array<u32>;
@group(0) @binding(3) var<storage, read_write>  freeCtrl  : array<atomic<u32>>;

const CELL_RADIUS     : f32  = 60.0;
const MAX_PARTICLES_U : u32  = 2048u;

const PTYPE_CYTOPLASM : u32 = 7u;
const PTYPE_WASTE     : u32 = 18u;
const PTYPE_INACTIVE  : u32 = 255u;

fn pushFreeSlot(slot: u32) {
    let tail = atomicAdd(&freeCtrl[1], 1u);
    freeList[tail % MAX_PARTICLES_U] = slot;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= uniforms.numParticles { return; }

    var p = particles[idx];
    if p.ptype == PTYPE_INACTIVE || (p.flags & 1u) == 0u { return; }

    // Cytoplasm age → waste
    if p.ptype == PTYPE_CYTOPLASM && u32(p.age) > uniforms.cytoplLife {
        particles[idx].ptype = PTYPE_WASTE;
        particles[idx].age   = 0.0;
        return;
    }

    // Waste that has drifted far outside the cell → despawn
    if p.ptype == PTYPE_WASTE {
        let dist = length(p.pos);
        if dist > CELL_RADIUS * 2.0 {
            particles[idx].ptype = PTYPE_INACTIVE;
            particles[idx].flags = 0u;
            pushFreeSlot(idx);
        }
    }
}
