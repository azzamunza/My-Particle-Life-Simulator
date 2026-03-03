// Linked-list spatial grid construction compute shader.
// Each thread handles one particle and inserts it into the appropriate cell.

struct Sim {
    colours   : u32,
    r         : f32,
    force     : f32,
    friction  : f32,
    beta      : f32,
    delta     : f32,
    avoidance : f32,
    worldSize : f32,
    border    : u32,
    vortex    : u32,
};

struct Particle {
    // xy = position, zw = velocity
    pos : vec4<f32>,
};

struct ListNode {
    // Index of the next particle in the same cell (-1 = end of list)
    next : atomic<i32>,
};

@group(0) @binding(0) var<uniform>            sim        : Sim;
@group(0) @binding(1) var<storage, read>      particles  : array<Particle>;
// heads[cellIndex] = index of first particle in that cell (atomically updated)
@group(0) @binding(2) var<storage, read_write> heads     : array<atomic<i32>>;
// linked list: one node per particle
@group(0) @binding(3) var<storage, read_write> linked    : array<ListNode>;

// Grid dimensions derived from worldSize and interaction radius r
// Cells are squares of side r; grid covers [-worldSize, worldSize] in both axes.
fn gridDim(sim_: Sim) -> u32 {
    return u32(ceil(sim_.worldSize * 2.0 / sim_.r)) + 1u;
}

// Map a world-space coordinate component to a grid cell index along one axis
fn cellCoord(v: f32, sim_: Sim) -> u32 {
    let half = sim_.worldSize;
    let clamped = clamp(v, -half, half);
    let norm = (clamped + half) / (sim_.r);
    return u32(floor(norm));
}

// FNV-1a-inspired hash for a 2-D cell coordinate
fn hash2u(x: u32, y: u32) -> u32 {
    var h : u32 = 2166136261u;
    h ^= x; h *= 16777619u;
    h ^= y; h *= 16777619u;
    return h;
}

fn cellIndex(cx: u32, cy: u32, dim: u32) -> u32 {
    return cy * dim + cx;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let count = arrayLength(&particles);
    if idx >= count {
        return;
    }

    let p = particles[idx];
    let dim = gridDim(sim);
    let cx = cellCoord(p.pos.x, sim);
    let cy = cellCoord(p.pos.y, sim);
    let cell = cellIndex(cx, cy, dim);

    // Atomically insert this particle at the head of the cell's linked list
    let prev = atomicExchange(&heads[cell], i32(idx));
    atomicStore(&linked[idx].next, prev);
}
