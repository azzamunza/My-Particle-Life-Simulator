// Linked-list physics simulation compute shader.
// For each particle, walks nearby cells using the linked list,
// accumulates forces, then integrates velocity and position.

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
    pos : vec4<f32>,   // xy = position, zw = velocity
};

struct ListNode {
    next : atomic<i32>,
};

@group(0) @binding(0) var<uniform>            sim        : Sim;
// read from the "current" buffer
@group(0) @binding(1) var<storage, read>      particles  : array<Particle>;
// write updated state to the "next" buffer (ping-pong)
@group(0) @binding(2) var<storage, read_write> output    : array<Particle>;
// interaction matrix: force coefficient between colour types i and j
// stored as a flat array of size colours*colours (f32 each)
@group(0) @binding(3) var<storage, read>      matrix     : array<f32>;
// spatial acceleration structures
@group(0) @binding(4) var<storage, read>      heads      : array<i32>;
@group(0) @binding(5) var<storage, read>      linked     : array<i32>;
// colour index per particle
@group(0) @binding(6) var<storage, read>      colourIdx  : array<u32>;

// Replicate helpers from construct shader
fn gridDim(sim_: Sim) -> u32 {
    return u32(ceil(sim_.worldSize * 2.0 / sim_.r)) + 1u;
}

fn cellCoord(v: f32, sim_: Sim) -> i32 {
    let half = sim_.worldSize;
    let clamped = clamp(v, -half, half);
    return i32(floor((clamped + half) / sim_.r));
}

fn cellIndex(cx: i32, cy: i32, dim: i32) -> i32 {
    return cy * dim + cx;
}

// Attraction / repulsion force curve
// Returns signed force along the axis between two particles.
fn forceCurve(dist: f32, beta: f32, coeff: f32) -> f32 {
    if dist < beta {
        // Short-range repulsion (independent of coeff)
        return dist / beta - 1.0;
    } else if dist < 1.0 {
        // Mid-range attraction / repulsion controlled by coeff
        return coeff * (1.0 - abs(2.0 * dist - 1.0 - beta) / (1.0 - beta));
    }
    return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let count = arrayLength(&particles);
    if idx >= u32(count) {
        return;
    }

    let p   = particles[idx];
    var pos = p.pos.xy;
    var vel = p.pos.zw;

    let myColour = colourIdx[idx];
    let dim      = i32(gridDim(sim));
    let r        = sim.r;
    let beta     = sim.beta;
    let colours  = sim.colours;

    let cx = cellCoord(pos.x, sim);
    let cy = cellCoord(pos.y, sim);

    var acc = vec2<f32>(0.0, 0.0);

    // Iterate over 3×3 neighbourhood of cells
    for (var dy: i32 = -1; dy <= 1; dy++) {
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let nx = cx + dx;
            let ny = cy + dy;
            if nx < 0 || ny < 0 || nx >= dim || ny >= dim {
                continue;
            }
            let cell = cellIndex(nx, ny, dim);
            var j = heads[cell];
            while j >= 0 {
                if u32(j) != idx {
                    let q    = particles[j];
                    var diff = q.pos.xy - pos;

                    // Shortest-path wrap for toroidal world
                    if sim.border == 1u {
                        let half = sim.worldSize;
                        if diff.x >  half { diff.x -= half * 2.0; }
                        if diff.x < -half { diff.x += half * 2.0; }
                        if diff.y >  half { diff.y -= half * 2.0; }
                        if diff.y < -half { diff.y += half * 2.0; }
                    }

                    let distSq = dot(diff, diff);
                    if distSq > 0.0 && distSq < r * r {
                        let dist  = sqrt(distSq);
                        let normD = dist / r;          // normalised [0,1]
                        let dir   = diff / dist;

                        let theirColour = colourIdx[j];
                        let coeff       = matrix[myColour * colours + theirColour];
                        let f           = forceCurve(normD, beta, coeff);

                        // Close-range avoidance boost
                        var avoidF = 0.0;
                        if dist < sim.avoidance {
                            avoidF = -(sim.avoidance / dist - 1.0) * 0.5;
                        }

                        acc += dir * (f + avoidF);
                    }
                }
                j = linked[j];
            }
        }
    }

    // Scale acceleration
    acc *= sim.force * r;

    // Optional vortex
    if sim.vortex == 1u {
        acc += vec2<f32>(-vel.y, vel.x) * 0.01;
    }

    // Integrate
    vel = vel * (1.0 - sim.friction) + acc * sim.delta;
    pos = pos + vel * sim.delta;

    // Border handling
    let half = sim.worldSize;
    if sim.border == 1u {
        // Toroidal wrap
        if pos.x >  half { pos.x -= half * 2.0; }
        if pos.x < -half { pos.x += half * 2.0; }
        if pos.y >  half { pos.y -= half * 2.0; }
        if pos.y < -half { pos.y += half * 2.0; }
    } else {
        // Reflect off walls
        if pos.x >  half { pos.x =  half; vel.x = -abs(vel.x); }
        if pos.x < -half { pos.x = -half; vel.x =  abs(vel.x); }
        if pos.y >  half { pos.y =  half; vel.y = -abs(vel.y); }
        if pos.y < -half { pos.y = -half; vel.y =  abs(vel.y); }
    }

    output[idx].pos = vec4<f32>(pos, vel);
}
