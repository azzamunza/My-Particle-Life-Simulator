// Cell Biology Simulator render shader
// Draws each active particle as a circle quad, coloured by ptype.

struct Camera {
    pos           : vec2<f32>,
    zoom          : f32,
    particleRadius: f32,
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

struct VertexOut {
    @builtin(position) pos   : vec4<f32>,
    @location(0)       uv    : vec2<f32>,
    @location(1)       color : vec3<f32>,
};

@group(0) @binding(0) var<storage, read> particles : array<Particle>;
@group(0) @binding(1) var<uniform>       camera    : Camera;

const QUAD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
);

const WORLD_SIZE : f32 = 200.0;

// HSL → RGB (h in [0,1])
fn hsl2rgb(h: f32, s: f32, l: f32) -> vec3<f32> {
    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let x = c * (1.0 - abs(fract(h * 3.0) * 2.0 - 1.0));
    let m = l - c * 0.5;
    var rgb: vec3<f32>;
    let hi = u32(floor(h * 6.0)) % 6u;
    switch hi {
        case 0u: { rgb = vec3<f32>(c, x, 0.0); }
        case 1u: { rgb = vec3<f32>(x, c, 0.0); }
        case 2u: { rgb = vec3<f32>(0.0, c, x); }
        case 3u: { rgb = vec3<f32>(0.0, x, c); }
        case 4u: { rgb = vec3<f32>(x, 0.0, c); }
        default: { rgb = vec3<f32>(c, 0.0, x); }
    }
    return rgb + vec3<f32>(m);
}

fn typeColor(ptype: u32) -> vec3<f32> {
    switch ptype {
        case 0u:  { return vec3<f32>(0.2, 0.8, 0.2); }  // MEMBRANE:  green
        case 1u:  { return vec3<f32>(1.0, 0.8, 0.0); }  // CHANNEL:   gold
        case 2u:  { return vec3<f32>(1.0, 0.2, 0.2); }  // PUMP:      red
        case 3u:  { return vec3<f32>(0.5, 0.5, 1.0); }  // NUCLEUS:   blue
        case 4u:  { return vec3<f32>(0.8, 1.0, 0.8); }  // DNA_A:     light green
        case 5u:  { return vec3<f32>(0.8, 0.8, 1.0); }  // DNA_B:     light blue
        case 6u:  { return vec3<f32>(1.0, 0.6, 0.0); }  // RIBOSOME:  orange
        case 7u:  { return vec3<f32>(0.3, 0.7, 1.0); }  // CYTOPLASM: cyan
        case 8u:  { return vec3<f32>(0.6, 1.0, 0.6); }  // CILIA:     pale green
        case 9u:  { return vec3<f32>(0.9, 0.9, 0.3); }  // FLAGELLUM: yellow
        case 10u: { return vec3<f32>(0.8, 0.5, 0.9); }  // PSEUDOPOD: purple
        default: {
            if ptype >= 11u && ptype <= 17u {
                // NUTRIENT 1-7: rainbow
                let h = f32(ptype - 11u) / 7.0;
                return hsl2rgb(h, 0.9, 0.6);
            }
            if ptype == 18u { return vec3<f32>(0.4, 0.4, 0.4); }  // WASTE: grey
            return vec3<f32>(0.1, 0.1, 0.1);
        }
    }
}

@vertex
fn vs_main(
    @builtin(vertex_index)   vi : u32,
    @builtin(instance_index) ii : u32,
) -> VertexOut {
    let p = particles[ii];
    var out: VertexOut;

    // Inactive particles: push off screen
    if p.ptype == 255u || (p.flags & 1u) == 0u {
        out.pos   = vec4<f32>(2.0, 2.0, 0.0, 1.0);
        out.uv    = vec2<f32>(0.0);
        out.color = vec3<f32>(0.0);
        return out;
    }

    let corner  = QUAD[vi];
    let offset  = corner * camera.particleRadius;

    // World → clip: scale by zoom (world is ±200), then divide by canvas half-height
    let viewPos = (p.pos - camera.pos) * camera.zoom;
    let clipPos = viewPos + offset;

    out.pos   = vec4<f32>(clipPos, 0.0, 1.0);
    out.uv    = corner;
    out.color = typeColor(p.ptype);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let dist = length(in.uv);
    if dist > 1.0 { discard; }
    let alpha = 1.0 - smoothstep(0.7, 1.0, dist);
    return vec4<f32>(in.color * alpha, alpha);
}
