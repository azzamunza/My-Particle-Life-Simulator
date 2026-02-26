// Render shaders for particle life simulator
// Particles are rendered as small circular quads.
// Each instance corresponds to one particle (position, velocity, colour index).

struct Camera {
    pos   : vec2<f32>,
    zoom  : f32,
    _pad  : f32,
};

struct VertexOut {
    @builtin(position) pos   : vec4<f32>,
    @location(0)       uv    : vec2<f32>,
    @location(1)       color : vec3<f32>,
};

// Particle layout: vec4<f32> = (pos.x, pos.y, vel.x, vel.y)
@group(0) @binding(0) var<storage, read> particles  : array<vec4<f32>>;
// Colour index per particle (u32 packed as f32 for simplicity)
@group(0) @binding(1) var<storage, read> colourIdx  : array<u32>;
// Palette: one vec4<f32> per colour type (rgb + padding)
@group(0) @binding(2) var<storage, read> palette    : array<vec4<f32>>;
@group(0) @binding(3) var<uniform>       camera     : Camera;

// Unit-quad corners (two triangles forming a quad)
const QUAD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
);

// Particle radius in world units
const RADIUS : f32 = 0.6;

@vertex
fn vs_main(
    @builtin(vertex_index)   vi : u32,
    @builtin(instance_index) ii : u32,
) -> VertexOut {
    let particle = particles[ii];
    let worldPos = particle.xy;

    let corner = QUAD[vi];
    let offset = corner * RADIUS;

    // World → clip space
    let viewPos = (worldPos - camera.pos) * camera.zoom;
    let clipPos = viewPos + offset * camera.zoom;

    var out : VertexOut;
    out.pos   = vec4<f32>(clipPos, 0.0, 1.0);
    out.uv    = corner;
    out.color = palette[colourIdx[ii]].rgb;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let dist = length(in.uv);
    if dist > 1.0 {
        discard;
    }
    // Soft edge falloff
    let alpha = 1.0 - smoothstep(0.7, 1.0, dist);
    return vec4<f32>(in.color * alpha, alpha);
}
