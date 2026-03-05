// ============================================================
// Cell Biology Simulator — constants and buffer layout helpers
// ============================================================

// ---- Capacity limits ----
export const MAX_PARTICLES  = 4096;
export const MAX_BONDS      = 6144;
export const MAX_CHAIN_LEN  = 16;    // max nodes per cilia/flagellum chain

// ---- Cell geometry (world units, world is ~800×800) ----
export const CELL_RADIUS    = 80.0;
export const NUCLEUS_RADIUS = 28.0;  // 35% of cell radius
export const MEMBRANE_NODES = 60;
export const NUCLEUS_NODES  = 24;

// ---- Particle counts ----
export const DNA_STRANDS    = 2;
export const DNA_RUNGS      = 20;    // 20 rungs × 2 strands = 80 DNA particles
export const RIBOSOME_COUNT = 5;     // 5 × 4 particles = 20
export const CILIA_COUNT    = 8;     // 8 chains × 6 nodes = 48
export const FLAGELLUM_LEN  = 12;    // 12 nodes
export const PSEUDOPOD_COUNT= 3;     // 3 × 8 nodes = 24
export const CYTOPLASM_MAX  = 400;
export const NUTRIENT_COUNT = 500;
export const WASTE_MAX      = 80;
export const SOLUTION_COUNT = 1200;  // liquid solution particles that fill empty space

// ---- Cell division ----
export const DIVISION_THRESHOLD = 200; // cytoplasm count that triggers cell division

// ---- Physics ----
export const DT             = 0.016; // time step (~60fps)
export const XPBD_ITERS     = 4;     // constraint solver iterations
export const BROWNIAN_STR   = 1.5;   // Brownian motion strength (only for cytoplasm/nutrients/solution)
export const DAMPING        = 0.96;  // velocity damping per tick
export const MEMBRANE_K     = 1200.0;// membrane spring stiffness
export const NUCLEUS_K      = 900.0;
export const DNA_BACKBONE_K = 500.0;
export const DNA_RUNG_K     = 300.0;
export const RIBOSOME_K     = 900.0; // stiff
export const CILIA_K        = 200.0;
export const PRESSURE_K     = 80.0;  // radial pressure strength
export const CHANNEL_RADIUS = 22.0;  // nutrient scan radius (world units)
export const CYTOPLASM_LIFE = 600;   // ticks before cytoplasm becomes waste
export const CILIA_WAVE_STRENGTH    = 0.8;
export const FLAGELLUM_WAVE_STRENGTH= 1.0;

// ---- Particle type constants ----
export const PTYPE_MEMBRANE   = 0;
export const PTYPE_CHANNEL    = 1;   // gold membrane protein
export const PTYPE_PUMP       = 2;   // red membrane protein
export const PTYPE_NUCLEUS    = 3;
export const PTYPE_DNA_A      = 4;   // DNA strand alpha
export const PTYPE_DNA_B      = 5;   // DNA strand beta
export const PTYPE_RIBOSOME   = 6;
export const PTYPE_CYTOPLASM  = 7;
export const PTYPE_CILIA      = 8;
export const PTYPE_FLAGELLUM  = 9;
export const PTYPE_PSEUDOPOD  = 10;
export const PTYPE_NUTRIENT_1 = 11;
export const PTYPE_WASTE      = 18;
export const PTYPE_SOLUTION   = 19;  // liquid solution / extracellular fluid
export const PTYPE_INACTIVE   = 255; // slot is in the free list

// ---- Particle struct (12 × f32/u32 = 48 bytes) ----
// Offsets in u32/f32 units (index within the flat typed array view):
//   [0,1]  pos      vec2<f32>
//   [2,3]  vel      vec2<f32>
//   [4,5]  force    vec2<f32>
//   [6]    ptype    u32
//   [7]    flags    u32  (ACTIVE=1, BONDED=2)
//   [8]    age      f32
//   [9]    phase    f32
//   [10]   chainIdx u32
//   [11]   _pad     u32
export const PARTICLE_STRIDE = 48; // bytes
export const PARTICLE_FLOATS = 12; // values (u32/f32 union view)

// ---- Bond struct (4 × f32/u32 = 16 bytes) ----
//   [0]  a         u32
//   [1]  b         u32
//   [2]  restLen   f32
//   [3]  stiffness f32
export const BOND_STRIDE  = 16; // bytes
export const BOND_UINTS   = 4;  // values

// ---- Spatial grid ----
export const WORLD_SIZE     = 800.0; // world spans -800 … +800 on both axes
export const GRID_CELL_SIZE = CHANNEL_RADIUS; // 22.0
export const GRID_DIM       = Math.ceil((WORLD_SIZE * 2) / GRID_CELL_SIZE) + 1; // 74
export const GRID_CELLS     = GRID_DIM * GRID_DIM; // 5476

// ---- Fixed-point scale for atomic force / XPBD accumulation ----
export const FORCE_FP_SCALE = 1024.0;
export const XPBD_FP_SCALE  = 65536.0;

// ---- Uniform buffer layout (64 bytes = 16 × u32) ----
// Indices into a Uint32Array view of the uniform buffer:
export const U_TICK          = 0;
export const U_NUM_PARTICLES = 1;
export const U_NUM_BONDS     = 2;
export const U_PAD0          = 3;
// Float fields (same array, interpreted as f32):
export const U_DT            = 4;
export const U_BROWNIAN_STR  = 5;
export const U_DAMPING       = 6;
export const U_PRESSURE_K    = 7;
export const U_MEMBRANE_K    = 8;
export const U_NUCLEUS_K     = 9;
export const U_CILIA_WAVE    = 10;
export const U_FLAG_WAVE     = 11;
export const U_CHANNEL_R     = 12;
export const U_CYTOPLASM_LIFE= 13; // u32
export const U_PAD1          = 14;
export const U_PAD2          = 15;
export const UNIFORM_SIZE    = 64; // bytes
