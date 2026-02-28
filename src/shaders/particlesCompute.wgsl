struct Particle {
  pos : vec2<f32>,
  vel : vec2<f32>,
};

struct Params {
  width  : f32,
  height : f32,
  count  : u32,
  _pad   : u32,
};

@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<uniform>             params    : Params;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  var p = particles[i];

  p.pos = p.pos + p.vel;

  if (p.pos.x < 0.0)           { p.pos.x = 0.0;            p.vel.x = -p.vel.x; }
  if (p.pos.x > params.width)  { p.pos.x = params.width;   p.vel.x = -p.vel.x; }
  if (p.pos.y < 0.0)           { p.pos.y = 0.0;            p.vel.y = -p.vel.y; }
  if (p.pos.y > params.height) { p.pos.y = params.height;  p.vel.y = -p.vel.y; }

  particles[i] = p;
}