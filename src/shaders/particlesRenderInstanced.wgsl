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

@group(0) @binding(0) var<storage, read> particles : array<Particle>;
@group(0) @binding(1) var<uniform>       params    : Params;

fn quad_offset(vid : u32) -> vec2<f32> {
  var offsets = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(2.0, 0.0), vec2<f32>(0.0, 2.0),
    vec2<f32>(2.0, 0.0), vec2<f32>(2.0, 2.0), vec2<f32>(0.0, 2.0),
  );
  return offsets[vid];
}

@vertex
fn vs_main(
  @builtin(vertex_index)   vid : u32,
  @builtin(instance_index) iid : u32
) -> @builtin(position) vec4<f32> {
  let base = particles[iid].pos;
  let px   = base + quad_offset(vid);

  let x =  (px.x / params.width)  * 2.0 - 1.0;
  let y = -(px.y / params.height) * 2.0 + 1.0;

  return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  return vec4<f32>(0.506, 0.549, 0.973, 1.0);
}