struct Params {
  width  : u32,
  height : u32,
  flags  : u32,
  _pad   : u32,
};

@group(0) @binding(1) var srcTex : texture_2d<f32>;
@group(0) @binding(2) var dstTex : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params : Params;

fn in_bounds(x: u32, y: u32) -> bool {
  return x < params.width && y < params.height;
}

fn luma(rgb: vec3<f32>) -> f32 {
  return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (!in_bounds(x, y)) { return; }

  let c = textureLoad(srcTex, vec2<i32>(i32(x), i32(y)), 0);
  let g = luma(c.rgb);
  textureStore(dstTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(g, g, g, 1.0));
}