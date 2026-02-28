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

fn clamp_i(v: i32, lo: i32, hi: i32) -> i32 {
  return max(lo, min(hi, v));
}

fn load_rgb(x: i32, y: i32) -> vec3<f32> {
  let xi = clamp_i(x, 0, i32(params.width) - 1);
  let yi = clamp_i(y, 0, i32(params.height) - 1);
  return textureLoad(srcTex, vec2<i32>(xi, yi), 0).rgb;
}

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (!in_bounds(x, y)) { return; }

  let w0 = 0.227027;
  let w1 = 0.1945946;
  let w2 = 0.1216216;
  let w3 = 0.054054;
  let w4 = 0.016216;

  let ix = i32(x);
  let iy = i32(y);

  var acc: vec3<f32> = vec3<f32>(0.0);

  acc += load_rgb(ix,     iy) * w0;
  acc += load_rgb(ix - 1, iy) * w1;
  acc += load_rgb(ix + 1, iy) * w1;
  acc += load_rgb(ix - 2, iy) * w2;
  acc += load_rgb(ix + 2, iy) * w2;
  acc += load_rgb(ix - 3, iy) * w3;
  acc += load_rgb(ix + 3, iy) * w3;
  acc += load_rgb(ix - 4, iy) * w4;
  acc += load_rgb(ix + 4, iy) * w4;

  textureStore(dstTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(acc, 1.0));
}