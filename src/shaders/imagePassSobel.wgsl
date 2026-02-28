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

fn luma(rgb: vec3<f32>) -> f32 {
  return dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (!in_bounds(x, y)) { return; }

  let ix = i32(x);
  let iy = i32(y);

  let tl = luma(load_rgb(ix - 1, iy - 1));
  let  t = luma(load_rgb(ix + 0, iy - 1));
  let tr = luma(load_rgb(ix + 1, iy - 1));
  let l  = luma(load_rgb(ix - 1, iy + 0));
  let r  = luma(load_rgb(ix + 1, iy + 0));
  let bl = luma(load_rgb(ix - 1, iy + 1));
  let  b = luma(load_rgb(ix + 0, iy + 1));
  let br = luma(load_rgb(ix + 1, iy + 1));

  let gx = (-1.0 * tl) + (1.0 * tr)
         + (-2.0 * l ) + (2.0 * r )
         + (-1.0 * bl) + (1.0 * br);

  let gy = (-1.0 * tl) + (-2.0 * t) + (-1.0 * tr)
         + ( 1.0 * bl) + ( 2.0 * b) + ( 1.0 * br);

  let mag = sqrt(gx * gx + gy * gy);
  let e = clamp(mag * 1.2, 0.0, 1.0);

  textureStore(dstTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(e, e, e, 1.0));
}