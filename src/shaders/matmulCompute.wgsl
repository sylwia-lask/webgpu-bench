
struct Params {
  n : u32,
};

@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<uniform> params : Params;

const TILE : u32 = 16u;

var<workgroup> As : array<f32, 256>;
var<workgroup> Bs : array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn cs_main(
@builtin(global_invocation_id) gid : vec3 < u32>,
@builtin(local_invocation_id) lid : vec3 < u32>
)
{
  let n = params.n;

  let row = gid.y;
  let col = gid.x;

  //IMPORTANT:
  //no early return before barriers → must stay uniform
  let inBounds = (row < n) && (col < n);

  var acc : f32 = 0.0;

  let numTiles = (n + TILE - 1u) / TILE;

  for (var t : u32 = 0u; t < numTiles; t = t + 1u)
  {

    let aCol = t * TILE + lid.x;
    let bRow = t * TILE + lid.y;

    if (inBounds && aCol < n)
    {
      As[lid.y * TILE + lid.x] = A[row * n + aCol];
    } else {
      As[lid.y * TILE + lid.x] = 0.0;
    }

    if (inBounds && bRow < n)
    {
      Bs[lid.y * TILE + lid.x] = B[bRow * n + col];
    } else {
      Bs[lid.y * TILE + lid.x] = 0.0;
    }

    workgroupBarrier();

    for (var k : u32 = 0u; k < TILE; k = k + 1u)
    {
      acc = acc + As[lid.y * TILE + k] * Bs[k * TILE + lid.x];
    }

    workgroupBarrier();
  }

  //Store result only if inside matrix
  if (inBounds)
  {
    C[row * n + col] = acc;
  }
}
