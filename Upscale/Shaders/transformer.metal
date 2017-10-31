#include <metal_stdlib>
using namespace metal;

kernel void input_transformer(texture2d<float, access::read> input [[texture(0)]],
                              texture2d<float, access::write> output [[texture(1)]],
                              uint2 position [[thread_position_in_grid]]) {
    float4 bgra = input.read(position);
    float4 scaled_bgra = bgra;// * 2.0 - 1.0;
    output.write(scaled_bgra, position);
}

kernel void output_transformer(texture2d<float, access::read> input [[texture(0)]],
                              texture2d<float, access::write> output [[texture(1)]],
                              uint2 position [[thread_position_in_grid]]) {
    float4 bgra = input.read(position);
    output.write(bgra, position);
}

