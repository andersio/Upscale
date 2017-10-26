#include <metal_stdlib>
using namespace metal;

half4 rgba_normalize(half4 i) {
    return i / 127.5 - 1.0;
}

half4 rgba_unnormalize(half4 i) {
    return (i + 1.0) * 127.5;
}

kernel void input_transformer(texture2d<half, access::read> input [[texture(0)]],
                              texture2d<half, access::write> output [[texture(1)]],
                              uint2 position [[thread_position_in_grid]]) {
    half4 bgra = input.read(position);
    half4 bgra_norm = rgba_normalize(bgra);
    output.write(bgra_norm, position);
}

kernel void output_transformer(texture2d<half, access::read> input [[texture(0)]],
                              texture2d<half, access::write> output [[texture(1)]],
                              uint2 position [[thread_position_in_grid]]) {
    half4 bgra = input.read(position);
    half4 rgba_unorm = rgba_unnormalize(bgra.bgra);
    output.write(rgba_unorm, position);
}

