#include <metal_stdlib>
//#define SHOW_FEATURE_CHANNELS
using namespace metal;

constant int scaleFactor [[ function_constant(0) ]];
constant bool bgra_to_rgba [[ function_constant(1) ]];

kernel void input_transformer_f32(texture2d<float, access::read> input [[texture(0)]],
                              texture2d<float, access::write> output [[texture(1)]],
                              uint2 position [[thread_position_in_grid]]) {
    float4 rgba = input.read(position);
    output.write(rgba, position);
}

kernel void input_transformer(texture2d<float, access::read> input [[texture(0)]],
                              texture2d<float, access::write> output [[texture(1)]],
                              uint2 position [[thread_position_in_grid]]) {
    float4 rgba = input.read(position);
    if (bgra_to_rgba) {
        output.write(rgba.bgra * 255.0, position);
    } else {
        output.write(rgba * 255.0, position);
    }
}

kernel void output_transformer(texture2d_array<float, access::read> input [[texture(0)]],
                               texture2d<float, access::write> output [[texture(1)]],
                               uint2 position [[thread_position_in_grid]]) {
    // position.x: source x
    // position.y: source y
    // input.xy: source xy
    // input.z: N/4 feature channel slice

#ifdef SHOW_FEATURE_CHANNELS
    float4x4 r = {
        input.read(position, 11),
        input.read(position, 10),
        input.read(position, 9),
        input.read(position, 8)
    };

    for (int x = 0; x < scaleFactor; x++) {
        for (int y = 0; y < scaleFactor; y++) {
            float4 pixel = {r[x][y], r[x][y], r[x][y], 1.0};
            output.write(pixel, {32 * x + position.x, 32 * y + position.y});
        }
    }
#else

    uint2 origin = {position.x * scaleFactor, position.y * scaleFactor};

    for (int r = 0; r < scaleFactor; r++) {
        float4 r4 = (input.read(position, 0 + r) + 1.0) * 0.5;
        float4 g4 = (input.read(position, 4 + r) + 1.0) * 0.5;
        float4 b4 = (input.read(position, 8 + r) + 1.0) * 0.5;

        output.write({r4.x, g4.x, b4.x, 1.0},
                     {origin.x + r, origin.y});
        output.write({r4.y, g4.y, b4.y, 1.0},
                     {origin.x + r, origin.y + 1});
        output.write({r4.z, g4.z, b4.z, 1.0},
                     {origin.x + r, origin.y + 2});
        output.write({r4.w, g4.w, b4.w, 1.0},
                     {origin.x + r, origin.y + 3});
    }
#endif
}
