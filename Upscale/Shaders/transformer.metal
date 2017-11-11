#include <metal_stdlib>
//#define SHOW_FEATURE_CHANNELS
using namespace metal;

kernel void input_transformer_f32(texture2d<float, access::read> input [[texture(0)]],
                              texture2d<float, access::write> output [[texture(1)]],
                              uint2 position [[thread_position_in_grid]]) {
    float4 rgba = input.read(position);
    output.write(rgba, position);
}
kernel void input_transformer(texture2d<float, access::read> input [[texture(0)]],
                              texture2d<float, access::write> output [[texture(1)]],
                              uint2 position [[thread_position_in_grid]]) {
    float4 bgra = input.read(position);
    output.write(bgra.bgra * 255, position);
}

constant int scaleFactor [[ function_constant(0) ]];

kernel void output_transformer(texture2d_array<float, access::read> input [[texture(0)]],
                               texture2d<float, access::write> output [[texture(1)]],
                               uint2 position [[thread_position_in_grid]]) {
    // position.x: source x
    // position.y: source y
    // input.xy: source xy
    // input.z: N/4 feature channel slice

    float4x4 _r = {
        input.read(position, 11),
        input.read(position, 10),
        input.read(position, 9),
        input.read(position, 8)
    };
#ifdef SHOW_FEATURE_CHANNELS
    for (int x = 0; x < scaleFactor; x++) {
        for (int y = 0; y < scaleFactor; y++) {
            float4 pixel = {_r[x][y], _r[x][y], _r[x][y], 1.0};
            output.write(tanh(pixel), {32 * x + position.x, 32 * y + position.y});
        }
    }
#else

 uint2 origin = {position.x * 4, position.y * 4};

    float4x4 _g = {
        input.read(position, 7),
        input.read(position, 6),
        input.read(position, 5),
        input.read(position, 4)
    };

    float4x4 _b = {
        input.read(position, 3),
        input.read(position, 2),
        input.read(position, 1),
        input.read(position, 0)
    };

    float4x4 r = transpose(_r);
    float4x4 g = transpose(_g);
    float4x4 b = transpose(_b);

    for (int x = 0; x < scaleFactor; x++) {
        for (int y = 0; y < scaleFactor; y++) {
            float4 pixel = {r[x][y], g[x][y], b[x][y], 1.0};
            output.write(tanh(pixel), {origin.x + x, origin.y + y});
        }
    }
#endif
}

float4x4 transpose(float4x4 org) {
    return {
        {org[0][0], org[1][0], org[2][0], org[3][0]},
        {org[0][1], org[1][1], org[2][1], org[3][1]},
        {org[0][2], org[1][2], org[2][2], org[3][2]},
        {org[0][3], org[1][3], org[2][3], org[3][3]}
    };
}
