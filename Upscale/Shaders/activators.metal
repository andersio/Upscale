#include <metal_stdlib>
using namespace metal;

constant float leak [[function_constant(0)]];

kernel void activator_leaky_relu(texture2d_array<float, access::read_write> values [[texture(0)]],
                                 uint3 position [[thread_position_in_grid]]) {
    if (position.x >= values.get_width() || position.y >= values.get_height() || position.z >= values.get_array_size()) {
        return;
    }

    float4 value = values.read(position.xy, position.z);
    float4 f1 = 0.5 * (1 + leak);
    float4 f2 = 0.5 * (1 - leak);
    float4 result = f1 * value + f2 * abs(value);
    values.write(result, position.xy, position.z);
}

kernel void activator_tanh(texture2d_array<float, access::read_write> values [[texture(0)]],
                           uint3 position [[thread_position_in_grid]]) {
    if (position.x >= values.get_width() || position.y >= values.get_height() || position.z >= values.get_array_size()) {
        return;
    }

    float4 value = values.read(position.xy, position.z);
    float4 result = tanh(value);
    values.write(result, position.xy, position.z);
}
