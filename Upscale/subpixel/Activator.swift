import MetalPerformanceShaders
import Metal

final class Activator {
    enum Kind {
        case leakyRELU(Float)
        case tanh
    }

    private let device: MTLDevice
    private let state: MTLComputePipelineState
    let kind: Kind

    init(device: MTLDevice, library: MTLLibrary, kind: Kind) throws {
        self.device = device
        self.kind = kind

        let function: MTLFunction

        switch kind {
        case .leakyRELU(var leak):
            let constants = MTLFunctionConstantValues()
            constants.setConstantValue(&leak, type: .float, index: 0)
            function = try library.makeFunction(name: "activator_leaky_relu", constantValues: constants)
        case .tanh:
            function = library.makeFunction(name: "activator_tanh")!
        }

        state = try device.makeComputePipelineState(function: function)
    }

    func encode(to commandBuffer: MTLCommandBuffer, modifying texture: MTLTexture) {
        assert(texture.depth == 1)
        assert(texture.textureType == .type2DArray)

        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setTexture(texture, index: 0)
        encoder.useResource(texture, usage: [.read, .write])

        let threadGroupSize = MTLSizeMake(state.threadExecutionWidth,
                                          state.maxTotalThreadsPerThreadgroup / state.threadExecutionWidth,
                                          1)
        encoder.dispatchThreads(MTLSizeMake(texture.width, texture.height, texture.arrayLength),
                                threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
    }
}

