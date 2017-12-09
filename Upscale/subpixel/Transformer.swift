import MetalPerformanceShaders
import Metal

final class Transformer {
    private let device: MTLDevice
    private let unormInputState: MTLComputePipelineState
    private let f32InputState: MTLComputePipelineState
    private let outputState: MTLComputePipelineState

    private let finalFeatureChannels: Int
    private let scaleFactor: Int

    init(device: MTLDevice, library: MTLLibrary, finalFeatureChannels: Int, scaleFactor: Int) throws {
        self.device = device
        self.finalFeatureChannels = finalFeatureChannels
        self.scaleFactor = scaleFactor

        assert(finalFeatureChannels % 3 == 0 && (finalFeatureChannels / 3) % scaleFactor == 0)
        var r = Int(exactly: sqrt(Double(finalFeatureChannels / 3)))!

        #if os(iOS)
        var convertFromBgra = true
        #else
        var convertFromBgra = false
        #endif

        let inputTransformerArgs = MTLFunctionConstantValues()
        inputTransformerArgs.setConstantValue(&convertFromBgra, type: .bool, index: 1)
        let function = try! library.makeFunction(name: "input_transformer",
                                                  constantValues: inputTransformerArgs)
        unormInputState = try device.makeComputePipelineState(function: function)

        let function3 = library.makeFunction(name: "input_transformer_f32")!
        f32InputState = try device.makeComputePipelineState(function: function3)

        let values = MTLFunctionConstantValues()
        values.setConstantValue(&r, type: .int, index: 0)
        let function2 = try! library.makeFunction(name: "output_transformer",
                                                  constantValues: values)
        outputState = try device.makeComputePipelineState(function: function2)
    }

    func encodeInputTransform(to commandBuffer: MTLCommandBuffer, source: MPSImage) -> MPSImage {
        let state: MTLComputePipelineState

        switch source.texture.pixelFormat {
        case .bgra8Unorm:
            state = unormInputState
        case .rgba32Float:
            state = f32InputState
        default:
            fatalError()
        }

        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(state)
        
        encoder.setTexture(source.texture, index: 0)
        encoder.useResource(source.texture, usage: [.read])

        let descriptor = MPSImageDescriptor(channelFormat: .float16, width: source.width, height: source.height, featureChannels: 3)
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        let output = MPSImage(device: device, imageDescriptor: descriptor)
        encoder.setTexture(output.texture, index: 1)
        encoder.useResource(output.texture, usage: [.write])

        let threadGroupSize = MTLSizeMake(state.threadExecutionWidth,
                                          state.maxTotalThreadsPerThreadgroup / state.threadExecutionWidth,
                                          1)
        encoder.dispatchThreads(MTLSizeMake(descriptor.width, descriptor.height, 1),
                                threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
        
        return output
    }
    
    func encodeOutputTransform(to commandBuffer: MTLCommandBuffer, source: MPSImage, slice: Int = 0) -> MPSImage {
        assert(source.pixelFormat == .rgba16Float
            && source.featureChannels == finalFeatureChannels
            && source.textureType == .type2DArray)
        
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(outputState)

        encoder.setTexture(source.texture, index: 0)
        encoder.useResource(source.texture, usage: [.read])
        
        let descriptor = MTLTextureDescriptor()
        descriptor.pixelFormat = .rgba8Unorm
        descriptor.width = source.width * scaleFactor
        descriptor.height = source.height * scaleFactor
        descriptor.usage = [.shaderWrite, .shaderRead]
        
        let output = device.makeTexture(descriptor: descriptor)!
        encoder.setTexture(output, index: 1)
        encoder.useResource(output, usage: [.write])
        
        let threadGroupSize = MTLSizeMake(outputState.threadExecutionWidth,
                                          outputState.maxTotalThreadsPerThreadgroup / outputState.threadExecutionWidth,
                                          1)
        encoder.dispatchThreads(MTLSizeMake(source.width, source.height, 1),
                                threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()

        (source as? MPSTemporaryImage)?.readCount -= 1

        return MPSImage(texture: output, featureChannels: 3)
    }
}

