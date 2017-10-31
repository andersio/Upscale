import MetalPerformanceShaders
import Metal

final class Transformer {
    private let device: MTLDevice
    private let inputState: MTLComputePipelineState
    private let outputState: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device

        let function = library.makeFunction(name: "input_transformer")!
        inputState = try device.makeComputePipelineState(function: function)
        
        let function2 = library.makeFunction(name: "output_transformer")!
        outputState = try device.makeComputePipelineState(function: function2)
    }

    func encodeInputTransform(to commandBuffer: MTLCommandBuffer, source: MPSImage) -> MPSImage {
        assert(source.texture.pixelFormat == .bgra8Unorm)

        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(inputState)
        
        encoder.setTexture(source.texture, index: 0)
        encoder.useResource(source.texture, usage: [.read])

        let descriptor = MPSImageDescriptor(channelFormat: .float16, width: source.width, height: source.height, featureChannels: 3)
        descriptor.usage = [.shaderRead, .shaderWrite]
        
        let output = MPSImage(device: device, imageDescriptor: descriptor)
        encoder.setTexture(output.texture, index: 1)
        encoder.useResource(output.texture, usage: [.write])

        let threadGroupSize = MTLSizeMake(inputState.threadExecutionWidth,
                                          inputState.maxTotalThreadsPerThreadgroup / inputState.threadExecutionWidth,
                                          1)
        encoder.dispatchThreads(MTLSizeMake(descriptor.width, descriptor.height, 1),
                                threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
        
        return output
    }
    
    func encodeOutputTransform(to commandBuffer: MTLCommandBuffer, source: MPSImage, slice: Int = 0) -> MPSImage {
        assert(source.texture.pixelFormat == .rgba16Float)
        
        let sourceTexture: MTLTexture
        
        if source.textureType == .type2D {
            sourceTexture = source.texture
        } else {
            sourceTexture = source.texture.makeTextureView(pixelFormat: .rgba16Float,
                                                           textureType: .type2D,
                                                           levels: 0 ..< 1,
                                                           slices: slice ..< slice + 1)!
        }
        
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(outputState)
        
        encoder.setTexture(sourceTexture, index: 0)
        encoder.useResource(sourceTexture, usage: [.read])
        
        let descriptor = MTLTextureDescriptor()
        descriptor.pixelFormat = .rgba8Unorm
        descriptor.width = source.width
        descriptor.height = source.height
        descriptor.usage = [.shaderWrite, .shaderRead]
        
        let output = device.makeTexture(descriptor: descriptor)!
        encoder.setTexture(output, index: 1)
        encoder.useResource(output, usage: [.write])
        
        let threadGroupSize = MTLSizeMake(outputState.threadExecutionWidth,
                                          outputState.maxTotalThreadsPerThreadgroup / outputState.threadExecutionWidth,
                                          1)
        encoder.dispatchThreads(MTLSizeMake(descriptor.width, descriptor.height, 1),
                                threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
        
        return MPSImage(texture: output, featureChannels: 3)
    }
}

