import MetalPerformanceShaders
import Metal
import Accelerate
import CoreImage

// MPS format
// weight[outputChannels][kernelHeight][kernelWidth][inputChannels/groups]

final class SuperResolution {
    struct WeightsDescriptor {
        let hiddenLayer0Weight: URL
        let hiddenLayer1Weight: URL
        let hiddenLayer2Weight: URL
        let hiddenLayer0Bias: URL
        let hiddenLayer1Bias: URL
        let hiddenLayer2Bias: URL
    }

    let device: MTLDevice
    let library: MTLLibrary
    let queue: MTLCommandQueue
    let inputBatchSize: Int
    let subpixelScale: Int

    let transformer: Transformer
    let hiddenLayer0: TransposedConvolutionLayer
    let hiddenLayer1: TransposedConvolutionLayer
    let hiddenLayer2: TransposedConvolutionLayer
    let weights: WeightsDescriptor

    init(
        device: MTLDevice,
        inputBatchSize: Int = 1,
        subpixelScale: Int = 4,
        colorChannels: Int = 3,
        featureChannels: Int = 64,
        weights: WeightsDescriptor? = nil
    ) throws {
        self.device = device
        self.library = device.makeDefaultLibrary()!
        self.queue = device.makeCommandQueue(maxCommandBufferCount: 4)!
        self.inputBatchSize = inputBatchSize
        self.subpixelScale = subpixelScale
        self.weights = weights!

        // Transformer
        transformer = try Transformer(device: device, library: library, finalFeatureChannels: 48, scaleFactor: 4)

        // Hidden Layer 0
        hiddenLayer0 = TransposedConvolutionLayer(label: "h0",
                                                  device: device,
                                                  kernelSize: (1, 1),
                                                  inputBatchSize: inputBatchSize,
                                                  inputFeatureChannels: colorChannels,
                                                  outputFeatureChannels: featureChannels,
                                                  weights: weights!.hiddenLayer0Weight,
                                                  bias: weights!.hiddenLayer0Bias,
                                                  activation: .relu,
                                                  isIntermediate: true)

        // Hidden Layer 1
        hiddenLayer1 = TransposedConvolutionLayer(label: "h1",
                                                  device: device,
                                                  kernelSize: (5, 5),
                                                  inputBatchSize: inputBatchSize,
                                                  inputFeatureChannels: featureChannels,
                                                  outputFeatureChannels: featureChannels,
                                                  weights: weights!.hiddenLayer1Weight,
                                                  bias: weights!.hiddenLayer1Bias,
                                                  activation: .relu,
                                                  isIntermediate: true)

        // Subpixel Convolution Layer
        hiddenLayer2 = TransposedConvolutionLayer(label: "h2",
                                                   device: device,
                                                   kernelSize: (5, 5),
                                                   inputBatchSize: inputBatchSize,
                                                   inputFeatureChannels: featureChannels,
                                                   outputFeatureChannels: 48,
                                                   weights: weights!.hiddenLayer2Weight,
                                                bias: weights!.hiddenLayer2Bias,
                                                   activation: .tanh,
                                                   isIntermediate: true)
    }

    func infer(_ image: MPSImage, completion: @escaping (MPSImage, TimeInterval) -> Void) {
        assert(image.featureChannels == 3)

        let buffer = queue.makeCommandBuffer()!

        let transformed = transformer.encodeInputTransform(to: buffer, source: image)
        let outputH0 = hiddenLayer0.encode(to: buffer, source: transformed)
        let outputH1 = hiddenLayer1.encode(to: buffer, source: outputH0)
        let outputH2 = hiddenLayer2.encode(to: buffer, source: outputH1)
        let converted = transformer.encodeOutputTransform(to: buffer, source: outputH2)

        let encoder = buffer.makeBlitCommandEncoder()!
        #if os(macOS)
        encoder.synchronize(resource: transformed.texture)
        if outputH0.texture.storageMode != .private {
            encoder.synchronize(resource: outputH0.texture)
        }
        if outputH1.texture.storageMode != .private {
            encoder.synchronize(resource: outputH1.texture)
        }
        if outputH2.texture.storageMode != .private {
            encoder.synchronize(resource: outputH2.texture)
        }
        encoder.synchronize(resource: converted.texture)
        #endif
        encoder.endEncoding()

        #if os(macOS)
        let startTime = DispatchTime.now().uptimeNanoseconds
        #endif

        buffer.addCompletedHandler { _ in
            #if os(iOS)
                let executionTime = buffer.gpuEndTime - buffer.gpuStartTime
            #endif

            #if os(macOS)
                let _executionTime = DispatchTime.now().uptimeNanoseconds - startTime
                let executionTime = Double(_executionTime) / Double(NSEC_PER_SEC)
            #endif

            completion(converted, executionTime)
        }

        buffer.commit()
    }
}

extension Float {
    static var randomNormal: Float {
        let uint32Max = Double(UInt32.max)

        // Uniform distribution
        let u = Double(arc4random()) / uint32Max
        let v = Double(arc4random()) / uint32Max

        // Box-Muller Transform
        let x = sqrt(-2 * log(u)) * cos(2 * .pi * v);
        let y = x * 0.02

        return Float(y)
    }
}

func printImage(_ transformed: MPSImage, channel: Int = 0) {
    assert(transformed.pixelFormat == .rgba16Float)

    let input = UnsafeMutableRawPointer.allocate(bytes: transformed.width * transformed.height * 2 * 4,
                                                 alignedTo: 2)
    transformed.texture.getBytes(input, bytesPerRow: transformed.width * 8, from: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0), size: MTLSize(width: transformed.width, height: transformed.height, depth: 1)), mipmapLevel: 0)

    var src = vImage_Buffer(data: input,
                            height: 1,
                            width: UInt(4 * transformed.width * transformed.height),
                            rowBytes: transformed.width * transformed.height * 2 * 4)
    var floats: [Float] = Array(repeating: 0.0, count: transformed.width * transformed.height * 4)
    var dest = vImage_Buffer(data: &floats,
                             height: 1,
                             width: UInt(4 * transformed.width * transformed.height),
                             rowBytes: transformed.width * transformed.height * 4 * 4)
    assert(vImageConvert_Planar16FtoPlanarF(&src, &dest, 0) == kvImageNoError)

    input.deallocate(bytes: transformed.width * transformed.height * 2 * transformed.featureChannels, alignedTo: 2)
}
