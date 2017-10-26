import MetalPerformanceShaders
import Metal

// MPS format
// weight[outputChannels][kernelHeight][kernelWidth][inputChannels/groups]

final class SuperResolution {
    struct WeightsDescriptor {
        let hiddenLayer0Weight: URL
        let hiddenLayer1Weight: URL
        let subpixelLayerWeight: URL
        let hiddenLayer0Bias: URL
        let hiddenLayer1Bias: URL
        let subpixelLayerBias: URL
    }

    let device: MTLDevice
    let library: MTLLibrary
    let queue: MTLCommandQueue
    let inputBatchSize: Int
    let subpixelScale: Int

    let transformer: Transformer
    let hiddenLayer0: TransposedConvolutionLayer
    let hiddenLayer1: TransposedConvolutionLayer
    let subpixelLayer: SubpixelConvolutionLayer

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
        
        // Transformer
        transformer = try Transformer(device: device, library: library)

        // Hidden Layer 0
        hiddenLayer0 = TransposedConvolutionLayer(device: device,
                                                  kernelSize: (1, 1),
                                                  inputBatchSize: inputBatchSize,
                                                  inputFeatureChannels: colorChannels,
                                                  outputFeatureChannels: featureChannels,
                                                  weights: weights!.hiddenLayer0Weight,
                                                  bias: weights!.hiddenLayer0Bias)

        // Hidden Layer 1
        hiddenLayer1 = TransposedConvolutionLayer(device: device,
                                                  kernelSize: (5, 5),
                                                  inputBatchSize: inputBatchSize,
                                                  inputFeatureChannels: featureChannels,
                                                  outputFeatureChannels: featureChannels,
                                                  weights: weights!.hiddenLayer1Weight,
                                                  bias: weights!.hiddenLayer1Bias)

        // Subpixel Convolution Layer
        subpixelLayer = SubpixelConvolutionLayer(device: device,
                                                 kernelSize: (5, 5),
                                                 inputBatchSize: inputBatchSize,
                                                 inputFeatureChannels: featureChannels,
                                                 subpixelScale: subpixelScale,
                                                 outputColorChannels: colorChannels,
                                                 weights: weights!.subpixelLayerWeight,
                                                 bias: weights!.subpixelLayerBias)
    }

    func infer(_ image: MPSImage, completion: @escaping (MPSImage) -> Void) {
        assert(image.featureChannels == 3)

        let buffer = queue.makeCommandBuffer()!
        let transformed = transformer.encodeInputTransform(to: buffer, source: image)
        let outputH0 = hiddenLayer0.encode(to: buffer, source: transformed)
        let outputH1 = hiddenLayer1.encode(to: buffer, source: outputH0)
        let finalOutput = subpixelLayer.encode(to: buffer, source: outputH1)
        let converted = transformer.encodeOutputTransform(to: buffer, source: finalOutput)

        buffer.addCompletedHandler { _ in
            completion(converted)
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
