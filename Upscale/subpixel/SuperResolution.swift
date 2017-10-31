import MetalPerformanceShaders
import Metal
import Accelerate

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
    let testLayer: TestLayer
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

        // Test Layer
        testLayer = TestLayer(label: "t0", device: device, inputBatchSize: inputBatchSize, inputFeatureChannels: colorChannels, outputFeatureChannels: colorChannels)
        
        // Hidden Layer 0
        hiddenLayer0 = TransposedConvolutionLayer(label: "h0",
                                                  device: device,
                                                  kernelSize: (1, 1),
                                                  inputBatchSize: inputBatchSize,
                                                  inputFeatureChannels: colorChannels,
                                                  outputFeatureChannels: featureChannels,
                                                  weights: weights!.hiddenLayer0Weight,
                                                  bias: weights!.hiddenLayer0Bias)

        // Hidden Layer 1
        hiddenLayer1 = TransposedConvolutionLayer(label: "h1",
                                                  device: device,
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
        //let finalOutput = testLayer.encode(to: buffer, source: transformed)
        let outputH0 = hiddenLayer0.encode(to: buffer, source: transformed)
        let outputH1 = hiddenLayer1.encode(to: buffer, source: outputH0)
        let finalOutput = subpixelLayer.encode(to: buffer, source: outputH1)
        let converted = transformer.encodeOutputTransform(to: buffer, source: finalOutput)

        buffer.addCompletedHandler { _ in
            //printImage(outputH1)
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

func printImage(_ transformed: MPSImage) {
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
    for i in 0 ..< (floats.count / 4) {
        print("\(floats[i << 2]) \(floats[i << 2 + 1]) \(floats[i << 2 + 2]) \(floats[i << 2 + 3])")
    }

    input.deallocate(bytes: transformed.width * transformed.height * 2 * transformed.featureChannels, alignedTo: 2)
}
