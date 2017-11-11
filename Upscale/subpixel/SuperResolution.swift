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
    let subpixelLayer: TransposedConvolutionLayer
    let activator: Activator
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

        activator = try Activator(device: device, library: library, kind: .leakyRELU(0.2))

        // Transformer
        transformer = try Transformer(device: device, library: library, finalFeatureChannels: 48, scaleFactor: 4)

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
        subpixelLayer = TransposedConvolutionLayer(label: "h2",
                                                   device: device,
                                                   kernelSize: (5, 5),
                                                   inputBatchSize: inputBatchSize,
                                                   inputFeatureChannels: featureChannels,
                                                   outputFeatureChannels: 48,
                                                   weights: weights!.subpixelLayerWeight,
                                                   bias: weights!.subpixelLayerBias)
    }

    func infer(_ image: MPSImage, completion: @escaping (MPSImage) -> Void) {
        assert(image.featureChannels == 3)

        let buffer = queue.makeCommandBuffer()!
        let neuron = MPSCNNNeuronReLU(device: device, a: 0.2)
        neuron.destinationImageAllocator = MPSImage.defaultAllocator()

        let transformed = transformer.encodeInputTransform(to: buffer, source: image)
        let h0 = hiddenLayer0.encode(to: buffer, source: transformed)
        let outputH0 = neuron.encode(commandBuffer: buffer, sourceImage: h0)

        let h1 = hiddenLayer1.encode(to: buffer, source: outputH0)
        let outputH1 = neuron.encode(commandBuffer: buffer, sourceImage: h1)

        let finalOutput = subpixelLayer.encode(to: buffer, source: outputH1)
        let converted = transformer.encodeOutputTransform(to: buffer, source: finalOutput)

        let encoder = buffer.makeBlitCommandEncoder()!
        encoder.synchronize(resource: transformed.texture)
        encoder.synchronize(resource: outputH0.texture)
        encoder.synchronize(resource: outputH1.texture)
        encoder.synchronize(resource: finalOutput.texture)
        encoder.synchronize(resource: converted.texture)
        encoder.endEncoding()

        buffer.commit()
        buffer.waitUntilCompleted()

            print("IN_R")
            printImage(transformed, channel: 0)
            print("IN_G")
            printImage(transformed, channel: 1)
            print("IN_B")
            printImage(transformed, channel: 2)
            print("H0_R")
            printImage(outputH0, channel: 0)
            print("H0_G")
            printImage(outputH0, channel: 1)
            print("H0_B")
            printImage(outputH0, channel: 2)
            print("H1_R")
            printImage(outputH1, channel: 0)
            print("H1_G")
            printImage(outputH1, channel: 1)
            print("H1_B")
            printImage(outputH1, channel: 2)
            print("H2_R")
            printImage(finalOutput, channel: 0)
            print("H2_G")
            printImage(finalOutput, channel: 1)
            print("H2_B")
            printImage(finalOutput, channel: 2)

            completion(converted)
    }

    func cpuInfer() -> CIImage {
        let floatSize = MemoryLayout<Float>.size

        func loadTensor(_ url: URL, size: Int) -> BNNSLayerData {
            let byteSize = size * floatSize
            let raw = UnsafeMutableRawPointer.allocate(bytes: byteSize, alignedTo: floatSize)
            try! Data(contentsOf: url)
                .copyBytes(to: raw.assumingMemoryBound(to: UInt8.self), count: byteSize)
            return BNNSLayerData(data: raw, data_type: .float)
        }

        let input = loadTensor(Bundle.main.url(forResource: "in_test_bnns", withExtension: nil)!, size: 32 * 32 * 3)
        let h0w = loadTensor(Bundle.main.url(forResource: "w_h0_bnns", withExtension: nil)!, size: 3 * 64 * 1 * 1)
        let h0b = loadTensor(weights.hiddenLayer0Bias, size: 64)
        let h1w = loadTensor(Bundle.main.url(forResource: "w_h1_bnns", withExtension: nil)!, size: 64 * 64 * 5 * 5)
        let h1b = loadTensor(weights.hiddenLayer1Bias, size: 64)
        let h2w = loadTensor(Bundle.main.url(forResource: "w_h2_bnns", withExtension: nil)!, size: 64 * 48 * 5 * 5)
        let h2b = loadTensor(weights.subpixelLayerBias, size: 48)

        defer {
            input.data!.deallocate(bytes: 32 * 32 * 3 * floatSize, alignedTo: floatSize)
            h0w.data!.deallocate(bytes: 3 * 64 * floatSize, alignedTo: floatSize)
            h1w.data!.deallocate(bytes: 64 * 64 * 5 * 5 * floatSize, alignedTo: floatSize)
            h2w.data!.deallocate(bytes: 64 * 48 * 5 * 5 * floatSize, alignedTo: floatSize)
            h0b.data!.deallocate(bytes: 64 * floatSize, alignedTo: floatSize)
            h1b.data!.deallocate(bytes: 64 * floatSize, alignedTo: floatSize)
            h2b.data!.deallocate(bytes: 48 * floatSize, alignedTo: floatSize)
        }

        let lrelu = BNNSActivation(function: .leakyRectifiedLinear, alpha: 0.2, beta: 1.0)
        let tanh = BNNSActivation(function: .tanh)
        var filterParams = BNNSFilterParameters()

        var input0 = BNNSImageStackDescriptor(width: 32, height: 32, channels: 3, row_stride: 32, image_stride: 32*32, data_type: .float)
        var output0 = BNNSImageStackDescriptor(width: 32, height: 32, channels: 64, row_stride: 32, image_stride: 32*32, data_type: .float)
        var params0 = BNNSConvolutionLayerParameters(x_stride: 1, y_stride: 1, x_padding: 0, y_padding: 0, k_width: 1, k_height: 1, in_channels: 3, out_channels: 64, weights: h0w, bias: h0b, activation: lrelu)
        let h0 = BNNSFilterCreateConvolutionLayer(&input0, &output0, &params0, &filterParams)

        var output1 = BNNSImageStackDescriptor(width: 32, height: 32, channels: 64, row_stride: 32, image_stride: 32*32, data_type: .float)
        var params1 = BNNSConvolutionLayerParameters(x_stride: 1, y_stride: 1, x_padding: 4, y_padding: 4, k_width: 5, k_height: 5, in_channels: 64, out_channels: 64, weights: h1w, bias: h1b, activation: lrelu)
        let h1 = BNNSFilterCreateConvolutionLayer(&output0, &output1, &params1, &filterParams)

        var output2 = BNNSImageStackDescriptor(width: 32, height: 32, channels: 64, row_stride: 32, image_stride: 32*32, data_type: .float)
        var params2 = BNNSConvolutionLayerParameters(x_stride: 1, y_stride: 1, x_padding: 4, y_padding: 4, k_width: 5, k_height: 5, in_channels: 64, out_channels: 64, weights: h2w, bias: h2b, activation: tanh)
        let h2 = BNNSFilterCreateConvolutionLayer(&output1, &output2, &params2, &filterParams)

        let outH0 = UnsafeMutableRawPointer.allocate(bytes: 32 * 32 * 64 * floatSize, alignedTo: floatSize)
        let outH1 = UnsafeMutableRawPointer.allocate(bytes: 32 * 32 * 64 * floatSize, alignedTo: floatSize)
        let outH2 = UnsafeMutableRawPointer.allocate(bytes: 32 * 32 * 48 * floatSize, alignedTo: floatSize)

        defer {
            outH0.deallocate(bytes: 32 * 32 * 64 * floatSize, alignedTo: floatSize)
            outH1.deallocate(bytes: 32 * 32 * 64 * floatSize, alignedTo: floatSize)
            outH2.deallocate(bytes: 32 * 32 * 48 * floatSize, alignedTo: floatSize)
        }

        assert(BNNSFilterApply(h0, input.data!, outH0) == 0)
        assert(BNNSFilterApply(h1, outH0, outH1) == 0)
        assert(BNNSFilterApply(h2, outH1, outH2) == 0)

        // Phase Shift operation + Repack into RGBA.
        let values = UnsafeBufferPointer(start: outH2.assumingMemoryBound(to: Float.self), count: 32 * 32 * 48)
        var rgba = Array<Float>(repeating: 1.0, count: 128 * 128 * 4)

        let r = 4
        for x in 0 ..< 128 {
            for y in 0 ..< 128 {
                for c in 0 ..< 3 {
                    let cccc = c + 1
                    let a = x / r
                    let b = y / r
                    let d = cccc * r * (y % r) + cccc * (x % r)
                    rgba[y * 128 * 4 + x * 4 + c] = values[d * 32 * 32 + a * 32 + b]
                }
            }
        }

        let image = rgba.withUnsafeBytes { bytes -> CIImage in
            let data = Data(bytes: bytes.baseAddress!, count: bytes.count)
            return CIImage(bitmapData: data,
                           bytesPerRow: 128 * 16,
                           size: CGSize(width: 128, height: 128),
                           format: kCIFormatRGBAf,
                           colorSpace: CGColorSpaceCreateDeviceRGB())
        }

        return image
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

    for i in 0 ..< transformed.width {
        print("\(floats[i * 4 + channel])", terminator: " ")
    }
    print("")

    input.deallocate(bytes: transformed.width * transformed.height * 2 * transformed.featureChannels, alignedTo: 2)
}
