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
    let inputBatchSize: Int
    let subpixelScale: Int

    let hiddenLayer0: TransposedConvolutionLayer
    //let activator0: Activator
    let hiddenLayer1: TransposedConvolutionLayer
    //let activator1: Activator
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
        self.inputBatchSize = inputBatchSize
        self.subpixelScale = subpixelScale

        // Hidden Layer 0
        hiddenLayer0 = TransposedConvolutionLayer(device: device,
                                                  kernelSize: (1, 1),
                                                  inputBatchSize: inputBatchSize,
                                                  inputFeatureChannels: colorChannels,
                                                  outputFeatureChannels: featureChannels)
        //activator0 = try Activator(device: device, library: library, kind: .leakyRELU(0.2))

        // Hidden Layer 1
        hiddenLayer1 = TransposedConvolutionLayer(device: device,
                                                  kernelSize: (5, 5),
                                                  inputBatchSize: inputBatchSize,
                                                  inputFeatureChannels: featureChannels,
                                                  outputFeatureChannels: featureChannels)
        //activator1 = try Activator(device: device, library: library, kind: .leakyRELU(0.2))

        // Subpixel Convolution Layer
        subpixelLayer = SubpixelConvolutionLayer(device: device,
                                                 kernelSize: (5, 5),
                                                 inputBatchSize: inputBatchSize,
                                                 inputFeatureChannels: featureChannels,
                                                 subpixelScale: subpixelScale,
                                                 outputColorChannels: colorChannels)

        // Load weights and bias
        if let weights = weights {
            try load(weights)
        }
    }

    private func load(_ weights: WeightsDescriptor) throws {
        hiddenLayer0.initialize(bias: try Data(contentsOf: weights.hiddenLayer0Bias, options: []),
                                weights: try Data(contentsOf: weights.hiddenLayer0Weight, options: []))
        hiddenLayer1.initialize(bias: try Data(contentsOf: weights.hiddenLayer1Bias, options: []),
                                weights: try Data(contentsOf: weights.hiddenLayer1Weight, options: []))
        subpixelLayer.initialize(bias: try Data(contentsOf: weights.subpixelLayerBias, options: []),
                                 weights: try Data(contentsOf: weights.subpixelLayerWeight, options: []))
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
