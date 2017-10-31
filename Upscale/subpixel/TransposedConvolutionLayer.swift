import MetalPerformanceShaders
import Metal

final class TransposedConvolutionLayer: NSObject, MPSCNNConvolutionDataSource {
    private var kernel: MPSCNNConvolutionTranspose!
    private let _descriptor: MPSCNNConvolutionDescriptor

    private let _label: String
    private let numberOfWeights: Int
    private let outputFeatureChannels: Int

    private let weightsUrl: URL
    private let biasUrl: URL
    private var weightsBuffer: UnsafeMutablePointer<Float>?
    private var biasBuffer: UnsafeMutablePointer<Float>?

    init(label: String, device: MTLDevice, kernelSize: (width: Int, height: Int), inputBatchSize: Int, inputFeatureChannels: Int, outputFeatureChannels: Int, weights: URL, bias: URL) {
        numberOfWeights = kernelSize.width * kernelSize.height * outputFeatureChannels * inputFeatureChannels

        self._label = label
        self.weightsUrl = weights
        self.biasUrl = bias
        self.outputFeatureChannels = outputFeatureChannels

        _descriptor = MPSCNNConvolutionDescriptor(kernelWidth: kernelSize.width,
                                                  kernelHeight: kernelSize.height,
                                                  inputFeatureChannels: inputFeatureChannels,
                                                  outputFeatureChannels: outputFeatureChannels)
        _descriptor.groups = inputBatchSize
        _descriptor.setNeuronType(.reLU, parameterA: 0.2, parameterB: 1.0)

        var means: [Float] = []
        var variances: [Float] = []
        var gammas: [Float] = []
        var betas: [Float] = []
        var epsilons: [Float] = []

        super.init()
        kernel = MPSCNNConvolutionTranspose(device: device, weights: self)
        kernel.destinationImageAllocator = MPSImage.defaultAllocator()
    }

    deinit {
        purge()
    }

    func encode(to commandBuffer: MTLCommandBuffer, source: MPSImage) -> MPSImage {
        return kernel.encode(commandBuffer: commandBuffer, sourceImage: source)
    }

    func biasTerms() -> UnsafeMutablePointer<Float>? {
        return biasBuffer!
    }

    func dataType() -> MPSDataType {
        return .float32
    }

    func descriptor() -> MPSCNNConvolutionDescriptor {
        return _descriptor
    }

    func label() -> String? {
        return _label
    }

    func load() -> Bool {
        weightsBuffer = UnsafeMutablePointer.allocate(capacity: numberOfWeights)
        biasBuffer = UnsafeMutablePointer<Float>.allocate(capacity: outputFeatureChannels)
        biasBuffer!.initialize(to: 0.0, count: outputFeatureChannels)

        let bias = try! Data(contentsOf: biasUrl, options: [])
        let weights = try! Data(contentsOf: weightsUrl, options: [])

        assert(bias.count == outputFeatureChannels * MemoryLayout<Float>.size
            && weights.count == numberOfWeights * MemoryLayout<Float>.size)
        let copiedBiasBytes = bias.copyBytes(to: UnsafeMutableBufferPointer(start: biasBuffer!, count: outputFeatureChannels))
        assert(copiedBiasBytes == outputFeatureChannels * MemoryLayout<Float>.size)

        weights.withUnsafeBytes { (weights: UnsafePointer<Float>) in
            let numberOfWeightsPerOutput = numberOfWeights / outputFeatureChannels

            for outputIndex in 0 ..< outputFeatureChannels {
                var sum = Float(0.0)

                for weightIndex in 0 ..< numberOfWeightsPerOutput {
                    let index = outputIndex * numberOfWeightsPerOutput + weightIndex
                    sum += weights[index]
                }

                for weightIndex in 0 ..< numberOfWeightsPerOutput {
                    let index = outputIndex * numberOfWeightsPerOutput + weightIndex
                    weightsBuffer![index] = weights[index] / sum
                }
            }
        }

        return true
    }

    func purge() {
        weightsBuffer?.deinitialize()
        weightsBuffer?.deallocate(capacity: numberOfWeights)
        weightsBuffer = nil
        biasBuffer?.deinitialize()
        biasBuffer?.deallocate(capacity: outputFeatureChannels)
        biasBuffer = nil
    }

    func weights() -> UnsafeMutableRawPointer {
        return UnsafeMutableRawPointer(weightsBuffer)!
    }
}
