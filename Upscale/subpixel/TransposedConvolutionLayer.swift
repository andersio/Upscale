import MetalPerformanceShaders
import Metal

final class TransposedConvolutionLayer: NSObject, MPSCNNConvolutionDataSource {
    enum Activation {
        case relu
        case tanh
    }

    private var kernel: MPSCNNConvolutionTranspose!
    private let _descriptor: MPSCNNConvolutionDescriptor

    private let _label: String
    private let numberOfWeights: Int
    public let inputFeatureChannels: Int
    private let outputFeatureChannels: Int
    private let kernelHeight: Int
    private let kernelWidth: Int

    private let weightsUrl: URL
    private let biasUrl: URL
    private var weightsBuffer: UnsafeMutablePointer<Float>?
    private var biasBuffer: UnsafeMutablePointer<Float>?

    init(label: String, device: MTLDevice, kernelSize: (width: Int, height: Int), inputBatchSize: Int, inputFeatureChannels: Int, outputFeatureChannels: Int, weights: URL, bias: URL, activation: Activation, isIntermediate: Bool = true) {
        numberOfWeights = kernelSize.width * kernelSize.height * outputFeatureChannels * inputFeatureChannels
        kernelHeight = kernelSize.height
        kernelWidth = kernelSize.width
        self.inputFeatureChannels = inputFeatureChannels

        self._label = label
        self.weightsUrl = weights
        self.biasUrl = bias
        self.outputFeatureChannels = outputFeatureChannels

        _descriptor = MPSCNNConvolutionDescriptor(kernelWidth: kernelSize.width,
                                                  kernelHeight: kernelSize.height,
                                                  inputFeatureChannels: inputFeatureChannels,
                                                  outputFeatureChannels: outputFeatureChannels)
        _descriptor.groups = inputBatchSize
        switch activation {
        case .relu:
            _descriptor.setNeuronType(.reLU, parameterA: 0.2, parameterB: 1.0)
        case .tanh:
            _descriptor.setNeuronType(.tanH, parameterA: 1.0, parameterB: 1.0)
        }

        super.init()
        kernel = MPSCNNConvolutionTranspose(device: device, weights: self)

        if !isIntermediate {
            kernel.destinationImageAllocator = MPSImage.defaultAllocator()
        }
    }

    deinit {
        purge()
    }

    func encode(to commandBuffer: MTLCommandBuffer, source: MPSImage) -> MPSImage {
        return kernel.encode(commandBuffer: commandBuffer, sourceImage: source, convolutionState: nil)
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
        let copiedWeightBytes = weights.copyBytes(to: UnsafeMutableBufferPointer(start: weightsBuffer!, count: numberOfWeights))
        assert(copiedWeightBytes == numberOfWeights * MemoryLayout<Float>.size)

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
