import MetalPerformanceShaders
import Metal

final class TransposedConvolutionLayer: NSObject, MPSCNNConvolutionDataSource {
    private var kernel: MPSCNNConvolutionTranspose!
    private let _descriptor: MPSCNNConvolutionDescriptor

    private let numberOfWeights: Int
    private let weightsBuffer: UnsafeMutablePointer<Float>
    private let biasBuffer: UnsafeMutablePointer<Float>

    init(device: MTLDevice, kernelSize: (width: Int, height: Int), inputBatchSize: Int, inputFeatureChannels: Int, outputFeatureChannels: Int) {
        numberOfWeights = kernelSize.width * kernelSize.height * outputFeatureChannels * inputFeatureChannels
        weightsBuffer = UnsafeMutablePointer.allocate(capacity: numberOfWeights)
        for offset in 0 ..< numberOfWeights {
            weightsBuffer.advanced(by: offset).initialize(to: .randomNormal)
        }

        biasBuffer = UnsafeMutablePointer<Float>.allocate(capacity: outputFeatureChannels)
        biasBuffer.initialize(to: 0.0, count: outputFeatureChannels)

        _descriptor = MPSCNNConvolutionDescriptor()
        _descriptor.kernelWidth = kernelSize.width
        _descriptor.kernelHeight = kernelSize.height
        _descriptor.inputFeatureChannels = inputFeatureChannels
        _descriptor.outputFeatureChannels = outputFeatureChannels
        _descriptor.groups = inputBatchSize
        _descriptor.setNeuronType(.reLU, parameterA: 0.2, parameterB: 1.0)

        super.init()
        kernel = MPSCNNConvolutionTranspose(device: device, weights: self)
    }

    deinit {
        weightsBuffer.deinitialize()
        weightsBuffer.deallocate(capacity: numberOfWeights)
        biasBuffer.deinitialize()
        biasBuffer.deallocate(capacity: kernel.outputFeatureChannels)
    }

    func initialize(bias: Data, weights: Data) {
        assert(bias.count == kernel.outputFeatureChannels * MemoryLayout<Float>.size
            && weights.count == numberOfWeights * MemoryLayout<Float>.size)
        _ = bias.copyBytes(to: UnsafeMutableBufferPointer(start: biasBuffer, count: kernel.outputFeatureChannels))
        _ = weights.copyBytes(to: UnsafeMutableBufferPointer(start: weightsBuffer, count: numberOfWeights))
    }

    func encode(to commandBuffer: MTLCommandBuffer, source: MPSImage) -> MPSImage {
        return kernel.encode(commandBuffer: commandBuffer, sourceImage: source)
    }

    func biasTerms() -> UnsafeMutablePointer<Float>? {
        return biasBuffer
    }

    func dataType() -> MPSDataType {
        return .float32
    }

    func descriptor() -> MPSCNNConvolutionDescriptor {
        return _descriptor
    }

    func label() -> String? {
        return nil
    }

    func load() -> Bool {
        return true
    }

    func purge() {}

    func weights() -> UnsafeMutableRawPointer {
        return UnsafeMutableRawPointer(weightsBuffer)
    }
}
