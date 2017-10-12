import MetalPerformanceShaders
import Metal

final class SubpixelConvolutionLayer: NSObject, MPSCNNConvolutionDataSource {
    private var kernel: MPSCNNConvolution!
    private let _descriptor: MPSCNNSubPixelConvolutionDescriptor

    private let numberOfWeights: Int
    private let weightsBuffer: UnsafeMutablePointer<Float>
    private let biasBuffer: UnsafeMutablePointer<Float>

    init(device: MTLDevice, kernelSize: (width: Int, height: Int), inputBatchSize: Int, inputFeatureChannels: Int, subpixelScale: Int, outputColorChannels: Int) {
        // Support only 1 for now.
        assert(inputBatchSize == 1)

        let outputFeatureChannels = outputColorChannels * subpixelScale * subpixelScale

        numberOfWeights = kernelSize.width * kernelSize.height * outputFeatureChannels * inputFeatureChannels
        weightsBuffer = UnsafeMutablePointer.allocate(capacity: numberOfWeights)
        for offset in 0 ..< numberOfWeights {
            weightsBuffer.advanced(by: offset).initialize(to: .randomNormal)
        }

        biasBuffer = UnsafeMutablePointer<Float>.allocate(capacity: outputFeatureChannels)
        biasBuffer.initialize(to: 0.0, count: outputFeatureChannels)

        _descriptor = MPSCNNSubPixelConvolutionDescriptor()
        _descriptor.kernelWidth = kernelSize.width
        _descriptor.kernelHeight = kernelSize.height
        _descriptor.groups = inputBatchSize
        _descriptor.inputFeatureChannels = inputFeatureChannels
        _descriptor.outputFeatureChannels = outputColorChannels * subpixelScale * subpixelScale
        _descriptor.subPixelScaleFactor = subpixelScale
        _descriptor.setNeuronType(.tanH, parameterA: 1.0, parameterB: 1.0)

        super.init()
        kernel = MPSCNNConvolution(device: device, weights: self)
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


