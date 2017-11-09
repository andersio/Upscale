import MetalPerformanceShaders
import Metal

final class SubpixelConvolutionLayer: NSObject, MPSCNNConvolutionDataSource {
    private var kernel: MPSCNNConvolutionTranspose!
    private let _descriptor: MPSCNNConvolutionDescriptor

    private let numberOfWeights: Int
    private let outputFeatureChannels: Int
    private let weightsUrl: URL
    private let biasUrl: URL
    private var weightsBuffer: UnsafeMutablePointer<Float>?
    private var biasBuffer: UnsafeMutablePointer<Float>?

    init(device: MTLDevice, kernelSize: (width: Int, height: Int), inputBatchSize: Int, inputFeatureChannels: Int, subpixelScale: Int, outputColorChannels: Int, weights: URL, bias: URL) {
        // Support only 1 for now.
        assert(inputBatchSize == 1)

        outputFeatureChannels = outputColorChannels * subpixelScale * subpixelScale

        numberOfWeights = kernelSize.width * kernelSize.height * outputFeatureChannels * inputFeatureChannels

        self.weightsUrl = weights
        self.biasUrl = bias

        _descriptor = MPSCNNConvolutionDescriptor(kernelWidth: kernelSize.width,
                                                  kernelHeight: kernelSize.height,
                                                  inputFeatureChannels: inputFeatureChannels,
                                                  outputFeatureChannels: outputColorChannels * subpixelScale * subpixelScale)
        _descriptor.groups = inputBatchSize
        //_descriptor.subPixelScaleFactor = subpixelScale
        _descriptor.setNeuronType(.none, parameterA: 1.0, parameterB: 1.0)

        super.init()
        kernel = MPSCNNConvolutionTranspose(device: device, weights: self)
        kernel.destinationImageAllocator = MPSImage.defaultAllocator()
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
        return nil
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
            weightsBuffer?.assign(from: weights, count: numberOfWeights)
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


