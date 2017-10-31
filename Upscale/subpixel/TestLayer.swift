import MetalPerformanceShaders
import Metal

final class TestLayer: NSObject, MPSCNNConvolutionDataSource {
    private var kernel: MPSCNNConvolutionTranspose!
    private let _descriptor: MPSCNNConvolutionDescriptor
    
    private let _label: String
    private let numberOfWeights: Int
    private let outputFeatureChannels: Int
    
    private var weightsBuffer: UnsafeMutablePointer<Float>?
    
    init(label: String, device: MTLDevice, inputBatchSize: Int, inputFeatureChannels: Int, outputFeatureChannels: Int) {
        numberOfWeights = outputFeatureChannels * inputFeatureChannels
        
        self._label = label
        self.outputFeatureChannels = outputFeatureChannels
        
        _descriptor = MPSCNNConvolutionDescriptor(kernelWidth: 1,
                                                  kernelHeight: 1,
                                                  inputFeatureChannels: inputFeatureChannels,
                                                  outputFeatureChannels: outputFeatureChannels)
        _descriptor.groups = inputBatchSize
        _descriptor.setNeuronType(.none, parameterA: 0.2, parameterB: 1.0)
        
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
        return nil
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
        weightsBuffer!.initialize(to: 1.0, count: numberOfWeights)
        
        return true
    }
    
    func purge() {
        weightsBuffer?.deinitialize()
        weightsBuffer?.deallocate(capacity: numberOfWeights)
        weightsBuffer = nil
    }
    
    func weights() -> UnsafeMutableRawPointer {
        return UnsafeMutableRawPointer(weightsBuffer)!
    }
}

