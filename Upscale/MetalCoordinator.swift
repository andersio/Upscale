import AVFoundation
import CoreMedia
import Metal
import MetalPerformanceShaders
import ReactiveSwift
import Result

protocol Renderer: class {
    var device: MTLDevice { get }

    func draw(to drawable: MTLDrawable, using descriptor: MTLRenderPassDescriptor)
}

final class MetalCoordinator: Renderer {
    let device: MTLDevice

    let superResolution: SuperResolution

    private let library: MTLLibrary
    private let pipelineState: MTLRenderPipelineState
    private let textureCache: CVMetalTextureCache
    private var texture: MTLTexture?

    init() {
        device = MTLCreateSystemDefaultDevice()!

        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        textureCache = cache!

        library = device.makeDefaultLibrary()!

        let pipeline = MTLRenderPipelineDescriptor()
        pipeline.sampleCount = 1
        pipeline.colorAttachments[0].pixelFormat = .bgra8Unorm
        pipeline.depthAttachmentPixelFormat = .invalid
        pipeline.vertexFunction = library.makeFunction(name: "test_vertex")
        pipeline.fragmentFunction = library.makeFunction(name: "test_fragment")
        pipelineState = try! device.makeRenderPipelineState(descriptor: pipeline)

        let bundle = Bundle(for: MetalCoordinator.self)
        let descriptor = SuperResolution.WeightsDescriptor(
            hiddenLayer0Weight: bundle.url(forResource: "w_conv1", withExtension: nil)!,
            hiddenLayer1Weight: bundle.url(forResource: "w_conv2", withExtension: nil)!,
            hiddenLayer2Weight: bundle.url(forResource: "w_conv3", withExtension: nil)!,
            hiddenLayer0Bias: bundle.url(forResource: "b_conv1", withExtension: nil)!,
            hiddenLayer1Bias: bundle.url(forResource: "b_conv2", withExtension: nil)!,
            hiddenLayer2Bias: bundle.url(forResource: "b_conv3", withExtension: nil)!
        )

        superResolution = try! SuperResolution(device: device,
                                               inputBatchSize: 1,
                                               subpixelScale: 4,
                                               colorChannels: 3,
                                               featureChannels: 64,
                                               weights: descriptor)
    }

    func draw(to drawable: MTLDrawable, using descriptor: MTLRenderPassDescriptor) {
        guard let texture = self.texture else { return }
        draw(texture, to: drawable, using: descriptor)
    }


    func draw(_ texture: MTLTexture, to drawable: MTLDrawable, using descriptor: MTLRenderPassDescriptor) {
        let commandBuffer = device.makeCommandQueue()!.makeCommandBuffer()!
        let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor)!
        encoder.pushDebugGroup("RenderFrame")
        encoder.setRenderPipelineState(pipelineState)
        encoder.setFragmentTexture(texture, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: 1)
        encoder.popDebugGroup()
        encoder.endEncoding()

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    func handle(_ sample: CVImageBuffer) {
        let rect = CVImageBufferGetCleanRect(sample)
        var texture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, sample, nil, .bgra8Unorm, Int(rect.width), Int(rect.height), 0, &texture)

        if let texture = texture.flatMap(CVMetalTextureGetTexture) {
            self.texture = texture
        }
    }
}
