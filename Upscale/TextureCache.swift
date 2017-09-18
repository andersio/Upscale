import AVFoundation
import CoreMedia
import Metal
import MetalKit
import MetalPerformanceShaders
import ReactiveSwift
import Result

final class MetalCoordinator {
    let device: MTLDevice
    let textures: Signal<(original: MTLTexture, transformed: MTLTexture), NoError>
    private let texturesObserver: Signal<(original: MTLTexture, transformed: MTLTexture), NoError>.Observer

    private let textureCache: CVMetalTextureCache

    init() {
        device = MTLCreateSystemDefaultDevice()!

        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        textureCache = cache!

        (textures, texturesObserver) = Signal.pipe()
    }
}
