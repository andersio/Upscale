import ReactiveSwift
import Result
import UIKit
import Metal
import MetalKit
import MetalPerformanceShaders

final class InferenceViewModel {
    enum Event {
        case close
    }

    let original: UIImage
    let events: Signal<Event, NoError>

    let image: Property<UIImage?>

    private let eventsObserver: Signal<Event, NoError>.Observer

    init(original: UIImage, coordinator: MetalCoordinator) {
        assert(original.size == CGSize(width: 128, height: 128))
        self.original = original

        (events, eventsObserver) = Signal.pipe()

        let _image = MutableProperty<UIImage?>(nil)
        self.image = Property(_image)

        let downsampled = original.resized(to: CGSize(width: 32, height: 32))!
        let loader = MTKTextureLoader(device: coordinator.device)
        let texture = try! loader.newTexture(cgImage: downsampled.cgImage!, options: [:])
        let image = MPSImage(texture: texture, featureChannels: 3)

        let start = DispatchTime.now()

        coordinator.superResolution.infer(image) { output in
            let end = DispatchTime.now()

            print("\(end.uptimeNanoseconds - start.uptimeNanoseconds) ns")
            print(output.pixelFormat.rawValue)
            print(output.pixelSize)
            print(output.width)
            print(output.height)

            _image.value = UIImage(texture: output.texture)
        }
    }

    @objc func close() {
        eventsObserver.send(value: .close)
    }
}

extension UIImage {
    convenience init(texture: MTLTexture) {
        assert(texture.pixelFormat == .rgba8Unorm, "Pixel format of texture must be MTLPixelFormatRGBA8Unorm to create UIImage")
        
        let imageByteCount = texture.width * texture.height * 4
        let imageBytes = malloc(imageByteCount)!
        let bytesPerRow = texture.width * 4
        let region = MTLRegionMake2D(0, 0, texture.width, texture.height)

        texture.getBytes(imageBytes, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)

        for pixel in 0 ..< texture.width * texture.height {
            imageBytes.advanced(by: pixel * 4 + 3)
                .assumingMemoryBound(to: UInt8.self)
                .pointee = .max
        }

        let provider = CGDataProvider(dataInfo: nil,
                                      data: imageBytes,
                                      size: imageByteCount,
                                      releaseData: { _, data, _ in free(UnsafeMutableRawPointer(mutating: data)) })!
        let bitsPerComponent = 8
        let bitsPerPixel = 32
        let colorSpaceRef = CGColorSpaceCreateDeviceRGB();
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.first.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        let image = CGImage(width: texture.width,
                            height: texture.height,
                            bitsPerComponent: bitsPerComponent,
                            bitsPerPixel: bitsPerPixel,
                            bytesPerRow: bytesPerRow,
                            space: colorSpaceRef,
                            bitmapInfo: bitmapInfo,
                            provider: provider,
                            decode: nil,
                            shouldInterpolate: false,
                            intent: .defaultIntent)!

        self.init(cgImage: image, scale: 0.0, orientation: .downMirrored)
    }
}
