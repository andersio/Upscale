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

    let c: MetalCoordinator
    let original: UIImage
    let events: Signal<Event, NoError>

    let image: Property<UIImage?>

    private let eventsObserver: Signal<Event, NoError>.Observer

    init(original: UIImage, coordinator: MetalCoordinator) {
        c = coordinator

        assert(original.size == CGSize(width: 32, height: 32))
        self.original = original

        (events, eventsObserver) = Signal.pipe()

        let _image = MutableProperty<UIImage?>(nil)
        self.image = Property(_image)

        let downsampled = original.resized(to: CGSize(width: 32, height: 32))!

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: 32, height: 32, mipmapped: false)
        let texture = coordinator.device.makeTexture(descriptor: descriptor)!

        let data = try! Data(contentsOf: Bundle.main.url(forResource: "in_test", withExtension: nil)!)
        var rgba: [Float] = Array(repeating: 0.0, count: 32 * 32 * 4)
        data.withUnsafeBytes { (floats: UnsafePointer<Float>) in
            for i in 0 ..< 32*32 {
                rgba[i * 4] = floats[i * 3]
                rgba[i * 4 + 1] = floats[i * 3 + 1]
                rgba[i * 4 + 2] = floats[i * 3 + 2]
                rgba[i * 4 + 3] = 255.0
            }
        }

        texture.replace(region: MTLRegionMake2D(0, 0, 32, 32), mipmapLevel: 0, withBytes: &rgba, bytesPerRow: 512)

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
        self.init(cgImage: CGImage.make(texture: texture), scale: 0.0, orientation: .up)
    }
}
