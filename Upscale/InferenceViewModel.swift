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
    let billinear: UIImage
    let downsampled: UIImage

    private let eventsObserver: Signal<Event, NoError>.Observer

    init(original: UIImage, coordinator: MetalCoordinator) {
        c = coordinator

        //assert(original.size == CGSize(width: 32, height: 32))
        self.original = original

        (events, eventsObserver) = Signal.pipe()

        let _image = MutableProperty<UIImage?>(nil)
        self.image = Property(capturing: _image)

        downsampled = original.resized(to: CGSize(width: 32, height: 32))!
        billinear = downsampled.resized(to: CGSize(width: 128, height: 128))!

        let loader = MTKTextureLoader(device: coordinator.device)
        let texture = try! loader.newTexture(cgImage: downsampled.cgImage!, options: nil)
        let image = MPSImage(texture: texture, featureChannels: 3)

        coordinator.superResolution.infer(image) { output, time in
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
