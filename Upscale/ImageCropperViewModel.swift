import UIKit
import Result
import ReactiveSwift

class ImageCropperViewModel {
    enum Event {
        case userDidFinishCropping(UIImage)
    }

    let image: UIImage
    let events: Signal<Event, NoError>
    private let eventsObserver: Signal<Event, NoError>.Observer

    init(for image: UIImage) {
        self.image = image
        (events, eventsObserver) = Signal.pipe()
    }

    func commit(_ rect: CGRect) {
        let targetSize = CGSize(width: 128, height: 128)
        UIGraphicsBeginImageContextWithOptions(targetSize, true, image.scale)
        UIRectClip(CGRect(x: -rect.minX, y: -rect.minY, width: rect.width, height: rect.height))
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        defer { UIGraphicsEndImageContext() }
        guard let finalImage = UIGraphicsGetImageFromCurrentImageContext()
            else { return }
        eventsObserver.send(value: .userDidFinishCropping(finalImage))
    }
}

