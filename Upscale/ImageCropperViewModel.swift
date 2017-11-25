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

    func commit(_ rect: CGRect, _ angle: Int) {
        guard let finalImage = image.resized(to: CGSize(width: 128, height: 128), cropping: rect)
            else { return }
        eventsObserver.send(value: .userDidFinishCropping(finalImage))
    }
}
