import ReactiveSwift
import Result
import UIKit

final class InferenceViewModel {
    enum Event {
        case close
    }

    let original: UIImage
    let events: Signal<Event, NoError>

    let isExecuting: Property<Bool>

    private let eventsObserver: Signal<Event, NoError>.Observer

    init(original: UIImage, coordinator: MetalCoordinator) {
        assert(original.size == CGSize(width: 128, height: 128))
        self.original = original

        (events, eventsObserver) = Signal.pipe()

        let _isExecuting = MutableProperty(true)
        isExecuting = Property(_isExecuting)
    }

    @objc func close() {
        eventsObserver.send(value: .close)
    }
}
