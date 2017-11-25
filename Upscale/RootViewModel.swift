import ReactiveSwift
import Result
import CoreVideo
import UIKit

final class RootViewModel {
    private let (lifetime, token) = Lifetime.make()

    enum Event {
        case showCropper(UIImage)
    }

    enum Axis {
        case horizontal
        case vertical
    }

    enum ZoomAction {
        case began(Double)
        case changed(Double)
        case ended(Double)
    }

    enum Mode {
        case showBoth
        case showBefore
        case showAfter
    }

    let mode: Property<Mode>
    let axis: Property<Axis>
    let flipAxis: Action<(), Never, NoError>
    let switchMode: Action<(), Never, NoError>
    let zoom: BindingTarget<ZoomAction>

    let events: Signal<Event, NoError>
    let infer: Action<(), Never, NoError>

    private let cameraSource: CameraSource

    init(cameraSource: CameraSource) {
        self.cameraSource = cameraSource

        let mode = MutableProperty(Mode.showBoth)
        self.mode = Property(capturing: mode)

        self.switchMode = Action {
            return SignalProducer.empty.on(completed: {
                mode.modify { mode in
                    switch mode {
                    case .showBoth:
                        mode = .showBefore
                    case .showBefore:
                        mode = .showAfter
                    case .showAfter:
                        mode = .showBoth
                    }
                }
            })
        }

        let axis = MutableProperty(Axis.vertical)
        self.flipAxis = Action {
            return SignalProducer.empty.on(completed: {
                axis.modify { $0 = $0 == .horizontal ? .vertical : .horizontal }
            })
        }
        self.axis = Property(capturing: axis)

        var initial: Double?
        self.zoom = BindingTarget(lifetime: lifetime) { action in
            switch action {
            case let .began(scale):
                initial = scale
            case let .changed(scale), let .ended(scale):
                if let initial = initial {
                    let factor = scale / initial
                    print(factor)
                }
            }
        }

        self.infer = Action { _ in .empty }
        events = infer.completed
            .withLatest(from: cameraSource.samples)
            .observe(on: QueueScheduler())
            .map { buffer in
                let intermediate = CIImage(cvImageBuffer: buffer.1)
                    .oriented(.right)
                let image = CIContext().createCGImage(intermediate, from: intermediate.extent)!
                return UIImage(cgImage: image,
                               scale: UIScreen.main.scale,
                               orientation: .up)
            }
            .map(Event.showCropper)
    }
}
