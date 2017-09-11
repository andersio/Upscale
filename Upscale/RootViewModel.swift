import ReactiveSwift
import Result

final class RootViewModel {
    private let (lifetime, token) = Lifetime.make()

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

    init() {
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
    }
}
