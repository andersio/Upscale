import UIKit
import ReactiveCocoa
import ReactiveSwift
import Result
import MetalKit

class RootViewController: UIViewController {
    private let viewModel: RootViewModel
    fileprivate let renderer: Renderer

    private let stackView: UIStackView
    private let flipButton: UIBarButtonItem
    private let modeButton: UIBarButtonItem
    private let inferenceButton: UIBarButtonItem
    private let pinchGesture: UIGestureRecognizer

    // MTKView
    private let afterView: MTKView

    override var prefersStatusBarHidden: Bool {
        return true
    }

    init(viewModel: RootViewModel, renderer: Renderer) {
        self.viewModel = viewModel
        self.renderer = renderer

        stackView = UIStackView()
        flipButton = UIBarButtonItem()
        modeButton = UIBarButtonItem()
        inferenceButton = UIBarButtonItem()
        pinchGesture = UIPinchGestureRecognizer()

        afterView = MTKView(frame: .zero, device: renderer.device)

        super.init(nibName: nil, bundle: nil)

        afterView.delegate = self
    }

    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func loadView() {
        view = stackView
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        edgesForExtendedLayout = []

        view.addGestureRecognizer(pinchGesture)
        viewModel.zoom <~ pinchGesture.reactive.stateChanged
            .filterMap { recognizer -> RootViewModel.ZoomAction? in
                let recognizer = unsafeDowncast(recognizer, to: UIPinchGestureRecognizer.self)
                switch recognizer.state {
                case .began:
                    return .began(Double(recognizer.scale))
                case .ended:
                    return .ended(Double(recognizer.scale))
                case .changed:
                    return .changed(Double(recognizer.scale))
                case .cancelled, .failed, .possible:
                    return nil
                }
            }

        flipButton.title = "Flip"
        flipButton.reactive.pressed = CocoaAction(viewModel.flipAxis)

        modeButton.title = "Mode"
        modeButton.reactive.pressed = CocoaAction(viewModel.switchMode)

        inferenceButton.title = "Infer"
        inferenceButton.reactive.pressed = CocoaAction(viewModel.infer)

        stackView.reactive[\.axis] <~ viewModel.axis.map { $0 == .horizontal ? .horizontal : .vertical }

        stackView.distribution = .fillEqually
        stackView.spacing = 1.0

        afterView.isHidden = true
        afterView.backgroundColor = .red
        stackView.addArrangedSubview(afterView)

        let overlayMargins = UIEdgeInsets(top: 8, left: 8, bottom: 8, right: 8)
        afterView.layoutMargins = overlayMargins

        viewModel.mode.producer.startWithValues { [weak self] mode in
            self?.updateStackView(for: mode)
            self?.updateToolbar(for: mode)
        }
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)

        navigationController?.setNavigationBarHidden(true, animated: animated)
        navigationController?.setToolbarHidden(false, animated: animated)
    }

    private func updateStackView(for mode: RootViewModel.Mode) {
        self.stackView.insertArrangedSubview(afterView, at: 0)
        UIView.animate(
            withDuration: 0.40,
            delay: 0.0,
            usingSpringWithDamping: 0.6,
            initialSpringVelocity: 0.2,
            options: [],
            animations: {
                self.afterView.isHidden = false
            },
            completion: nil
        )
    }

    private func updateToolbar(for mode: RootViewModel.Mode) {
        let common = [UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil),
                      inferenceButton,
                      UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil)]
        setToolbarItems(common, animated: true)
    }
}

extension RootViewController: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {

    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let descriptor = view.currentRenderPassDescriptor
            else { return }
        renderer.draw(to: drawable, using: descriptor)
    }
}
