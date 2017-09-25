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
    private let pinchGesture: UIGestureRecognizer

    // MTKView
    fileprivate let beforeView: MTKView
    private let beforeLabel: ViewportLabel

    // MTKView
    private let afterView: MTKView
    private let afterLabel: ViewportLabel

    override var prefersStatusBarHidden: Bool {
        return true
    }

    init(viewModel: RootViewModel, renderer: Renderer) {
        self.viewModel = viewModel
        self.renderer = renderer

        stackView = UIStackView()
        flipButton = UIBarButtonItem()
        modeButton = UIBarButtonItem()
        pinchGesture = UIPinchGestureRecognizer()

        beforeView = MTKView(frame: .zero, device: renderer.device)
        beforeLabel = ViewportLabel()
        afterView = MTKView(frame: .zero, device: renderer.device)
        afterLabel = ViewportLabel()

        super.init(nibName: nil, bundle: nil)

        beforeView.delegate = self
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

        stackView.reactive[\.axis] <~ viewModel.axis.map { $0 == .horizontal ? .horizontal : .vertical }

        stackView.distribution = .fillEqually
        stackView.spacing = 1.0

        beforeView.isHidden = true
        afterView.isHidden = true
        beforeView.backgroundColor = .green
        afterView.backgroundColor = .red

        beforeLabel.translatesAutoresizingMaskIntoConstraints = false
        afterLabel.translatesAutoresizingMaskIntoConstraints = false
        beforeView.addSubview(beforeLabel)
        afterView.addSubview(afterLabel)
        stackView.addArrangedSubview(beforeView)
        stackView.addArrangedSubview(afterView)

        let overlayMargins = UIEdgeInsets(top: 8, left: 8, bottom: 8, right: 8)
        beforeView.layoutMargins = overlayMargins
        afterView.layoutMargins = overlayMargins
        beforeLabel.label.text = "Before"
        afterLabel.label.text = "After"

        NSLayoutConstraint.activate([
            beforeLabel.leftAnchor.constraint(equalTo: beforeView.layoutMarginsGuide.leftAnchor),
            beforeLabel.bottomAnchor.constraint(equalTo: beforeView.layoutMarginsGuide.bottomAnchor),
            afterLabel.leftAnchor.constraint(equalTo: afterView.layoutMarginsGuide.leftAnchor),
            afterLabel.bottomAnchor.constraint(equalTo: afterView.layoutMarginsGuide.bottomAnchor)
        ])

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
        switch mode {
        case .showBoth:
            self.stackView.insertArrangedSubview(self.beforeView, at: 0)
            self.stackView.insertArrangedSubview(self.afterView, at: 1)
            UIView.animate(
                withDuration: 0.40,
                delay: 0.0,
                usingSpringWithDamping: 0.6,
                initialSpringVelocity: 0.2,
                options: [],
                animations: {
                    self.beforeView.isHidden = false
                    self.afterView.isHidden = false
                    self.beforeLabel.alpha = 1.0
                    self.afterLabel.alpha = 1.0
                },
                completion: nil
            )
        case .showBefore:
            self.stackView.insertArrangedSubview(beforeView, at: 0)
            UIView.animate(
                withDuration: 0.40,
                delay: 0.0,
                usingSpringWithDamping: 0.6,
                initialSpringVelocity: 0.2,
                options: [],
                animations: {
                    self.beforeView.isHidden = false
                    self.beforeLabel.alpha = 1.0
                    self.afterView.isHidden = true
                    self.afterLabel.alpha = 0.0
                },
                completion: { _ in
                    self.stackView.removeArrangedSubview(self.afterView)
                }
            )
        case .showAfter:
            self.stackView.insertArrangedSubview(afterView, at: 0)
            UIView.animate(
                withDuration: 0.40,
                delay: 0.0,
                usingSpringWithDamping: 0.6,
                initialSpringVelocity: 0.2,
                options: [],
                animations: {
                    self.beforeView.isHidden = true
                    self.beforeLabel.alpha = 0.0
                    self.afterView.isHidden = false
                    self.afterLabel.alpha = 1.0
                },
                completion: {_ in
                    self.stackView.removeArrangedSubview(self.beforeView)
                }
            )
        }
    }

    private func updateToolbar(for mode: RootViewModel.Mode) {
        let common = [UIBarButtonItem(barButtonSystemItem: .flexibleSpace, target: nil, action: nil), modeButton]
        switch mode {
        case .showBoth:
            setToolbarItems([flipButton] + common, animated: true)
        case .showBefore, .showAfter:
            setToolbarItems(common, animated: true)
        }
    }
}

extension RootViewController: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {

    }

    func draw(in view: MTKView) {
        //if view === beforeView {
            guard let drawable = view.currentDrawable,
                  let descriptor = view.currentRenderPassDescriptor
                else { return }
            renderer.draw(to: drawable, using: descriptor)
        //}
    }
}
