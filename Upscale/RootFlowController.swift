import UIKit

final class RootFlowController {
    private weak var presenting: UIViewController?
    private let builders: RootChildBuilders

    init(presenting: UIViewController, builders: RootChildBuilders) {
        self.presenting = presenting
        self.builders = builders
    }

    func route(for event: RootViewModel.Event) {
        switch event {
        case .showCropper(let image):
            builders.makeImageCropper(for: image, router: self.route(for:))
                |> { presenting?.present($0, animated: true, completion: nil) }
        }
    }

    func route(for event: ImageCropperViewModel.Event) {
        switch event {
        case let .userDidFinishCropping(image):
            presenting?.dismiss(animated: true, completion: nil)

            builders.makeInferrer(for: image, router: self.route(for:))
                |> { presenting?.present($0, animated: true, completion: nil) }
        }
    }

    func route(for event: InferenceViewModel.Event) {
        switch event {
        case .close:
            presenting?.dismiss(animated: true, completion: nil)
        }
    }
}
