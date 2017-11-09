import UIKit
import TOCropViewController
import ReactiveSwift
import Result

protocol RootChildBuilders {
    func makeImageCropper(for image: UIImage, router: @escaping (ImageCropperViewModel.Event) -> Void) -> UIViewController
    func makeInferrer(for image: UIImage, router: @escaping (InferenceViewModel.Event) -> Void) -> UIViewController
}

struct _RootChildBuilders: RootChildBuilders {
    private let coordinator: MetalCoordinator

    init(coordinator: MetalCoordinator) {
        self.coordinator = coordinator
    }

    func makeInferrer(for image: UIImage, router: @escaping (InferenceViewModel.Event) -> Void) -> UIViewController {
        let viewModel = InferenceViewModel(original: image, coordinator: coordinator)
        let viewController = InferenceViewController(viewModel: viewModel)

        viewModel.events
            .observe(on: UIScheduler())
            .observeValues(router)

        return UINavigationController(rootViewController: viewController)
    }

    func makeImageCropper(for image: UIImage, router: @escaping (ImageCropperViewModel.Event) -> Void) -> UIViewController {
        let viewModel = ImageCropperViewModel(for: image)
        let viewController = ImageCropperViewController(viewModel: viewModel)

        viewModel.events
            .observe(on: UIScheduler())
            .observeValues(router)

        return viewController
    }
}

struct RootBuilder {
    func make(cameraSource: CameraSource) -> UIViewController {
        let navigationController = UINavigationController()
        let viewModel = RootViewModel(cameraSource: cameraSource)
        let renderer = MetalCoordinator()
        let rootViewController = RootViewController(viewModel: viewModel, renderer: renderer)
        navigationController.setViewControllers([rootViewController], animated: false)

        let flowController = RootFlowController(presenting: navigationController, builders: _RootChildBuilders(coordinator: renderer))

        viewModel.events
            .observe(on: UIScheduler())
            .observeValues(flowController.route(for:))

        cameraSource
            .samples
            .observeValues { [weak renderer] in renderer?.handle($0) }

        return navigationController
    }
}
