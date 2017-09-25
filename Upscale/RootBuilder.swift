import UIKit

struct RootBuilder {
    func make(cameraSource: CameraSource) -> UIViewController {
        let navigationController = UINavigationController()
        let viewModel = RootViewModel()
        let renderer = MetalCoordinator(cameraSource: cameraSource)
        let rootViewController = RootViewController(viewModel: viewModel, renderer: renderer)
        navigationController.setViewControllers([rootViewController], animated: false)

        return navigationController
    }
}
