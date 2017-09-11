import UIKit

struct RootBuilder {
    func make() -> UIViewController {
        let navigationController = UINavigationController()
        let viewModel = RootViewModel()
        let rootViewController = RootViewController(viewModel: viewModel)
        navigationController.setViewControllers([rootViewController], animated: false)
        return navigationController
    }
}
