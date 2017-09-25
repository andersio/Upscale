import UIKit
import ReactiveSwift
import Result

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplicationLaunchOptionsKey: Any]?) -> Bool {
        let window = UIWindow()
        self.window = window

        AVCameraSource.make()
            .observe(on: UIScheduler())
            .startWithResult { result in
                switch result {
                case let .success(source):
                    let builder = RootBuilder()
                    window.rootViewController = builder.make(cameraSource: source)
                    window.makeKeyAndVisible()
                case let .failure(error):
                    window.rootViewController = UIViewController()
                    window.makeKeyAndVisible()

                    let alertController = UIAlertController(title: "Error", message: "\(error)", preferredStyle: .alert)
                    window.rootViewController!.present(alertController, animated: true, completion: nil)
                }
            }

        return true
    }

}

