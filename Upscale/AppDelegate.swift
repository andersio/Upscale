import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplicationLaunchOptionsKey: Any]?) -> Bool {
        let builder = RootBuilder()
        let window = UIWindow()
        self.window = window

        window.rootViewController = builder.make()
        window.makeKeyAndVisible()

        return true
    }

}

