import UIKit

extension UIImage {
    func resized(to size: CGSize, cropping rect: CGRect? = nil) -> UIImage? {
        let rect = rect ?? CGRect(origin: .zero, size: self.size)
        UIGraphicsBeginImageContextWithOptions(size, true, scale)
        UIRectClip(CGRect(x: -rect.minX, y: -rect.minY, width: rect.width, height: rect.height))
        draw(in: CGRect(origin: .zero, size: size))
        defer { UIGraphicsEndImageContext() }
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}
