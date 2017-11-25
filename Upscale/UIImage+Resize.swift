import UIKit
import CoreGraphics

extension UIImage {
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, true, 0.0)
        UIGraphicsGetCurrentContext()!.interpolationQuality = .high
        draw(in: CGRect(origin: .zero, size: size))
        defer { UIGraphicsEndImageContext() }
        return UIGraphicsGetImageFromCurrentImageContext()
    }

    func resized(to size: CGSize, cropping rect: CGRect) -> UIImage? {
        let cropped = cgImage!.cropping(to: CGRect(x: rect.minX * scale,
                                                   y: rect.minY * scale,
                                                   width: rect.width * scale,
                                                   height: rect.height * scale))!

        UIGraphicsBeginImageContext(size)
        let context = UIGraphicsGetCurrentContext()!
        context.interpolationQuality = .high
        context.concatenate(CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: size.height))
        context.draw(cropped, in: CGRect(origin: .zero, size: size))
        defer { UIGraphicsEndImageContext() }
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}
