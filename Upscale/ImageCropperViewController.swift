import TOCropViewController
import UIKit

class ImageCropperViewController: TOCropViewController, TOCropViewControllerDelegate {
    private var viewModel: ImageCropperViewModel!

    convenience init(viewModel: ImageCropperViewModel) {
        self.init(croppingStyle: .default, image: viewModel.image)

        self.viewModel = viewModel
        aspectRatioLockEnabled = true
        aspectRatioPreset = .presetSquare
        rotateButtonsHidden = true
        rotateClockwiseButtonHidden = true
        delegate = self
    }

    func cropViewController(_ cropViewController: TOCropViewController, didCropToRect cropRect: CGRect, angle: Int) {
        viewModel.commit(cropRect)
    }
}
