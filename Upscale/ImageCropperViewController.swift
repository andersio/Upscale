import TOCropViewController
import UIKit

class ImageCropperViewController: TOCropViewController, TOCropViewControllerDelegate {
    private var viewModel: ImageCropperViewModel!

    convenience init(viewModel: ImageCropperViewModel) {
        self.init(croppingStyle: .default, image: viewModel.image)

        self.viewModel = viewModel
        aspectRatioLockEnabled = true
        resetAspectRatioEnabled = false
        aspectRatioPreset = .presetSquare
        delegate = self
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        rotateButtonsHidden = true
        rotateClockwiseButtonHidden = true
    }

    func cropViewController(_ cropViewController: TOCropViewController, didCropToRect cropRect: CGRect, angle: Int) {
        viewModel.commit(cropRect, angle)
    }
}
