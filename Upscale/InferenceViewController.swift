import UIKit
import ReactiveSwift
import ReactiveCocoa

class InferenceViewController: UIViewController {
    private let viewModel: InferenceViewModel
    private var imageView: UIImageView!

    init(viewModel: InferenceViewModel) {
        self.viewModel = viewModel
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder aDecoder: NSCoder) {
        fatalError()
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        view.backgroundColor = .white

        imageView = UIImageView()
        imageView.image = viewModel.original
        imageView.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(imageView)

        let activityIndicator = UIActivityIndicatorView()
        activityIndicator.translatesAutoresizingMaskIntoConstraints = false
        activityIndicator.color = .gray
        activityIndicator.reactive.isHidden <~ viewModel.image.map { $0 != nil }
        activityIndicator.reactive.isAnimating <~ viewModel.image.map { $0 == nil }

        view.addSubview(activityIndicator)

        NSLayoutConstraint.activate([
            imageView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            imageView.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            activityIndicator.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            activityIndicator.centerXAnchor.constraint(equalTo: view.centerXAnchor),
        ])

        navigationItem.leftBarButtonItem = UIBarButtonItem(title: "Close", style: .plain, target: viewModel, action: #selector(viewModel.close))

        let segmentedControl = UISegmentedControl()
        toolbarItems = [UIBarButtonItem(customView: segmentedControl)]
        navigationController?.setToolbarHidden(false, animated: true)

        segmentedControl.insertSegment(withTitle: "Original", at: 0, animated: false)
        segmentedControl.insertSegment(withTitle: "Subpixel", at: 1, animated: false)
        segmentedControl.insertSegment(withTitle: "Bilinear", at: 2, animated: false)
        segmentedControl.selectedSegmentIndex = 0

        segmentedControl.reactive.selectedSegmentIndexes
            .take(duringLifetimeOf: self)
            .observeValues { [viewModel, imageView] index in
                switch index {
                case 0:
                    imageView?.image = viewModel.original
                case 1:
                    imageView?.image = viewModel.image.value
                case 2:
                    imageView?.image = nil
                default:
                    fatalError()
                }
            }
    }
}
