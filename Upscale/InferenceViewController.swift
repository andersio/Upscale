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
        activityIndicator.color = .gray
        activityIndicator.reactive.isHidden <~ viewModel.isExecuting.negate()
        activityIndicator.reactive.isAnimating <~ viewModel.isExecuting

        view.addSubview(activityIndicator)

        NSLayoutConstraint.activate([
            imageView.widthAnchor.constraint(equalToConstant: 128),
            imageView.heightAnchor.constraint(equalToConstant: 128),
            imageView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            imageView.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            activityIndicator.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            activityIndicator.centerXAnchor.constraint(equalTo: view.centerXAnchor),
        ])

        navigationItem.leftBarButtonItem = UIBarButtonItem(title: "Close", style: .plain, target: viewModel, action: #selector(viewModel.close))
    }
}
