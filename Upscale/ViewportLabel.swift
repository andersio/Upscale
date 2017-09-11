import UIKit

class ViewportLabel: UIView {
    let effect: UIBlurEffect
    let background: UIVisualEffectView
    let vibrancy: UIVisualEffectView
    let label: UILabel

    init() {
        label = UILabel()
        effect = UIBlurEffect(style: .dark)
        background = UIVisualEffectView(effect: effect)
        vibrancy = UIVisualEffectView(effect: UIVibrancyEffect(blurEffect: effect))

        super.init(frame: .zero)

        translatesAutoresizingMaskIntoConstraints = false
        label.translatesAutoresizingMaskIntoConstraints = false
        background.translatesAutoresizingMaskIntoConstraints = false

        addSubview(background)
        background.contentView.addSubview(label)

        NSLayoutConstraint.activate([
            background.topAnchor.constraint(equalTo: topAnchor),
            background.bottomAnchor.constraint(equalTo: bottomAnchor),
            background.leftAnchor.constraint(equalTo: leftAnchor),
            background.rightAnchor.constraint(equalTo: rightAnchor),
            label.leftAnchor.constraint(equalTo: layoutMarginsGuide.leftAnchor),
            label.topAnchor.constraint(equalTo: layoutMarginsGuide.topAnchor),
            label.rightAnchor.constraint(equalTo: layoutMarginsGuide.rightAnchor),
            label.bottomAnchor.constraint(equalTo: layoutMarginsGuide.bottomAnchor)
        ])

        layoutMargins = UIEdgeInsets(top: 2, left: 8, bottom: 2, right: 8)
        backgroundColor = .clear
        layer.masksToBounds = true
        label.font = UIFont.preferredFont(forTextStyle: .subheadline)
        label.textColor = UIColor(white: 0.95, alpha: 0.85)
    }

    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        layer.cornerRadius = frame.height / 2.0
    }
}
