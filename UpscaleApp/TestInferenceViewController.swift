import AppKit
import Metal
import MetalPerformanceShaders
import CoreGraphics

class TestInferenceViewController: NSViewController {
    let coordinator: MetalCoordinator
    let original: NSImageView
    let inferred: NSImageView

    init() {
        coordinator = MetalCoordinator()
        original = NSImageView()
        inferred = NSImageView()
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func loadView() {
        view = NSView()
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        view.addSubview(original)
        view.addSubview(inferred)

        original.translatesAutoresizingMaskIntoConstraints = false
        inferred.translatesAutoresizingMaskIntoConstraints = false

        NSLayoutConstraint.activate([
            original.widthAnchor.constraint(equalToConstant: 128),
            original.heightAnchor.constraint(equalToConstant: 128),
            inferred.widthAnchor.constraint(equalToConstant: 128),
            inferred.heightAnchor.constraint(equalToConstant: 128),
            original.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            original.trailingAnchor.constraint(equalTo: inferred.leadingAnchor, constant: 10),
            inferred.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            original.topAnchor.constraint(equalTo: view.topAnchor),
            original.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            inferred.centerYAnchor.constraint(equalTo: original.centerYAnchor)
        ])

        let image = NSImage(contentsOf: Bundle.main.url(forResource: "one_input", withExtension: "png")!)!
        original.image = image

        let texture = make(image: image, device: coordinator.device)
        let mpsImage = MPSImage(texture: texture, featureChannels: 3)
        coordinator.superResolution.infer(mpsImage) { result in
            DispatchQueue.main.async {
                self.inferred.image = NSImage(cgImage: CGImage.make(texture: result.texture),
                                              size: NSSize(width: result.width, height: result.height))
            }
        }
    }
}

    func make(image: NSImage, device: MTLDevice) -> MTLTexture {
        var imageRect = NSMakeRect(0, 0, image.size.width, image.size.height)
        let cgImage = image.cgImage(forProposedRect: &imageRect, context: nil, hints: nil)!

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let rawData = UnsafeMutableRawPointer.allocate(bytes: cgImage.height * cgImage.width * 4, alignedTo: 4)

        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * cgImage.width
        let bitsPerComponent = 8
        let bitmapContext = CGContext(data: rawData,
                                      width: cgImage.width,
                                      height: cgImage.height,
                                      bitsPerComponent: bitsPerComponent,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue)!

        // Flip the context so the positive Y axis points down
        bitmapContext.translateBy(x: 0, y: CGFloat(cgImage.height))
        bitmapContext.scaleBy(x: 1, y: -1)
        bitmapContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: CGFloat(cgImage.width), height: CGFloat(cgImage.height)))

        let textureDescriptor = MTLTextureDescriptor
            .texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                 width: cgImage.width,
                                 height: cgImage.height,
                                 mipmapped: false)
        let texture = device.makeTexture(descriptor: textureDescriptor)!

        let region = MTLRegionMake2D(0, 0, cgImage.width, cgImage.height)
        texture.replace(region: region, mipmapLevel: 0, withBytes: rawData, bytesPerRow: bytesPerRow)

        rawData.deallocate(bytes: cgImage.height * cgImage.width * 4, alignedTo: 4)

        return texture
    }

