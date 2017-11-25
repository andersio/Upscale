import AVFoundation
import ReactiveSwift
import Result

protocol CameraSource: class {
    var samples: Signal<CVImageBuffer, NoError> { get }
    static func make() -> SignalProducer<CameraSource, CameraError>
}

enum CameraError: Error {
    enum SourceCreationFailure {
        case noDevice
        case inputCreationDenied
        case inputRegistrationDenied
        case outputRegistrationDenied
    }

    case unauthorized
    case sourceCreationFailed(SourceCreationFailure)
}

final class AVCameraSource: NSObject, CameraSource {
    private let queue: DispatchQueue
    private let session: AVCaptureSession
    private let input: AVCaptureDeviceInput
    private let output: AVCaptureVideoDataOutput

    let samples: Signal<CVImageBuffer, NoError>
    fileprivate let samplesObserver: Signal<CVImageBuffer, NoError>.Observer

    static var ensureAuthorized: SignalProducer<Never, CameraError> {
        return SignalProducer { observer, _ in
            AVCaptureDevice.requestAccess(for: .video) { hasGranted in
                observer.send(hasGranted ? .completed : .failed(.unauthorized))
            }
        }
    }

    static var getDevice: SignalProducer<AVCaptureDevice, CameraError> {
        let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                             for: .video,
                                             position: .front)
        return SignalProducer(result: Result(device, failWith: .sourceCreationFailed(.noDevice)))
    }

    static func createIOAndSession(for device: AVCaptureDevice) -> SignalProducer<CameraSource, CameraError> {
        return SignalProducer { () -> Result<CameraSource, CameraError> in
            guard let input = try? AVCaptureDeviceInput(device: device)
                else { return .failure(.sourceCreationFailed(.inputCreationDenied)) }
            let output = AVCaptureVideoDataOutput()
            let session = AVCaptureSession()
            session.beginConfiguration()

            guard session.canAddInput(input)
                else { return .failure(.sourceCreationFailed(.inputRegistrationDenied)) }

            let source = self.init(session: session, input: input, output: output)
            output.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            output.setSampleBufferDelegate(source, queue: source.queue)

            guard session.canAddOutput(output)
                else { return .failure(.sourceCreationFailed(.outputRegistrationDenied)) }

            session.addInput(input)
            session.addOutput(output)

            defer {
                session.commitConfiguration()
                session.startRunning()
            }

            return .success(source)
        }
    }

    static func make() -> SignalProducer<CameraSource, CameraError> {
        return ensureAuthorized
            .then(getDevice)
            .flatMap(.race, createIOAndSession)
    }

    private init(session: AVCaptureSession, input: AVCaptureDeviceInput, output: AVCaptureVideoDataOutput) {
        (self.session, self.input, self.output) = (session, input, output)
        queue = DispatchQueue(label: "com.andersha.Upscale.CameraSource.sampleBufferDelegateQueue")
        (samples, samplesObserver) = Signal.pipe()
        super.init()
    }
}

extension AVCameraSource: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            samplesObserver.send(value: imageBuffer)
        }
    }
}
