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
            AVCaptureDevice.requestAccess(forMediaType: AVMediaTypeVideo) { hasGranted in
                observer.send(hasGranted ? .completed : .failed(.unauthorized))
            }
        }
    }

    static var getDevice: SignalProducer<AVCaptureDevice, CameraError> {
        let device = AVCaptureDevice.defaultDevice(withDeviceType: .builtInWideAngleCamera,
                                                   mediaType: AVMediaTypeVideo,
                                                   position: .back)
        return SignalProducer(result: Result(device, failWith: .sourceCreationFailed(.noDevice)))
    }

    static func createIOAndSession(for device: AVCaptureDevice) -> SignalProducer<(AVCaptureSession, AVCaptureDeviceInput, AVCaptureVideoDataOutput), CameraError> {
        return SignalProducer { () -> Result<(AVCaptureSession, AVCaptureDeviceInput, AVCaptureVideoDataOutput), CameraError> in
            guard let input = try? AVCaptureDeviceInput(device: device)
                else { return .failure(.sourceCreationFailed(.inputCreationDenied)) }
            let output = AVCaptureVideoDataOutput()
            let session = AVCaptureSession()
            session.beginConfiguration()
            defer { session.commitConfiguration() }

            guard session.canAddInput(input)
                else { return .failure(.sourceCreationFailed(.inputRegistrationDenied)) }
            guard session.canAddOutput(output)
                else { return .failure(.sourceCreationFailed(.outputRegistrationDenied)) }

            return .success((session, input, output))
        }
    }

    static func make() -> SignalProducer<CameraSource, CameraError> {
        return ensureAuthorized
            .then(getDevice)
            .flatMap(.race, createIOAndSession)
            .map(self.init)
            .map { $0 }
    }

    private init(session: AVCaptureSession, input: AVCaptureDeviceInput, output: AVCaptureVideoDataOutput) {
        (self.session, self.input, self.output) = (session, input, output)
        queue = DispatchQueue(label: "com.andersha.Upscale.CameraSource.sampleBufferDelegateQueue")
        (samples, samplesObserver) = Signal.pipe()
        super.init()

        session.beginConfiguration()
        defer {
            session.commitConfiguration()
            session.startRunning()
        }

        output.setSampleBufferDelegate(self, queue: queue)
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        ]
    }
}

extension AVCameraSource: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput!, didOutputSampleBuffer sampleBuffer: CMSampleBuffer!, from connection: AVCaptureConnection!) {
        if let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            samplesObserver.send(value: imageBuffer)
        }
    }
}
