import CoreGraphics
import Metal

extension CGImage {
    static func make(texture: MTLTexture) -> CGImage {
        assert(texture.pixelFormat == .rgba8Unorm, "Pixel format of texture must be MTLPixelFormatRGBA8Unorm to create UIImage")

        let imageByteCount = texture.width * texture.height * 4
        let imageBytes = malloc(imageByteCount)!
        let bytesPerRow = texture.width * 4
        let region = MTLRegionMake2D(0, 0, texture.width, texture.height)

        texture.getBytes(imageBytes, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)

        for pixel in 0 ..< texture.width * texture.height {
            imageBytes.advanced(by: pixel * 4 + 3)
                .assumingMemoryBound(to: UInt8.self)
                .pointee = .max
        }

        let provider = CGDataProvider(dataInfo: nil,
                                      data: imageBytes,
                                      size: imageByteCount,
                                      releaseData: { _, data, _ in free(UnsafeMutableRawPointer(mutating: data)) })!
        let bitsPerComponent = 8
        let bitsPerPixel = 32
        let colorSpaceRef = CGColorSpaceCreateDeviceRGB();
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.first.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        return self.init(width: texture.width,
                         height: texture.height,
                         bitsPerComponent: bitsPerComponent,
                         bitsPerPixel: bitsPerPixel,
                         bytesPerRow: bytesPerRow,
                         space: colorSpaceRef,
                         bitmapInfo: bitmapInfo,
                         provider: provider,
                         decode: nil,
                         shouldInterpolate: false,
                         intent: .defaultIntent)!
    }
}
