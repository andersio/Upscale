#import <Foundation/Foundation.h>
#import "Upscale-Bridging-Header.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session.h"

@implementation TFModel: NSObject
tensorflow::Session *session;

-(instancetype) init {
    self = [super init];

    tensorflow::SessionOptions options;
    tensorflow::Status status = tensorflow::NewSession(options, &session);
    NSAssert(status.ok(), @"TF is not initialised properly.");

    return self;
}

@end
