//
//  AppDelegate.swift
//  UpscaleApp
//
//  Created by Anders on 9/11/2017.
//  Copyright © 2017 Anders Ha. All rights reserved.
//

import Cocoa

@NSApplicationMain
class AppDelegate: NSObject, NSApplicationDelegate {
    @IBOutlet weak var window: NSWindow!

    func applicationDidFinishLaunching(_ aNotification: Notification) {
        window.contentViewController = TestInferenceViewController()
    }
}
