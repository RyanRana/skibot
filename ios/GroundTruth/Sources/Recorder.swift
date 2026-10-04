import CoreLocation
import CoreMotion
import Foundation

/// Runs the sensors for one hike. Background location (UIBackgroundModes: location, started in the foreground) is
/// what keeps the app alive with the screen locked; motion and the barometer keep delivering while it is alive.
final class Recorder: NSObject, ObservableObject, CLLocationManagerDelegate {
    @Published private(set) var running = false
    @Published private(set) var live: SessionState?
    @Published private(set) var hacc: Double = -1
    @Published private(set) var problems: [String] = []
    @Published var error: String?

    private let motion = CMMotionManager()
    private let altimeter = CMAltimeter()
    private let pedometer = CMPedometer()
    private let location = CLLocationManager()
    private let sensorQueue = OperationQueue()
    private var writer: SessionWriter?
    private var lastGoodFix: CLLocation?
    private var uiTimer: Timer?
    private var onAuthorized: (() -> Void)?

    static let deniedLocation = "location is off for ground truth. turn it on in settings > privacy & security > location services > ground truth > while using the app."

    override init() {
        super.init()
        location.delegate = self
        location.activityType = .fitness
        location.desiredAccuracy = kCLLocationAccuracyBest
        location.distanceFilter = kCLDistanceFilterNone
        location.pausesLocationUpdatesAutomatically = false
        sensorQueue.maxConcurrentOperationCount = 1
    }

    func start(_ state: SessionState) {
        error = nil
        switch location.authorizationStatus {
        case .notDetermined:
            onAuthorized = { [weak self] in self?.begin(state) }
            location.requestWhenInUseAuthorization()
        case .denied, .restricted:
            error = Self.deniedLocation
        default:
            begin(state)
        }
    }

    func locationManagerDidChangeAuthorization(_ m: CLLocationManager) {
        guard let next = onAuthorized else { return }
        switch m.authorizationStatus {
        case .authorizedWhenInUse, .authorizedAlways:
            onAuthorized = nil
            next()
        case .denied, .restricted:
            onAuthorized = nil
            error = Self.deniedLocation
        default:
            break
        }
    }

    private func begin(_ initial: SessionState) {
        var s = initial
        s.status = .recording
        if s.started == nil { s.started = Date().timeIntervalSince1970 }
        problems = []
        let w = SessionWriter(state: s)
        writer = w
        sensorQueue.underlyingQueue = w.queue
        w.start()

        location.allowsBackgroundLocationUpdates = true
        location.showsBackgroundLocationIndicator = true
        location.startUpdatingLocation()

        if motion.isDeviceMotionAvailable {
            motion.deviceMotionUpdateInterval = 1.0 / 100
            motion.startDeviceMotionUpdates(using: .xArbitraryCorrectedZVertical, to: sensorQueue) { [weak self, weak w] m, err in
                guard let w, let m else {
                    if let err { self?.report("motion: \(err.localizedDescription)") }
                    return
                }
                let g = 9.80665
                let a = m.userAcceleration, gr = m.gravity, r = m.rotationRate, q = m.attitude.quaternion
                let f = m.magneticField
                w.imu(t: unixTime(m.timestamp), acc: (a.x * g, a.y * g, a.z * g), grav: (gr.x * g, gr.y * g, gr.z * g),
                      gyro: (r.x, r.y, r.z), quat: (q.x, q.y, q.z, q.w),
                      mag: f.accuracy == .uncalibrated ? nil : (f.field.x, f.field.y, f.field.z))
            }
        } else {
            report("no motion sensors on this device (simulator?)")
        }

        if CMAltimeter.isRelativeAltitudeAvailable() {
            altimeter.startRelativeAltitudeUpdates(to: sensorQueue) { [weak self, weak w] d, err in
                guard let w, let d else {
                    if let err { self?.report("barometer: \(err.localizedDescription) (motion & fitness permission?)") }
                    return
                }
                w.baro(t: unixTime(d.timestamp), relAlt: d.relativeAltitude.doubleValue, kpa: d.pressure.doubleValue,
                       absAlt: .nan, absAcc: .nan)
            }
        } else {
            report("no barometer on this device")
        }
        if CMAltimeter.isAbsoluteAltitudeAvailable() {
            altimeter.startAbsoluteAltitudeUpdates(to: sensorQueue) { [weak w] d, _ in
                guard let w, let d else { return }
                w.baro(t: unixTime(d.timestamp), relAlt: .nan, kpa: .nan, absAlt: d.altitude, absAcc: d.accuracy)
            }
        }

        running = true
        uiTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in self?.refresh() }
        refresh()
    }

    /// Stops every sensor, closes the files, adds the step count, and returns the finished session.
    func stop() async -> SessionState? {
        motion.stopDeviceMotionUpdates()
        altimeter.stopRelativeAltitudeUpdates()
        altimeter.stopAbsoluteAltitudeUpdates()
        location.stopUpdatingLocation()
        location.allowsBackgroundLocationUpdates = false
        uiTimer?.invalidate()
        uiTimer = nil
        guard let w = writer else { return nil }
        var s = w.close()
        writer = nil
        s.status = .ended
        s.ended = Date().timeIntervalSince1970
        s.warnings += problems
        if let st = s.started, CMPedometer.isStepCountingAvailable() {
            s.pedometer = await steps(from: Date(timeIntervalSince1970: st), to: Date(timeIntervalSince1970: s.ended!))
        }
        SessionStore.save(s)
        running = false
        live = s
        return s
    }

    private func steps(from: Date, to: Date) async -> [String: Double]? {
        await withCheckedContinuation { c in
            pedometer.queryPedometerData(from: from, to: to) { d, _ in
                guard let d else { return c.resume(returning: nil) }
                var out = ["steps": d.numberOfSteps.doubleValue]
                if let v = d.distance { out["distance_m"] = v.doubleValue }
                if let v = d.floorsAscended { out["floors_up"] = v.doubleValue }
                if let v = d.floorsDescended { out["floors_down"] = v.doubleValue }
                c.resume(returning: out)
            }
        }
    }

    func locationManager(_ m: CLLocationManager, didUpdateLocations locs: [CLLocation]) {
        guard let w = writer else { return }
        for l in locs {
            var step = 0.0
            if l.horizontalAccuracy >= 0, l.horizontalAccuracy <= 30 {
                if let p = lastGoodFix { step = l.distance(from: p) }
                lastGoodFix = l
            }
            let c = l.coordinate, stepM = step
            w.queue.async {
                w.gps(t: l.timestamp.timeIntervalSince1970, lat: c.latitude, lon: c.longitude, alt: l.altitude,
                      hacc: l.horizontalAccuracy, vacc: l.verticalAccuracy, speed: l.speed, course: l.course, stepMeters: stepM)
            }
        }
        hacc = locs.last?.horizontalAccuracy ?? -1
    }

    func locationManager(_ m: CLLocationManager, didFailWithError e: Error) {
        if (e as? CLError)?.code == .locationUnknown { return }  // transient, iOS keeps trying
        report("gps: \(e.localizedDescription)")
    }

    private func report(_ msg: String) {
        DispatchQueue.main.async {
            if !self.problems.contains(msg) { self.problems.append(msg) }
        }
    }

    private func refresh() {
        guard let w = writer else { return }
        live = w.snapshot()
        if let e = w.lastError { report(e) }
    }
}

/// CoreMotion timestamps count seconds since boot; convert to unix time.
private func unixTime(_ uptime: TimeInterval) -> Double {
    uptime + Date().timeIntervalSince1970 - ProcessInfo.processInfo.systemUptime
}
