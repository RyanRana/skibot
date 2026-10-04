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
    // Live values for the recording screen.
    @Published private(set) var motionNow = SessionWriter.Live()
    @Published private(set) var speed: Double = -1     // m/s from GPS, -1 when unknown
    @Published private(set) var course: Double = -1    // degrees from north, -1 when unknown
    @Published private(set) var steps: Int?            // since the hike started, from the step counter
    @Published private(set) var cadence: Double?       // steps per minute, right now
    @Published private(set) var grade: Double?         // percent, over the last 30 m of trail
    @Published private(set) var here: CLLocationCoordinate2D?  // latest GPS fix, for the map

    private let motion = CMMotionManager()
    private let altimeter = CMAltimeter()
    private let pedometer = CMPedometer()
    private let location = CLLocationManager()
    private let sensorQueue = OperationQueue()
    private var writer: SessionWriter?
    private var lastGoodFix: CLLocation?
    private var uiTimer: Timer?
    private var onAuthorized: (() -> Void)?
    private var gradeTrail: [(dist: Double, alt: Double)] = []

    static let deniedLocation = "Location is off for ground truth. Turn it on in Settings > Privacy & Security > Location Services > ground truth > While Using the App."

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
                    if let err { self?.report("Motion: \(err.localizedDescription)") }
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
            report("No motion sensors on this device (simulator?)")
        }

        if CMAltimeter.isRelativeAltitudeAvailable() {
            altimeter.startRelativeAltitudeUpdates(to: sensorQueue) { [weak self, weak w] d, err in
                guard let w, let d else {
                    if let err { self?.report("Barometer: \(err.localizedDescription) (Motion & Fitness permission?)") }
                    return
                }
                w.baro(t: unixTime(d.timestamp), relAlt: d.relativeAltitude.doubleValue, kpa: d.pressure.doubleValue,
                       absAlt: .nan, absAcc: .nan)
            }
        } else {
            report("No barometer on this device")
        }
        if CMAltimeter.isAbsoluteAltitudeAvailable() {
            altimeter.startAbsoluteAltitudeUpdates(to: sensorQueue) { [weak w] d, _ in
                guard let w, let d else { return }
                w.baro(t: unixTime(d.timestamp), relAlt: .nan, kpa: .nan, absAlt: d.altitude, absAcc: d.accuracy)
            }
        }

        if CMPedometer.isStepCountingAvailable(), let st = s.started {
            pedometer.startUpdates(from: Date(timeIntervalSince1970: st)) { [weak self] d, _ in
                guard let d else { return }
                DispatchQueue.main.async {
                    self?.steps = d.numberOfSteps.intValue
                    self?.cadence = d.currentCadence.map { $0.doubleValue * 60 }
                }
            }
        } else {
            report("No step counter on this device")
        }

        steps = nil; cadence = nil; grade = nil; speed = -1; course = -1; gradeTrail = []; here = nil
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
        pedometer.stopUpdates()
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
        if let l = locs.last {
            if l.horizontalAccuracy >= 0 { here = l.coordinate }
            speed = l.speed >= 0 ? l.speed : -1
            course = l.course >= 0 ? l.course : -1
        }
    }

    func locationManager(_ m: CLLocationManager, didFailWithError e: Error) {
        if (e as? CLError)?.code == .locationUnknown { return }  // transient, iOS keeps trying
        report("GPS: \(e.localizedDescription)")
    }

    private func report(_ msg: String) {
        DispatchQueue.main.async {
            if !self.problems.contains(msg) { self.problems.append(msg) }
        }
    }

    private func refresh() {
        guard let w = writer else { return }
        let (state, now) = w.liveSnapshot()
        live = state
        motionNow = now
        // Slope: altitude change over the last 30 m of distance, both measured, so no map is needed.
        if let alt = now.relAlt {
            if gradeTrail.isEmpty || state.distance - gradeTrail.last!.dist >= 2 { gradeTrail.append((state.distance, alt)) }
            gradeTrail = Array(gradeTrail.suffix(200))
            if let old = gradeTrail.last(where: { state.distance - $0.dist >= 30 }) {
                grade = (alt - old.alt) / (state.distance - old.dist) * 100
            }
        }
        if let e = w.lastError { report(e) }
    }
}

/// CoreMotion timestamps count seconds since boot; convert to unix time.
private func unixTime(_ uptime: TimeInterval) -> Double {
    uptime + Date().timeIntervalSince1970 - ProcessInfo.processInfo.systemUptime
}
