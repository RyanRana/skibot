import Foundation

/// Writes sensor records to disk in the layout ground/schema.py reads: packed little endian, one file per kind per
/// 5 minute segment (imu_000.bin, gps_000.bin, baro_000.bin, ...). Everything runs on `queue`; the motion and
/// altimeter handlers are delivered straight onto it, location fixes are hopped onto it.
final class SessionWriter {
    static let imuSize = 72, gpsSize = 44, baroSize = 24  // must match ground/schema.py
    static let segmentSeconds: Double = 300
    static let gapSeconds: Double = 0.5

    let dir: URL
    let queue = DispatchQueue(label: "groundtruth.writer", qos: .userInitiated)
    private var state: SessionState
    private var handles: [String: FileHandle] = [:]
    private var buffers: [String: Data] = ["imu": Data(), "gps": Data(), "baro": Data()]
    private var segmentStarted = Date()
    private var timer: DispatchSourceTimer?
    private(set) var lastError: String?

    /// What the recording screen shows, updated as samples arrive and summarised once a second.
    struct Live {
        var accRms = 0.0     // user acceleration, m/s^2, root mean square over the last second
        var gyroDps = 0.0    // rotation rate, degrees per second, root mean square over the last second
        var relAlt: Double?  // metres since the start, barometer
        var kpa: Double?     // air pressure
    }
    private var live = Live()
    private var accSq = 0.0, gyroSq = 0.0, liveN = 0
    private var climbRef: Double?
    static let climbStep = 1.0  // a rise has to beat this to count as climbing, so barometer noise does not

    init(state: SessionState) {
        self.state = state
        self.dir = SessionStore.dir(state.id)
        // A resumed session starts a fresh segment so nothing is appended after a record cut off by a crash.
        if state.counts.values.reduce(0, +) > 0 { self.state.segment += 1 }
    }

    func start() {
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + 1, repeating: 1)
        t.setEventHandler { [weak self] in self?.flush() }
        t.resume()
        timer = t
    }

    /// Flushes, closes files and returns the final state. Call from any thread.
    func close() -> SessionState {
        queue.sync {
            timer?.cancel()
            timer = nil
            flush()
            handles.values.forEach { try? $0.close() }
            handles = [:]
            return state
        }
    }

    func snapshot() -> SessionState { queue.sync { state } }
    func liveSnapshot() -> (SessionState, Live) { queue.sync { (state, live) } }

    // MARK: records (call on `queue`)

    /// Motion sample: unix time, user acceleration and gravity in m/s^2, rotation rate in rad/s, quaternion, magnetic field in uT.
    func imu(t: Double, acc: (Double, Double, Double), grav: (Double, Double, Double), gyro: (Double, Double, Double),
             quat: (Double, Double, Double, Double), mag: (Double, Double, Double)?) {
        if let last = state.lastImuT, t - last > Self.gapSeconds { state.gaps.append([last, t]) }
        state.lastImuT = t
        accSq += acc.0 * acc.0 + acc.1 * acc.1 + acc.2 * acc.2
        gyroSq += gyro.0 * gyro.0 + gyro.1 * gyro.1 + gyro.2 * gyro.2
        liveN += 1
        var d = Data(capacity: Self.imuSize)
        d.put(t)
        for v in [acc.0, acc.1, acc.2, grav.0, grav.1, grav.2, gyro.0, gyro.1, gyro.2, quat.0, quat.1, quat.2, quat.3] {
            d.put(Float32(v))
        }
        for v in mag.map({ [$0.0, $0.1, $0.2] }) ?? [.nan, .nan, .nan] { d.put(Float32(v)) }
        append("imu", d)
    }

    func gps(t: Double, lat: Double, lon: Double, alt: Double, hacc: Double, vacc: Double, speed: Double,
             course: Double, stepMeters: Double) {
        var d = Data(capacity: Self.gpsSize)
        d.put(t); d.put(lat); d.put(lon)
        for v in [alt, hacc, vacc, speed, course] { d.put(Float32(v)) }
        state.distance += stepMeters
        append("gps", d)
    }

    func baro(t: Double, relAlt: Double, kpa: Double, absAlt: Double, absAcc: Double) {
        var d = Data(capacity: Self.baroSize)
        d.put(t)
        for v in [relAlt, kpa, absAlt, absAcc] { d.put(Float32(v)) }
        append("baro", d)
        if relAlt.isFinite {
            live.relAlt = relAlt
            live.kpa = kpa
            if let ref = climbRef {
                if relAlt - ref >= Self.climbStep { state.climb = (state.climb ?? 0) + relAlt - ref; climbRef = relAlt }
                else if ref - relAlt >= Self.climbStep { climbRef = relAlt }
            } else {
                climbRef = relAlt
            }
        }
    }

    // MARK: files

    private func append(_ kind: String, _ d: Data) {
        buffers[kind, default: Data()].append(d)
        state.counts[kind, default: 0] += 1
    }

    private func flush() {
        if liveN > 0 {
            live.accRms = (accSq / Double(liveN)).squareRoot()
            live.gyroDps = (gyroSq / Double(liveN)).squareRoot() * 180 / .pi
            accSq = 0; gyroSq = 0; liveN = 0
        }
        if Date().timeIntervalSince(segmentStarted) >= Self.segmentSeconds, !handles.isEmpty {
            writeBuffers()
            handles.values.forEach { try? $0.close() }
            handles = [:]
            state.segment += 1
            segmentStarted = Date()
        }
        writeBuffers()
        SessionStore.save(state)
    }

    private func writeBuffers() {
        for (kind, data) in buffers where !data.isEmpty {
            do {
                try handle(kind).write(contentsOf: data)
                buffers[kind] = Data()
            } catch {
                lastError = "could not write \(kind): \(error.localizedDescription)"  // buffer kept, retried next second
            }
        }
    }

    private func handle(_ kind: String) throws -> FileHandle {
        if let h = handles[kind] { return h }
        let url = dir.appendingPathComponent(String(format: "%@_%03d.bin", kind, state.segment))
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        let h = try FileHandle(forWritingTo: url)
        try h.seekToEnd()
        handles[kind] = h
        return h
    }
}

extension Data {
    /// Appends the raw little endian bytes of a value (iOS is little endian).
    mutating func put<T>(_ v: T) { Swift.withUnsafeBytes(of: v) { append(contentsOf: $0) } }
}
