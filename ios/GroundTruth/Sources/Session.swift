import Foundation
import UIKit

let consentVersion = "gt-consent-1"  // must match CONSENT_VERSION in ground/server.py

/// The link Photon texted: groundtruth://join?token=...&server=https://...
struct Invite: Codable, Equatable {
    var token: String
    var server: String
}

struct SessionState: Codable {
    enum Status: String, Codable { case consented, recording, ended, uploaded }

    var id: String
    var token: String
    var server: String
    var placement: String = "pocket"
    var status: Status = .consented
    var started: Double?
    var ended: Double?
    var lastImuT: Double?
    var gaps: [[Double]] = []
    var segment: Int = 0
    var counts: [String: Int] = [:]
    var distance: Double = 0
    var pedometer: [String: Double]?
    var warnings: [String] = []
    var climb: Double?  // metres gained, from the barometer
}

enum SessionStore {
    static var root: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("sessions")
    }

    static func dir(_ id: String) -> URL {
        let d = root.appendingPathComponent(id)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    static func save(_ s: SessionState) {
        let url = dir(s.id).appendingPathComponent("state.json")
        if let data = try? JSONEncoder().encode(s) { try? data.write(to: url, options: .atomic) }
    }

    static func load(_ id: String) -> SessionState? {
        let url = root.appendingPathComponent(id).appendingPathComponent("state.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(SessionState.self, from: data)
    }

    /// The newest session that isn't fully uploaded, if any.
    static func unfinished() -> SessionState? {
        let ids = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        return ids.compactMap(load).filter { $0.status != .uploaded }.max { ($0.started ?? 0) < ($1.started ?? 0) }
    }

    /// Data files for upload, each with its size; meta.json lists them so the server knows when it has everything.
    static func dataFiles(_ id: String) -> [(name: String, size: Int)] {
        let d = dir(id)
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: d.path)) ?? []).filter { $0.hasSuffix(".bin") }.sorted()
        return names.map { n in
            let attrs = try? FileManager.default.attributesOfItem(atPath: d.appendingPathComponent(n).path)
            return (n, (attrs?[.size] as? Int) ?? 0)
        }
    }

    static func writeMeta(_ s: SessionState) throws {
        var files: [String: Int] = [:]
        for f in dataFiles(s.id) { files[f.name] = f.size }
        var sys = utsname()
        uname(&sys)
        let model = withUnsafeBytes(of: &sys.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        let meta: [String: Any] = [
            "session_id": s.id, "placement": s.placement, "consent_version": consentVersion,
            "started": s.started ?? 0, "ended": s.ended ?? 0, "gaps_on_device": s.gaps, "counts": s.counts,
            "pedometer": s.pedometer ?? [:], "warnings": s.warnings, "files": files,
            "device": ["model": model, "ios": UIDevice.current.systemVersion],
            "timezone": TimeZone.current.identifier,
        ]
        let data = try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: dir(s.id).appendingPathComponent("meta.json"), options: .atomic)
    }
}
