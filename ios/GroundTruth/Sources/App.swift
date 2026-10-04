import SwiftUI
import UIKit

@main
struct GroundTruthApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(model)
                .environmentObject(model.recorder)
                .environmentObject(model.uploader)
                .onOpenURL { model.open($0) }
        }
    }
}

final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(_ application: UIApplication, handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        Uploader.shared.backgroundCompletion = completionHandler
    }
}

@MainActor
final class AppModel: ObservableObject {
    enum Stage { case noInvite, consent, setup, recording, interrupted, ended }

    @Published var invite: Invite? {
        didSet { UserDefaults.standard.set(try? JSONEncoder().encode(invite), forKey: "invite") }
    }
    @Published var session: SessionState?
    @Published var placement = "pocket"
    @Published var busy = false
    @Published var error: String?

    let recorder = Recorder()
    let uploader = Uploader.shared

    init() {
        if let d = UserDefaults.standard.data(forKey: "invite") { invite = try? JSONDecoder().decode(Invite.self, from: d) }
        session = SessionStore.unfinished()
        if let s = session, s.status == .ended { uploader.upload(s) }  // finish an upload cut off last time
        #if DEBUG
        Task { await autopilot() }
        #endif
    }

    #if DEBUG
    /// Simulator end-to-end test: launch with `-GTAutopilot <seconds> -GTJoin <join link>`. Runs the same
    /// consent, start, end and upload calls the buttons do.
    private func autopilot() async {
        let args = ProcessInfo.processInfo.arguments
        if args.contains("-GTDiscard") {
            try? await Task.sleep(for: .seconds(4))
            discard()
            return
        }
        guard let i = args.firstIndex(of: "-GTAutopilot"), i + 1 < args.count, let secs = Double(args[i + 1]) else { return }
        if let j = args.firstIndex(of: "-GTJoin"), j + 1 < args.count, let u = URL(string: args[j + 1]) {
            try? await Task.sleep(for: .seconds(2))
            open(u)
        }
        while invite == nil { try? await Task.sleep(for: .milliseconds(300)) }
        try? await Task.sleep(for: .seconds(3))
        await consent()
        try? await Task.sleep(for: .seconds(3))
        start()
        try? await Task.sleep(for: .seconds(secs))
        await end()
    }
    #endif

    var stage: Stage {
        if recorder.running { return .recording }
        if let s = session {
            switch s.status {
            case .consented: return .setup
            case .recording: return .interrupted
            case .ended, .uploaded: return .ended
            }
        }
        return invite == nil ? .noInvite : .consent
    }

    /// groundtruth://join?token=...&server=...
    func open(_ url: URL) {
        let q = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        guard url.host == "join", let token = q.first(where: { $0.name == "token" })?.value,
              let server = q.first(where: { $0.name == "server" })?.value, !server.isEmpty else {
            error = "That link is missing its invite. Open the newest link from your messages."
            return
        }
        if recorder.running { return }
        invite = Invite(token: token, server: server)
        error = nil
        // A newer link carries the server's current address; use it for an upload still waiting to go out.
        if var s = session, s.status == .ended, s.token == token {
            s.server = server
            SessionStore.save(s)
            session = s
            uploader.upload(s)
        } else if session?.status == .uploaded || session?.token != token {
            session = nil
        }
    }

    func requestTestInvite(server: String) async {
        let base = server.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        do {
            let j = try await post(base + "/api/invite", [:])
            guard let token = j["token"] as? String else { throw Fail("The server gave no token.") }
            invite = Invite(token: token, server: base)
        } catch {
            self.error = describe(error)
        }
    }

    func consent() async {
        guard let inv = invite else { return }
        let id = UUID().uuidString.lowercased()
        do {
            _ = try await post(inv.server + "/api/consent", [
                "token": inv.token, "session_id": id, "consent_version": consentVersion,
                "consented_at": Date().timeIntervalSince1970, "device_ios": UIDevice.current.systemVersion,
            ])
            let s = SessionState(id: id, token: inv.token, server: inv.server)
            SessionStore.save(s)
            session = s
            error = nil
        } catch {
            self.error = describe(error)
        }
    }

    func start() {
        guard var s = session else { return }
        s.placement = placement
        session = s
        recorder.start(s)
    }

    func end() async {
        busy = true
        defer { busy = false }
        guard let s = await recorder.stop() ?? session.map({ var x = $0; x.status = .ended; x.ended = x.ended ?? x.lastImuT ?? Date().timeIntervalSince1970; return x }) else { return }
        SessionStore.save(s)
        do {
            try SessionStore.writeMeta(s)
        } catch {
            self.error = "Could not write the session summary: \(error.localizedDescription)"
        }
        session = s
        uploader.upload(s)
    }

    func retryUpload() {
        if let s = session { uploader.upload(s) }
    }

    func newHike() {
        session = nil
        error = nil
    }

    /// Where the back arrow can take you: from the placement screen to consent, from consent to waiting for an invite.
    var canGoBack: Bool { !recorder.running && (stage == .setup || stage == .consent) }

    /// One step back. Leaving the placement screen drops the consented session (nothing is recorded yet; a fresh consent
    /// is sent if they return). Leaving consent forgets the invite, so the next link or the testing field sets a new one,
    /// which is also the way out when a stored server address has gone stale.
    func back() {
        guard canGoBack else { return }
        switch stage {
        case .setup:
            if let s = session { try? FileManager.default.removeItem(at: SessionStore.root.appendingPathComponent(s.id)) }
            session = nil
        case .consent:
            invite = nil
        default:
            break
        }
        error = nil
    }

    /// Gives up on a hike that can't finish uploading (the server was reset, or no longer knows it): deletes the
    /// recording from this phone and goes back to the start. The button asks for a long press first.
    func discard() {
        guard let s = session else { return }
        uploader.forget(s.id)
        try? FileManager.default.removeItem(at: SessionStore.root.appendingPathComponent(s.id))
        session = nil
        error = nil
    }

    private func post(_ url: String, _ body: [String: Any]) async throws -> [String: Any] {
        guard let u = URL(string: url) else { throw Fail("Bad server address: \(url)") }
        var r = URLRequest(url: u, timeoutInterval: 20)
        r.httpMethod = "POST"
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, resp) = try await URLSession.shared.data(for: r)
        let j = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else { throw Fail("Server said \(code): \(j["error"] as? String ?? "no details")") }
        return j
    }
}

struct Fail: Error, CustomStringConvertible {
    let description: String
    init(_ d: String) { description = d }
}

func describe(_ e: Error) -> String { (e as? Fail)?.description ?? e.localizedDescription }
