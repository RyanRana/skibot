import Foundation

/// Uploads a finished session with a background URLSession, so it completes even if the app is suspended.
/// Each file is a PUT to /api/session/<id>/file/<name>; meta.json lists every data file and its size, and the
/// server registers the session once all of them have arrived. Failures are kept on screen and retried.
final class Uploader: NSObject, ObservableObject, URLSessionDataDelegate {
    static let shared = Uploader()
    static let identifier = "com.ygadipalli.groundtruth.upload"
    static let retrySeconds: Double = 20

    @Published private(set) var pending: [String] = []   // "<session>/<file>"
    @Published private(set) var total = 0
    @Published private(set) var lastError: String?
    @Published private(set) var server: [String: Any]?    // GET /api/session/<id> once uploaded

    var backgroundCompletion: (() -> Void)?
    private var bodies: [Int: Data] = [:]
    private var polling: Timer?

    private lazy var session: URLSession = {
        let c = URLSessionConfiguration.background(withIdentifier: Self.identifier)
        c.sessionSendsLaunchEvents = true
        c.isDiscretionary = false
        return URLSession(configuration: c, delegate: self, delegateQueue: .main)
    }()

    override init() {
        super.init()
        session.getAllTasks { tasks in
            DispatchQueue.main.async { self.pending = tasks.compactMap(\.taskDescription) }
        }
    }

    func upload(_ s: SessionState) {
        lastError = nil
        server = nil
        session.getAllTasks { tasks in
            tasks.filter { $0.taskDescription?.hasPrefix(s.id + "/") == true }.forEach { $0.cancel() }
            DispatchQueue.main.async {
                let names = SessionStore.dataFiles(s.id).map(\.name) + ["meta.json"]
                self.total = names.count
                self.pending = names.map { "\(s.id)/\($0)" }
                names.forEach { self.put(s, $0) }
            }
        }
    }

    private func put(_ s: SessionState, _ name: String) {
        guard let url = URL(string: "\(s.server)/api/session/\(s.id)/file/\(name)") else {
            lastError = "bad server address: \(s.server)"
            return
        }
        var r = URLRequest(url: url)
        r.httpMethod = "PUT"
        r.setValue(s.token, forHTTPHeaderField: "X-Token")
        r.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        let task = session.uploadTask(with: r, fromFile: SessionStore.dir(s.id).appendingPathComponent(name))
        task.taskDescription = "\(s.id)/\(name)"
        task.resume()
    }

    func urlSession(_ s: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        bodies[dataTask.taskIdentifier, default: Data()].append(data)
    }

    func urlSession(_ s: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        let body = bodies.removeValue(forKey: task.taskIdentifier) ?? Data()
        guard let desc = task.taskDescription else { return }
        let parts = desc.split(separator: "/", maxSplits: 1).map(String.init)
        let code = (task.response as? HTTPURLResponse)?.statusCode ?? 0
        if (error as? URLError)?.code == .cancelled { return }
        if error == nil, (200..<300).contains(code) {
            pending.removeAll { $0 == desc }
            if !pending.contains(where: { $0.hasPrefix(parts[0] + "/") }), var st = SessionStore.load(parts[0]) {
                st.status = .uploaded
                SessionStore.save(st)
                lastError = nil
                poll(st)
            }
            return
        }
        let serverMsg = (try? JSONSerialization.jsonObject(with: body) as? [String: Any])?["error"] as? String
        lastError = "\(parts[1]): " + (error.map { e in let n = e as NSError; return "\(n.localizedDescription.lowercased()) (\(n.domain) \(n.code))" }
            ?? "server said \(code) \(serverMsg ?? "")")
        DispatchQueue.main.asyncAfter(deadline: .now() + Self.retrySeconds) {
            guard self.pending.contains(desc), let st = SessionStore.load(parts[0]) else { return }
            self.put(st, parts[1])
        }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession s: URLSession) {
        backgroundCompletion?()
        backgroundCompletion = nil
    }

    /// Watches the server register the session (distance, trail match) so the done screen can show the result.
    func poll(_ s: SessionState) {
        polling?.invalidate()
        var tries = 0
        polling = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] t in
            tries += 1
            guard let url = URL(string: "\(s.server)/api/session/\(s.id)") else { return t.invalidate() }
            URLSession.shared.dataTask(with: url) { data, _, _ in
                guard let data, let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
                DispatchQueue.main.async {
                    self?.server = j
                    let st = j["status"] as? String
                    if st == "done" || st == "failed" || tries > 90 { t.invalidate() }
                }
            }.resume()
        }
    }
}
