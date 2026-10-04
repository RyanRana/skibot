import SwiftUI

private enum Ink {
    static let bg = Color(white: 0.97)
    static let text = Color(white: 0.1)
    static let mute = Color(white: 0.52)
    static let rule = Color(white: 0.84)
    static let bad = Color(red: 0.70, green: 0.15, blue: 0.12)
}

struct RootView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var recorder: Recorder
    @EnvironmentObject var uploader: Uploader

    var body: some View {
        ZStack {
            Ink.bg.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    Text("ground truth").font(.system(size: 15, weight: .light)).foregroundStyle(Ink.mute)
                        .padding(.bottom, 36)
                    switch model.stage {
                    case .noInvite: NoInviteView()
                    case .consent: ConsentView()
                    case .setup: SetupView()
                    case .recording: RecordingView()
                    case .interrupted: InterruptedView()
                    case .ended: EndedView()
                    }
                    if let e = model.error ?? recorder.error {
                        Text(e).foregroundStyle(Ink.bad).font(.system(size: 14, weight: .light)).padding(.top, 20)
                    }
                }
                .padding(.horizontal, 28)
                .padding(.vertical, 40)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .foregroundStyle(Ink.text)
        .font(.system(size: 17, weight: .light))
        .preferredColorScheme(.light)
    }
}

// MARK: pieces

struct Hairline: View {
    @Environment(\.displayScale) var scale
    var body: some View { Rectangle().fill(Ink.rule).frame(height: 1 / scale) }
}

struct LineButton: View {
    let title: String
    var disabled = false
    let action: () -> Void
    @Environment(\.displayScale) var scale

    var body: some View {
        Button(action: action) {
            Text(title).frame(maxWidth: .infinity).padding(.vertical, 15)
                .overlay(Rectangle().stroke(disabled ? Ink.rule : Ink.text, lineWidth: 1 / scale))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(disabled ? Ink.mute : Ink.text)
        .disabled(disabled)
    }
}

struct Row: View {
    let label: String
    let value: String
    var bad = false
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(label).foregroundStyle(Ink.mute)
                Spacer()
                Text(value).monospacedDigit().foregroundStyle(bad ? Ink.bad : Ink.text)
            }
            .padding(.vertical, 11)
            Hairline()
        }
        .font(.system(size: 15, weight: .light))
    }
}

struct Title: View {
    let text: String
    var body: some View {
        Text(text).font(.system(size: 32, weight: .light)).tracking(-0.4).padding(.bottom, 14)
    }
}

struct Note: View {
    let text: String
    var body: some View { Text(text).foregroundStyle(Ink.mute).fixedSize(horizontal: false, vertical: true) }
}

// MARK: screens

struct NoInviteView: View {
    @EnvironmentObject var model: AppModel
    @State private var server = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Title(text: "waiting for an invite")
            Note(text: "open the link we texted you. it brings you back here, ready to go.")
            Hairline().padding(.vertical, 36)
            Text("testing").font(.system(size: 13, weight: .light)).foregroundStyle(Ink.mute).padding(.bottom, 10)
            TextField("server, e.g. https://name.trycloudflare.com", text: $server)
                .textInputAutocapitalization(.never).autocorrectionDisabled().keyboardType(.URL)
                .font(.system(size: 15, weight: .light)).padding(.bottom, 8)
            Hairline().padding(.bottom, 16)
            LineButton(title: "get a test invite", disabled: server.isEmpty) {
                Task { await model.requestTestInvite(server: server) }
            }
        }
    }
}

struct ConsentView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Title(text: "turn this hike into robot training data")
            Note(text: "rescue robots need to learn how people move on real ground: mud, scree, roots, switchbacks. your hike can teach them.")
                .padding(.bottom, 28)
            Group {
                Row(label: "recorded", value: "motion, gps, altitude, steps")
                Row(label: "only while", value: "you're on a hike you started")
                Row(label: "never", value: "contacts, photos, audio")
                Row(label: "on maps", value: "first and last 200 m trimmed")
                Row(label: "battery", value: "about like a fitness app")
                Row(label: "delete", value: "anytime, just text us")
            }
            LineButton(title: "i'm in", disabled: model.busy) {
                Task { model.busy = true; await model.consent(); model.busy = false }
            }
            .padding(.top, 32)
        }
    }
}

struct SetupView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.displayScale) var scale
    private let spots = ["pocket", "hand", "backpack", "hip belt"]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Title(text: "where will your phone be?")
            Note(text: "it changes what the motion looks like, so it helps to know.").padding(.bottom, 24)
            HStack(spacing: 8) {
                ForEach(spots, id: \.self) { s in
                    Button { model.placement = s } label: {
                        Text(s).font(.system(size: 14, weight: .light)).frame(maxWidth: .infinity).padding(.vertical, 11)
                            .overlay(Rectangle().stroke(model.placement == s ? Ink.text : Ink.rule, lineWidth: 1 / scale))
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(model.placement == s ? Ink.text : Ink.mute)
                }
            }
            .padding(.bottom, 32)
            LineButton(title: "start hike") { model.start() }
            Note(text: "ios will ask for location and motion. allow both. you can lock your phone once it's recording.")
                .font(.system(size: 13, weight: .light)).padding(.top, 14)
        }
    }
}

struct RecordingView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var recorder: Recorder
    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { ctx in content(now: ctx.date) }
    }

    @ViewBuilder private func content(now: Date) -> some View {
        let s = recorder.live
        let started = s?.started ?? now.timeIntervalSince1970
        let gaps = s?.gaps ?? []
        let gapTime = gaps.reduce(0) { $0 + ($1[1] - $1[0]) }
        VStack(alignment: .leading, spacing: 0) {
            Text("recording").foregroundStyle(Ink.mute).font(.system(size: 14, weight: .light))
            Text(clock(now.timeIntervalSince1970 - started)).font(.system(size: 60, weight: .ultraLight)).monospacedDigit()
                .padding(.bottom, 24)
            Row(label: "distance", value: String(format: "%.2f km", (s?.distance ?? 0) / 1000))
            Row(label: "gps", value: recorder.hacc < 0 ? "searching" : String(format: "± %.0f m", recorder.hacc),
                bad: recorder.hacc > 30)
            Row(label: "motion samples", value: "\(s?.counts["imu"] ?? 0)", bad: (s?.counts["imu"] ?? 0) == 0)
            Row(label: "altitude samples", value: "\(s?.counts["baro"] ?? 0)")
            Row(label: "gaps", value: gaps.isEmpty ? "none" : "\(gaps.count), \(Int(gapTime)) s", bad: !gaps.isEmpty)
            ForEach(recorder.problems, id: \.self) { p in
                Text(p).foregroundStyle(Ink.bad).font(.system(size: 13, weight: .light)).padding(.top, 10)
            }
            Note(text: "lock your phone and put it away. recording keeps going. don't swipe the app closed, that stops it.")
                .font(.system(size: 14, weight: .light)).padding(.vertical, 28)
            HoldButton(title: "hold to end hike") { Task { await model.end() } }
        }
    }
}

struct HoldButton: View {
    let title: String
    let action: () -> Void
    @State private var progress: CGFloat = 0
    @Environment(\.displayScale) var scale

    var body: some View {
        Text(title).frame(maxWidth: .infinity).padding(.vertical, 15)
            .background(GeometryReader { g in
                Rectangle().fill(Ink.rule.opacity(0.7)).frame(width: g.size.width * progress)
            })
            .overlay(Rectangle().stroke(Ink.text, lineWidth: 1 / scale))
            .contentShape(Rectangle())
            .onLongPressGesture(minimumDuration: 1.5) {
                progress = 0
                action()
            } onPressingChanged: { pressing in
                withAnimation(pressing ? .linear(duration: 1.5) : .easeOut(duration: 0.3)) { progress = pressing ? 1 : 0 }
            }
            .accessibilityAddTraits(.isButton)
    }
}

struct InterruptedView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        let last = model.session?.lastImuT
        VStack(alignment: .leading, spacing: 0) {
            Title(text: "recording stopped")
            Note(text: last.map { "the app was closed or the phone restarted around \(time($0)). everything up to then is saved." }
                 ?? "the app was closed before any motion was recorded.")
                .padding(.bottom, 28)
            LineButton(title: "keep recording") { model.start() }.padding(.bottom, 12)
            LineButton(title: "end hike", disabled: model.busy) { Task { await model.end() } }
        }
    }
}

struct EndedView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var uploader: Uploader

    var body: some View {
        let s = model.session
        let mine = uploader.pending.filter { $0.hasPrefix((s?.id ?? "-") + "/") }
        let uploaded = s?.status == .uploaded || (SessionStore.load(s?.id ?? "")?.status == .uploaded)
        let srv = uploader.server
        let status = srv?["status"] as? String
        VStack(alignment: .leading, spacing: 0) {
            Title(text: uploaded ? "thank you" : "sending your hike")
            Row(label: "time", value: clock((s?.ended ?? 0) - (s?.started ?? 0)))
            Row(label: "distance", value: String(format: "%.2f km", (s?.distance ?? 0) / 1000))
            Row(label: "motion samples", value: "\(s?.counts["imu"] ?? 0)")
            let noMotion = (s?.counts["imu"] ?? 0) == 0
            Row(label: "gaps", value: noMotion ? "no motion data" : (s?.gaps.isEmpty ?? true) ? "none" : "\(s!.gaps.count)",
                bad: noMotion || !(s?.gaps.isEmpty ?? true))
            Row(label: "upload", value: uploaded ? "done" : mine.isEmpty ? "waiting" :
                "\(max(uploader.total - mine.count, 0)) of \(uploader.total) files",
                bad: uploader.lastError != nil)
            if uploaded {
                Row(label: "server", value: status == "done" ? "registered" : status == "failed" ? "failed" : "registering",
                    bad: status == "failed")
            }
            if let e = uploader.lastError {
                Text("\(e). retrying every 20 s.").foregroundStyle(Ink.bad).font(.system(size: 13, weight: .light)).padding(.top, 12)
                LineButton(title: "retry now") { model.retryUpload() }.padding(.top, 16)
            }
            if status == "failed", let e = srv?["error"] as? String {
                Text(e).foregroundStyle(Ink.bad).font(.system(size: 13, weight: .light)).padding(.top, 12)
            }
            if let msg = srv?["message"] as? String {
                Text(msg).font(.system(size: 15, weight: .light)).padding(.top, 24).fixedSize(horizontal: false, vertical: true)
            }
            if uploaded {
                LineButton(title: "record another hike") { model.newHike() }.padding(.top, 32)
            }
        }
        .onAppear { if uploaded, let s { uploader.poll(s) } }
    }
}

private func clock(_ secs: Double) -> String {
    let t = max(Int(secs), 0)
    return t >= 3600 ? String(format: "%d:%02d:%02d", t / 3600, t / 60 % 60, t % 60) : String(format: "%d:%02d", t / 60, t % 60)
}

private func time(_ unix: Double) -> String {
    let f = DateFormatter()
    f.dateFormat = "h:mm a"
    return f.string(from: Date(timeIntervalSince1970: unix)).lowercased()
}
