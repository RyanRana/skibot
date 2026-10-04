import SwiftUI

// The Hazard Intelligence look (web/site.css): white ground, Times for headlines, system sans for text,
// small mono uppercase labels, 1px rules, solid ink buttons.
private enum Ink {
    static let bg = Color.white
    static let panel = Color(red: 0.969, green: 0.969, blue: 0.961)   // #f7f7f5
    static let text = Color(red: 0.067, green: 0.075, blue: 0.082)    // #111315
    static let mute = Color(red: 0.373, green: 0.396, blue: 0.420)    // #5f656b
    static let faint = Color(red: 0.604, green: 0.624, blue: 0.647)   // #9a9fa5
    static let line = Color(red: 0.902, green: 0.906, blue: 0.910)    // #e6e7e8
    static let bad = Color(red: 0.702, green: 0.149, blue: 0.118)
    static let go = Color(red: 0.169, green: 0.541, blue: 0.243)      // #2b8a3e, the site's live dot
}

private func serif(_ size: CGFloat) -> Font { .custom("Times New Roman", size: size) }

struct RootView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var recorder: Recorder
    @EnvironmentObject var uploader: Uploader

    var body: some View {
        ZStack {
            Ink.bg.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    HStack(alignment: .firstTextBaseline) {
                        Text("Hi.").font(serif(34))
                        Spacer()
                        Kicker(text: "Ground Truth")
                    }
                    .padding(.bottom, 44)
                    switch model.stage {
                    case .noInvite: NoInviteView()
                    case .consent: ConsentView()
                    case .setup: SetupView()
                    case .recording: RecordingView()
                    case .interrupted: InterruptedView()
                    case .ended: EndedView()
                    }
                    if let e = model.error ?? recorder.error {
                        Problem(text: e).padding(.top, 20)
                    }
                }
                .padding(.horizontal, 24)
                .padding(.top, 12)
                .padding(.bottom, 40)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .foregroundStyle(Ink.text)
        .font(.system(size: 16))
        .preferredColorScheme(.light)
    }
}

// MARK: pieces

struct Hairline: View {
    var color: Color = Ink.line
    @Environment(\.displayScale) var scale
    var body: some View { Rectangle().fill(color).frame(height: 1 / scale) }
}

/// The site's small mono uppercase label (.k).
struct Kicker: View {
    let text: String
    var color: Color = Ink.faint
    var body: some View {
        Text(text.uppercased()).font(.system(size: 11, weight: .medium, design: .monospaced)).tracking(1.5).foregroundStyle(color)
    }
}

struct Title: View {
    let text: String
    var body: some View {
        Text(text).font(serif(40)).tracking(-0.8).lineSpacing(-4).fixedSize(horizontal: false, vertical: true).padding(.bottom, 16)
    }
}

struct Lede: View {
    let text: String
    var body: some View {
        Text(text).font(.system(size: 17)).foregroundStyle(Ink.mute).lineSpacing(3).fixedSize(horizontal: false, vertical: true)
    }
}

struct Problem: View {
    let text: String
    var body: some View {
        Text(text).font(.system(size: 14)).foregroundStyle(Ink.bad).fixedSize(horizontal: false, vertical: true)
    }
}

/// Solid ink button (primary) or a 1px outlined one.
struct InkButton: View {
    let title: String
    var primary = true
    var disabled = false
    let action: () -> Void
    @Environment(\.displayScale) var scale

    var body: some View {
        Button(action: action) {
            let solid = primary && !disabled  // disabled reads as the outlined style, never a grey slab
            Text(title).font(.system(size: 16, weight: .medium)).frame(maxWidth: .infinity).padding(.vertical, 16)
                .foregroundStyle(solid ? Ink.bg : disabled ? Ink.faint : Ink.text)
                .background(solid ? Ink.text : Ink.bg)
                .overlay(Rectangle().stroke(solid ? Ink.text : Ink.line, lineWidth: 1 / scale))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(disabled)
    }
}

/// A list in the site's style: an ink rule on top, hairlines between rows.
struct RuledList<Content: View>: View {
    @ViewBuilder let content: Content
    var body: some View {
        VStack(spacing: 0) {
            Hairline(color: Ink.text)
            content
        }
    }
}

/// A row: serif name on the left, muted detail on the right (the site's .links rows).
struct Row: View {
    let label: String
    let value: String
    var bad = false
    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .firstTextBaseline, spacing: 16) {
                Text(label).font(serif(20))
                Spacer(minLength: 8)
                Text(value).font(.system(size: 15)).monospacedDigit().multilineTextAlignment(.trailing)
                    .foregroundStyle(bad ? Ink.bad : Ink.mute)
            }
            .padding(.vertical, 15)
            Hairline()
        }
    }
}

/// A stat in the site's ledger: mono label, big serif number, muted note.
struct Stat: View {
    let label: String
    let value: String
    var note: String? = nil
    var bad = false
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Kicker(text: label)
            Text(value).font(serif(34)).monospacedDigit().foregroundStyle(bad ? Ink.bad : Ink.text)
                .lineLimit(1).minimumScaleFactor(0.6).padding(.top, 12)
            if let note {
                Text(note).font(.system(size: 13)).foregroundStyle(bad ? Ink.bad : Ink.mute).padding(.top, 6)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.vertical, 18)
    }
}

/// Two-column ledger with the rules the site draws between cells.
struct Ledger: View {
    let stats: [Stat]
    var body: some View {
        VStack(spacing: 0) {
            Hairline()
            ForEach(Array(stride(from: 0, to: stats.count, by: 2)), id: \.self) { i in
                HStack(alignment: .top, spacing: 0) {
                    stats[i].padding(.trailing, 16)
                    Rectangle().fill(Ink.line).frame(width: 1)
                    Group { if i + 1 < stats.count { stats[i + 1] } else { Color.clear } }.padding(.leading, 18)
                }
                .fixedSize(horizontal: false, vertical: true)
                Hairline()
            }
        }
    }
}

// MARK: screens

struct NoInviteView: View {
    @EnvironmentObject var model: AppModel
    @State private var server = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Title(text: "Waiting for your invite.")
            Lede(text: "Open the link we texted you. It brings you back here, ready to go.")
            Kicker(text: "Testing").padding(.top, 56).padding(.bottom, 12)
            TextField(text: $server, prompt: Text(verbatim: "https://name.trycloudflare.com").foregroundStyle(Ink.faint)) { EmptyView() }
                .textInputAutocapitalization(.never).autocorrectionDisabled().keyboardType(.URL)
                .font(.system(size: 16)).padding(12)
                .overlay(Rectangle().stroke(Ink.line, lineWidth: 1))
                .padding(.bottom, 10)
            InkButton(title: "Get a test invite", disabled: server.isEmpty) {
                Task { await model.requestTestInvite(server: server) }
            }
        }
    }
}

struct ConsentView: View {
    @EnvironmentObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Title(text: "Your hike can teach a rescue robot.")
            Lede(text: "Robots learn how people move on real ground, mud, scree, roots and switchbacks, from hikes like yours.")
            Kicker(text: "What we track").padding(.top, 40).padding(.bottom, 12)
            RuledList {
                Row(label: "Motion", value: "Accelerometer and gyroscope")
                Row(label: "Location", value: "GPS")
                Row(label: "Altitude", value: "Barometer")
                Row(label: "Steps", value: "Step count")
            }
            InkButton(title: "I'm in", disabled: model.busy) {
                Task { model.busy = true; await model.consent(); model.busy = false }
            }
            .padding(.top, 32)
        }
    }
}

struct SetupView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.displayScale) var scale
    private let spots = ["Pocket", "Hand", "Backpack", "Hip belt"]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Title(text: "Where will your phone be?")
            Lede(text: "It changes what the motion looks like, so it helps to know.")
            Kicker(text: "Placement").padding(.top, 40).padding(.bottom, 12)
            LazyVGrid(columns: [GridItem(.flexible(), spacing: 8), GridItem(.flexible(), spacing: 8)], spacing: 8) {
                ForEach(spots, id: \.self) { s in
                    let on = model.placement == s.lowercased()
                    Button { model.placement = s.lowercased() } label: {
                        Text(s).font(serif(20)).frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 14).padding(.vertical, 16)
                            .foregroundStyle(on ? Ink.text : Ink.faint)
                            .overlay(Rectangle().stroke(on ? Ink.text : Ink.line, lineWidth: on ? 1 : 1 / scale))
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
            InkButton(title: "Start hike") { model.start() }.padding(.top, 32)
            Text("iOS will ask for location and motion. Allow both. You can lock your phone once it's recording.")
                .font(.system(size: 13)).foregroundStyle(Ink.mute).padding(.top, 14).fixedSize(horizontal: false, vertical: true)
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
        let motion = s?.counts["imu"] ?? 0
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Circle().fill(Ink.go).frame(width: 8, height: 8)
                Kicker(text: "Recording", color: Ink.go)
            }
            Text(clock(now.timeIntervalSince1970 - started)).font(serif(84)).tracking(-1.5).monospacedDigit()
                .lineLimit(1).minimumScaleFactor(0.5).padding(.top, 10).padding(.bottom, 28)
            Ledger(stats: [
                Stat(label: "Distance", value: String(format: "%.2f", (s?.distance ?? 0) / 1000), note: "kilometres"),
                Stat(label: "GPS", value: recorder.hacc < 0 ? "—" : String(format: "± %.0f", recorder.hacc),
                     note: recorder.hacc < 0 ? "searching" : "metres", bad: recorder.hacc > 30),
                Stat(label: "Motion", value: motion.formatted(), note: "samples at 100 Hz", bad: motion == 0),
                Stat(label: "Gaps", value: "\(gaps.count)", note: gaps.isEmpty ? "none so far" : "\(Int(gapTime)) s missing",
                     bad: !gaps.isEmpty),
            ])
            ForEach(recorder.problems, id: \.self) { p in Problem(text: p).padding(.top, 10) }
            Lede(text: "Lock your phone and put it away. Recording keeps going. Don't swipe the app closed, that stops it.")
                .padding(.vertical, 28)
            HoldButton(title: "Hold to end hike") { Task { await model.end() } }
        }
    }
}

struct HoldButton: View {
    let title: String
    let action: () -> Void
    @State private var progress: CGFloat = 0

    var body: some View {
        GeometryReader { g in
            let label = Text(title).font(.system(size: 16, weight: .medium)).frame(width: g.size.width, height: g.size.height)
            ZStack(alignment: .leading) {
                label.foregroundStyle(Ink.text)
                Rectangle().fill(Ink.text).frame(width: g.size.width * progress)
                label.foregroundStyle(Ink.bg)  // white where the fill has reached
                    .mask(alignment: .leading) { Rectangle().frame(width: g.size.width * progress) }
            }
        }
        .frame(height: 52)
        .overlay(Rectangle().stroke(Ink.text, lineWidth: 1))
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
            Kicker(text: "Paused", color: Ink.bad).padding(.bottom, 14)
            Title(text: "Recording stopped.")
            Lede(text: last.map { "The app was closed or the phone restarted around \(time($0)). Everything up to then is saved." }
                 ?? "The app was closed before any motion was recorded.")
            InkButton(title: "Keep recording") { model.start() }.padding(.top, 32).padding(.bottom, 10)
            InkButton(title: "End hike", primary: false, disabled: model.busy) { Task { await model.end() } }
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
        let noMotion = (s?.counts["imu"] ?? 0) == 0
        VStack(alignment: .leading, spacing: 0) {
            Kicker(text: uploaded ? "Hike received" : "Uploading").padding(.bottom, 14)
            Title(text: uploaded ? "Thank you." : "Sending your hike.")
            Ledger(stats: [
                Stat(label: "Time", value: clock((s?.ended ?? 0) - (s?.started ?? 0))),
                Stat(label: "Distance", value: String(format: "%.2f", (s?.distance ?? 0) / 1000), note: "kilometres"),
                Stat(label: "Motion", value: (s?.counts["imu"] ?? 0).formatted(), note: "samples", bad: noMotion),
                Stat(label: "Gaps", value: noMotion ? "—" : "\(s?.gaps.count ?? 0)",
                     note: noMotion ? "no motion data" : nil, bad: noMotion || !(s?.gaps.isEmpty ?? true)),
            ])
            RuledList {
                Row(label: "Upload", value: uploaded ? "Done" : mine.isEmpty ? "Waiting" :
                    "\(max(uploader.total - mine.count, 0)) of \(uploader.total) files", bad: uploader.lastError != nil)
                if uploaded {
                    Row(label: "Server", value: status == "done" ? "Registered" : status == "failed" ? "Failed" : "Registering",
                        bad: status == "failed")
                }
            }
            .padding(.top, 32)
            if let e = uploader.lastError {
                Problem(text: "\(e). Retrying every 20 s.").padding(.top, 12)
                InkButton(title: "Retry now", primary: false) { model.retryUpload() }.padding(.top, 16)
            }
            if status == "failed", let e = srv?["error"] as? String {
                Problem(text: e).padding(.top, 12)
            }
            if let msg = srv?["message"] as? String {
                Kicker(text: "The text we sent").padding(.top, 32)
                Text(verbatim: msg).font(.system(size: 13, design: .monospaced)).foregroundStyle(Ink.mute).lineSpacing(3)
                    .padding(16).frame(maxWidth: .infinity, alignment: .leading).background(Ink.panel)
                    .padding(.top, 12).fixedSize(horizontal: false, vertical: true)
            }
            if uploaded {
                InkButton(title: "Record another hike") { model.newHike() }.padding(.top, 32)
            } else {
                Kicker(text: "Stuck on this hike?").padding(.top, 40).padding(.bottom, 10)
                Text("Discarding deletes this recording from your phone. It can't be undone.")
                    .font(.system(size: 13)).foregroundStyle(Ink.mute).padding(.bottom, 12)
                HoldButton(title: "Hold to discard this hike") { model.discard() }
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
    return f.string(from: Date(timeIntervalSince1970: unix))
}
