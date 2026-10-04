# Phone test: ground truth recorder

About 45 minutes, plus 15 to set up. You need a Mac with Xcode, an iPhone (iOS 17 or newer) and a cable. You'll do eight short recordings. Each one checks something we can't check in the simulator.

## What you're testing

The ground truth app records how a person moves on a hike:

- motion 100 times a second
- GPS
- altitude
- steps

It keeps recording with the phone locked in a pocket. When the hike ends, it uploads everything to our server. The questions are:

1. Does it really keep recording with the screen locked?
2. Are the sensor numbers right?
3. Does the upload work?

## Setup (once)

1. **Get the code.** Ask Yashu for the repo and run:

   ```
   brew install xcodegen
   ```

2. **Sign in to Xcode.** In Xcode, open Settings → Accounts and add your Apple ID. A free account works.

3. **Download the iOS platform.** In Xcode, open Settings → Components and download the iOS platform. Without it, Xcode says "iOS 26.5 is not installed".

4. **Turn on Developer Mode on the iPhone.** Go to Settings → Privacy & Security → Developer Mode, turn it on, and restart the phone when asked.

5. **Connect the phone.** Plug it into the Mac and tap Trust on the phone.

6. **Build and install the app.** In Terminal, from the repo folder:

   ```
   cd ios/GroundTruth && ./setup.sh && open GroundTruth.xcodeproj
   ```

   In Xcode, choose your iPhone at the top and press ▶. If it says the bundle ID is taken, run `GT_BUNDLE_ID=com.<yourname>.groundtruth ./setup.sh` instead, then press ▶ again.

7. **Trust the developer on the phone.** Settings → General → VPN & Device Management → your Apple ID → Trust. Then open the app.

8. **Connect the app to the server.** Ask Yashu for the server address. It looks like `https://something.trycloudflare.com`.

   - In the app, under "testing", paste the address and tap "get a test invite".
   - Tap "i'm in".
   - When iOS asks for location and motion, choose **Allow While Using App** and **Allow**.
   - If you denied either one, the app shows red text that says where to turn it back on.

## How every test works

Each test is one recording:

1. Choose where the phone will be.
2. Tap **start hike** and do the test.
3. **Press and hold "hold to end hike"** for about 2 seconds.
4. Wait until the upload row says **done** and the server row says **registered**.
5. Tap **record another hike** for the next test.

Write down the clock time when you start each test. Yashu matches recordings to tests by start time.

Screenshot anything in red.

## The tests

| # | Test | Phone placement | What to do | Why |
|---|---|---|---|---|
| 1 | Smoke | hand | Record 1 minute, screen on, walking around. | Proves that recording, upload and the server all work before anything else. |
| 2 | Rest | hand | Put the phone **flat on a table**, tap start, then don't touch the table for **60 s**. End. | Measures sensor noise, bias and drift. |
| 3 | Steps | pocket | Start, put the phone in a front jeans pocket, walk **exactly 100 steps** (count them), take it out, end. | Checks the step count and step timing against your count. |
| 4 | Rotate | hand | Phone flat on a table. Start, wait 5 s, then **spin it 3 full turns in one direction** (about 10 s), wait 5 s, end. | Checks the gyroscope angle against a known 1080°. |
| 5 | **Locked walk (the big one)** | pocket | Note the **battery %**. Start, **lock the phone**, pocket it, walk **15 minutes** outside. Unlock and screenshot the recording screen *before* ending, then end. Note the battery % again. | The whole product depends on this. "motion samples" should be about 90,000 and **gaps should say none**. |
| 6 | Swipe-away | hand | Start, walk 30 s, **swipe the app away** in the app switcher. Wait 20 s, reopen. It should say "recording stopped" and offer "keep recording". Tap it, walk 30 s, end. | Proves a killed app is reported as a gap, not hidden. |
| 7 | Known distance | pocket | Walk a route of known length, ideally **4 laps of a 400 m track** (1600 m). End. Tell Yashu the distance, and the climb in metres if the route has hills. | Checks GPS distance and barometer climb. |
| 8 | Upload on cellular | pocket | Record 1 minute. Before ending, **turn Wi-Fi off**. End and confirm the upload finishes on cellular. | Real hikes end far from Wi-Fi. |

## Send back to Yashu

- The start time of each test (1–8).
- Test 5: the screenshot before ending, battery % before and after, and gap count.
- Test 6: a screenshot of the "recording stopped" screen.
- Test 7: the route length and climb.
- Screenshots of any red text, and anything that felt confusing.

## For Yashu: checking the numbers

Run these on the Mac that runs the server:

```
python -m ground.validate list                                  # match ids to the tester's start times
python -m ground.validate rest <id>
python -m ground.validate steps <id> --count 100
python -m ground.validate rotate <id> --turns 3
python -m ground.validate distance <id> --meters 1600 [--climb-m N]
python -m ground.validate report                                # all results as one table
```

`report` writes `out/ground/validation_report.md`, which is the answer to "how did you verify the sensors?"

Each check prints:

- the measured value
- our target
- pass or fail

The targets come from typical phone sensor performance, not Apple specs. A failure tells us what to look at; it doesn't automatically mean the data is unusable.

The app records Apple's fused motion data. Gravity is computed by Apple and is about 1 g by design, so the script doesn't test "gravity = 9.81". It tests noise, bias, drift, steps, angle, distance and climb.
