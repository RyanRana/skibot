#!/bin/zsh
# Writes Local.xcconfig (your signing team, kept out of git) and generates GroundTruth.xcodeproj.
#   ./setup.sh               uses the first team signed into Xcode, or none
#   GT_TEAM=ABCDE12345 ./setup.sh
set -e
cd "$(dirname "$0")"
TEAM="${GT_TEAM:-$(defaults read com.apple.dt.Xcode IDEProvisioningTeamByIdentifier 2>/dev/null | sed -n 's/.*teamID = \([A-Z0-9]*\);.*/\1/p' | head -1)}"
BUNDLE="${GT_BUNDLE_ID:-com.ygadipalli.groundtruth}"
printf 'DEVELOPMENT_TEAM = %s\nGT_BUNDLE_ID = %s\n' "$TEAM" "$BUNDLE" > Local.xcconfig
xcodegen generate --quiet
if [[ -z "$TEAM" ]]; then
  echo "no signing team found: Xcode > Settings > Accounts > add your Apple ID, then run ./setup.sh again"
else
  echo "team $TEAM, bundle $BUNDLE. open GroundTruth.xcodeproj, pick your iPhone, press run"
fi
