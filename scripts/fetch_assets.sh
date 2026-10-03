#!/usr/bin/env bash
# Downloads the third-party files this repo does not store.
# NVIDIA GEAR-SONIC weights are under the NVIDIA Open Model License (see assets/sonic/LICENSE after download).
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p assets/sonic
base="https://huggingface.co/nvidia/GEAR-SONIC/resolve/main"
for f in model_encoder.onnx model_decoder.onnx; do
  echo "fetching $f"
  curl -fL --retry 3 -o "assets/sonic/$f" "$base/$f"
done
for f in config.json observation_config.yaml LICENSE README.md; do
  curl -fsL --retry 3 -o "assets/sonic/$f" "$base/$f" || echo "optional file $f not found, skipped"
done

# Optional: GoPro IMU/GPS ski runs on Marmolada (CC BY 4.0, Prochazka, doi:10.5281/zenodo.21777004)
if [[ "${WITH_ZENODO:-0}" == "1" ]]; then
  mkdir -p data/zenodo
  for f in GX010060ap.csv GX010061hc.csv; do
    curl -fL --retry 3 -o "data/zenodo/$f" "https://zenodo.org/records/21777004/files/$f?download=1"
  done
fi
echo "done"
