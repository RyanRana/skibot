"""Train the G1 ski policy on an AWS GPU, the same job train/modal_train.py runs on Modal.

Ships the code and terrain to S3, starts one EC2 GPU instance on the AWS Deep Learning AMI, runs train.train there,
syncs checkpoints to S3 every minute, and powers the instance off when training ends (it is launched to terminate
on shutdown, so nothing is left running).

Launch:   python train/aws_train.py --bucket my-g1-runs --iters 1500 --num-envs 4096
Dry run:  python train/aws_train.py --bucket my-g1-runs --dry-run      # prints the boot script, touches nothing
Watch:    aws s3 ls s3://my-g1-runs/runs/ --recursive | tail
Pull:     aws s3 sync s3://my-g1-runs/runs/<run>/ runs/<run>/

Needs boto3 and AWS credentials with EC2, S3 and IAM pass-role rights. --instance-profile names an IAM instance
profile that can read and write the bucket (the instance uses it to pull the code and push checkpoints).
"""

from __future__ import annotations

import argparse
import io
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The same files the Modal image carries.
SHIP = ["skisim", "train", "assets/unitree_g1", "assets/sonic/model_encoder.onnx", "assets/sonic/model_decoder.onnx",
        "out/terrains", "out/terrains_v2"]
DLAMI = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"


def bundle() -> bytes:
  buf = io.BytesIO()
  with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for rel in SHIP:
      tar.add(ROOT / rel, arcname=rel, filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)
  return buf.getvalue()


def boot_script(bucket: str, key: str, run: str, iters: int, num_envs: int, resume: str, terrain: str) -> str:
  resume_arg = f"--resume /runs/{resume}" if resume else ""
  return f"""#!/bin/bash
set -euxo pipefail
exec > /var/log/g1-train.log 2>&1
trap 'aws s3 cp /var/log/g1-train.log s3://{bucket}/runs/{run}/boot.log || true; shutdown -h now' EXIT
mkdir -p /root/ski /runs && cd /root/ski
aws s3 cp s3://{bucket}/{key} code.tgz && tar xzf code.tgz
python3 -m venv /opt/venv && . /opt/venv/bin/activate
pip install -q mjlab==1.6.0 onnx==1.23.1 tensorboard
export PYTHONPATH=/root/ski PYTHONUNBUFFERED=1 GIT_PYTHON_REFRESH=quiet MUJOCO_GL=egl
{f"aws s3 sync s3://{bucket}/runs/{resume.split('/')[0]}/ /runs/{resume.split('/')[0]}/" if resume else ""}
( while true; do sleep 60; aws s3 sync /runs/ s3://{bucket}/runs/ --quiet || true; done ) &
python -m train.train --device cuda --num-envs {num_envs} --iters {iters} --log-dir /runs --terrain /root/ski/{terrain} {resume_arg}
aws s3 sync /runs/ s3://{bucket}/runs/
"""


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--bucket", required=True)
  ap.add_argument("--region", default="us-east-1")
  ap.add_argument("--instance-type", default="g6e.xlarge", help="one GPU; p5.4xlarge for a single H100 like Modal")
  ap.add_argument("--instance-profile", default="g1-train", help="IAM instance profile with access to the bucket")
  ap.add_argument("--iters", type=int, default=100000)
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--resume", default="", help="<run>/model_<n>.pt already in s3://bucket/runs/")
  ap.add_argument("--terrain", default="out/terrains")
  ap.add_argument("--spot", action="store_true", help="spot capacity: cheaper, can be reclaimed")
  ap.add_argument("--dry-run", action="store_true")
  a = ap.parse_args()

  run = time.strftime("aws-%Y%m%d-%H%M%S")
  key = f"code/{run}.tgz"
  script = boot_script(a.bucket, key, run, a.iters, a.num_envs, a.resume, a.terrain)
  if a.dry_run:
    print(f"would ship {', '.join(SHIP)} to s3://{a.bucket}/{key} and start a {a.instance_type} in {a.region}\n")
    print(script)
    return

  import boto3
  s3 = boto3.client("s3", region_name=a.region)
  data = bundle()
  s3.put_object(Bucket=a.bucket, Key=key, Body=data)
  print(f"shipped {len(data) / 1e6:.1f} MB to s3://{a.bucket}/{key}")

  ami = boto3.client("ssm", region_name=a.region).get_parameter(Name=DLAMI)["Parameter"]["Value"]
  ec2 = boto3.client("ec2", region_name=a.region)
  kw = dict(
    ImageId=ami, InstanceType=a.instance_type, MinCount=1, MaxCount=1, UserData=script,
    IamInstanceProfile={"Name": a.instance_profile}, InstanceInitiatedShutdownBehavior="terminate",
    BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 150, "VolumeType": "gp3"}}],
    TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": f"g1-ski-{run}"}]}],
  )
  if a.spot:
    kw["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time"}}
  inst = ec2.run_instances(**kw)["Instances"][0]["InstanceId"]
  print(f"started {inst} ({a.instance_type}); checkpoints -> s3://{a.bucket}/runs/  log -> s3://{a.bucket}/runs/{run}/boot.log")


if __name__ == "__main__":
  main()
