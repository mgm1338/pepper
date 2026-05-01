#!/bin/bash
# Build the botserver binary for Linux and upload to S3.
# Run from the pepper repo root: ./scripts/build-and-upload.sh
set -euo pipefail

S3_PATH="s3://bidpepper-prod/deploy/botserver"

echo "Building botserver for Linux amd64..."
GOOS=linux GOARCH=amd64 go build -o /tmp/botserver-linux ./cmd/botserver/

echo "Uploading to $S3_PATH..."
aws s3 cp /tmp/botserver-linux "$S3_PATH"

rm /tmp/botserver-linux
echo "Done. Deploy pepper2 to pull the new binary."
