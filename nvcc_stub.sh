#!/usr/bin/env bash
# Minimal stub to simulate nvcc presence and fail so build.rs emits placeholder PTX
# Print something to stderr to look like an error; exit nonzero
echo "nvcc stub: compilation disabled in this environment" 1>&2
exit 1
