# Security Policy

## Supported code

The `main` branch is the actively maintained public research code. Tagged
releases are reproducibility snapshots; security fixes may land on `main`
before a new tag is published.

## Credentials and data

Keep API keys, licensed market data, generated model artifacts, and local
research outputs outside Git. Use environment variables and ignored local
paths described in the reproducibility documentation. If a credential is ever
committed, revoke or rotate it before removing it from Git history.

Treat downloaded market data, serialized artifacts, and third-party model
files as untrusted input. Do not deserialize or execute artifacts from unknown
sources.

## Reporting

Use GitHub's private vulnerability reporting for security issues. Do not place
credentials, exploit payloads, or sensitive datasets in a public issue.
