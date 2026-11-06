# Security Policy

## Supported versions

After the public release, security fixes are provided for the latest `v1.0.x`
release. Older tags and development snapshots are not supported.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository:

1. Open the repository on GitHub.
2. Go to **Security**.
3. Select **Report a vulnerability**.

If private reporting is not enabled, open a minimal public issue asking the
maintainer for a private channel. Do **not** include exploit details, credentials,
or personal data in that issue.

Please include:

- the affected version or commit;
- a description of the issue and its impact;
- reproduction steps or a minimal proof of concept;
- the environment used (OS, Python version, dependency versions).

Please do **not** include:

- real transaction data or personal data;
- API keys, passwords, tokens, or other credentials;
- datasets obtained from Kaggle or any other third party.

## Scope

This policy covers the source code, the Python dependencies declared in this
repository, the model artifacts, and the FastAPI service.

The FastAPI service is a **local research service**. It has no authentication,
rate limiting, or TLS. Do not expose it directly to the public internet; run it
behind an authenticated gateway if it must be reached remotely.

The model is a research artifact, not a security control. Fraud detection models
can be evaded, and model outputs are not a guarantee that a transaction is
legitimate. Do not use this model as the only fraud-prevention measure.

## Response

Reports are handled on a best-effort basis. No response or resolution SLA is
offered. Please allow reasonable time for a fix before any public disclosure.
