# Security Policy

## Supported versions

This is an alpha research package. Maintainers provide best-effort security
maintenance for the latest release and the default branch; older releases do
not have a separate long-term-support commitment. Use the tested locked uv or
native conda installation paths described in the installation guide.

## Reporting a vulnerability

Do not disclose vulnerabilities, credentials, or real datasets in public issues
or pull requests. If GitHub offers **Security → Report a vulnerability**, use
[private vulnerability reporting](https://github.com/pymc-labs/prior-generator/security/advisories/new).
Availability depends on repository settings, visibility, and the GitHub plan;
this policy does not imply that the feature is enabled.

If that option is unavailable, use the official
[PyMC Labs contact page](https://www.pymc-labs.com/contact) with a minimal,
non-sensitive summary identifying this repository and request a private
reporting channel. Do not send credentials or an exploit containing private data
through the initial contact form.

Once a private channel is established, include the affected version or commit,
installation method, a minimal synthetic reproducer, expected impact, and any
known mitigations. Maintainers will assess scope and coordinate remediation and
disclosure with the reporter. There is no guaranteed response or fix deadline.

## Trust boundaries

- Only load trusted package sources and release artifacts. Dependency and build
  installation executes code; checksum verification establishes file integrity,
  not trust in an unknown publisher.
- Corpus loading disables NumPy pickle deserialization and validates recognized
  schema metadata. It is not a sandbox: a compressed archive can still consume
  substantial memory or CPU. Bound resources before processing untrusted files.
- Documentation builds execute Python and notebooks. Do not run unreviewed
  documentation or pull-request code with credentials or access to sensitive data.
- Use synthetic, non-sensitive examples in reports. A synthetic generator does
  not make arbitrary input datasets anonymous or safe to publish.
- Dependency security changes require review and numerical verification; do not
  silently change the validated modeling stack or hide a resulting warning.
