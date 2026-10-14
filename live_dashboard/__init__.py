"""PyMC Generator live dashboard: publish generated worlds and follow them in a browser.

A generation worker publishes immutable task outputs (a corpus payload plus a versioned
JSON completion record) into a store; a separate web app polls the store, verifies and
extracts each newly published output once, and serves score cards and figures behind
HTTP Basic authentication. See ``live_dashboard/README.md`` for the publication
protocol, local use and the Modal deployment.
"""
