"""Dashboards. Python on this machine can't verify HTTPS certificates on its own, so point it at certifi's."""

import os

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
