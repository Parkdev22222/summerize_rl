#!/bin/bash
# SessionStart hook: ensure the superpowers plugin marketplace is fetched and
# the plugin is installed so its skills are available in this session.
#
# Runs synchronously (no async) on purpose: plugin skills are registered at
# session start, so installation must finish before the agent loop begins.
set -euo pipefail

MARKETPLACE="superpowers-dev"
MARKETPLACE_SOURCE="obra/superpowers"
PLUGIN="superpowers@${MARKETPLACE}"

log() { echo "[session-start] $*" >&2; }

# Ensure the marketplace is registered/cloned (idempotent: tolerate "already added").
if ! claude plugin marketplace list 2>/dev/null | grep -q "${MARKETPLACE}"; then
  log "Adding marketplace ${MARKETPLACE_SOURCE}..."
  claude plugin marketplace add "${MARKETPLACE_SOURCE}" >&2 || \
    log "marketplace add reported an issue (may already be declared); continuing"
else
  log "Marketplace ${MARKETPLACE} already registered."
fi

# Install the plugin (idempotent: reinstall is a no-op / refresh).
if claude plugin list 2>/dev/null | grep -q "^\s*>*\s*${PLUGIN}"; then
  log "Plugin ${PLUGIN} already installed."
else
  log "Installing ${PLUGIN}..."
  claude plugin install "${PLUGIN}" >&2 || \
    log "plugin install reported an issue; continuing"
fi

log "Done."
