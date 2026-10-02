# Current original-manuscript build from frozen evidence. No model reruns.
PY ?= python
HERE := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: all pdf figs verify verify-committed
all: pdf

pdf:
	cd $(HERE) && PY="$(PY)" bash scripts/rebuild_current.sh

figs:
	cd $(HERE) && $(PY) ../reproducibility/verify_method_contract.py
	cd $(HERE) && $(PY) scripts/build_field_revision.py
	cd $(HERE) && $(PY) verify_committed_claims.py --source-only

verify: verify-committed
	cd $(HERE) && $(PY) ../reproducibility/verify_method_contract.py
	cd $(HERE) && $(PY) scripts/check_revision_pdf.py
	cd $(HERE) && $(PY) scripts/bind_revision_contracts.py --check

verify-committed:
	cd $(HERE) && $(PY) verify_committed_claims.py

# Historical make_figs.R and submission snapshots are deliberately not targets.
# Do not clean tracked logs/bibliography or regenerate old-protocol figures here.
