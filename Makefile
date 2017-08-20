STACK=stack
GHCID_SIZE ?= 8
# no name shadowing because not all abstractions are finished
GHCI_FLAGS ?= --ghci-options=-Wno-name-shadowing --ghci-options=-Wno-type-defaults

sos_warn:
	@command -v sos >/dev/null 2>&1 || { \
	   echo "-----------------------------------------------------------------" ; \
	   echo "        ! File watching functionality non-operational !          " ; \
	   echo "                                                                 " ; \
	   echo " Install steeloverseer to automatically run tasks on file change " ; \
	   echo "                                                                 " ; \
	   echo " See https://github.com/schell/steeloverseer                     " ; \
	   echo "-----------------------------------------------------------------" ; \
	}

ghci:
	$(STACK) ghci $(GHCI_FLAGS)

ghci-test:
	$(STACK) ghci $(GHCI_FLAGS) --test

ghcid:
	ghcid --height=$(GHCID_SIZE) --topmost "--command=$(MAKE) ghci"

ghcid-test:
	ghcid --height=$(GHCID_SIZE) --topmost "--command=$(MAKE) ghci-test"

hlint: sos_warn
	@command -v sos >/dev/null 2>&1 || { hlint .; exit 0; }
	@command -v hlint >/dev/null 2>&1 && { sos -p 'app/.*\.hs' -p 'src/.*\.hs' -c 'hlint \0'; } || echo "hlint not found on PATH"

codex: sos_warn
	@command -v sos   >/dev/null 2>&1 || { codex update --force; exit 0; }
	@command -v codex >/dev/null 2>&1 && { sos -p '.*\.hs' -c 'codex update --force'; } || echo "codex not found on PATH"

weeder: sos_warn
	@command -v sos >/dev/null 2>&1 || { weeder .; exit 0; }
	@command -v weeder >/dev/null 2>&1 && { sos -p '.*\.hs' -c 'weeder .'; } || echo "weeder not found on PATH"

ff:
	$(STACK) build --file-watch

ff-test:
	$(STACK) test --file-watch

.PHONY: codex hlint ghcid-test ghci-test ghcid ghci sos_warn ff ff-test weeder
