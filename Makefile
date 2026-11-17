# Convenience aliases. No package manager or privileged operations are used.
.DEFAULT_GOAL := help
.PHONY: help build clean-build clean check test verify format format-check install run
help:
	@printf '%s\n' 'make build       - release build' 'make clean-build - remove local artifacts and rebuild' 'make clean       - remove local artifacts only' 'make check       - check all Rust targets' 'make test        - module/integration/doc tests' 'make verify      - check/test/lint/build' 'make format      - explicitly run rustfmt' 'make format-check - check formatting without writing' 'make install     - verify and install for the current user' 'make run         - build and launch' 'Root .sh scripts also support --help and build options.'
build:
	./build.sh
clean-build:
	./clean-build.sh
clean:
	./clean.sh
check:
	./check.sh
test:
	./test.sh
verify:
	./verify.sh
format:
	./format.sh
format-check:
	./format.sh --check
install:
	./install.sh
run:
	./run.sh
