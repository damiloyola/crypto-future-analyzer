# Reenvía cualquier comando al Makefile de la carpeta principal del proyecto,
# así `make cli SYMBOL=BTC`, `make setup`, etc. andan también desde acá.
# (Las variables como SYMBOL y ARGS pasan solas al make de arriba.)
# ANALYZER_INVOCATION_DIR: esta carpeta, para que una ruta relativa en ARGS
# (--db, --config) sea relativa a donde corriste make y no a la de arriba.
ANALYZER_INVOCATION_DIR ?= $(CURDIR)
export ANALYZER_INVOCATION_DIR

.PHONY: all
all:
	@$(MAKE) --no-print-directory -C .. help

.DEFAULT:
	@$(MAKE) --no-print-directory -C .. $@
