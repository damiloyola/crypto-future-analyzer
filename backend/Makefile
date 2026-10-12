# Reenvía cualquier comando al Makefile de la carpeta principal del proyecto,
# así `make cli SYMBOL=BTC`, `make setup`, etc. andan también desde acá.
# (Las variables como SYMBOL y ARGS pasan solas al make de arriba.)
.PHONY: all
all:
	@$(MAKE) --no-print-directory -C .. help

.DEFAULT:
	@$(MAKE) --no-print-directory -C .. $@
