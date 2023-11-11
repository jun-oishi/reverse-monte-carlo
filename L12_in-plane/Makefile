
CPPC = g++-11
CPPFLAGS = -std=c++20 -Wall -march=native -I./include -I/usr/include/eigen3
ASPYLIB = -shared -fPIC $(shell python3 -m pybind11 --includes)
PYLIBEXT = $(shell python3.11-config --extension-suffix)
LDFLAGS = -lstdc++ -lm
SRC = src
PYLIB = lib
BIN = bin
INCLUDE = include


clean:
	rm -rf $(PYLIB)/* $(BIN)/*

test: $(SRC)/test.cpp $(INCLUDE)/*.hpp
	$(CPPC) $(CPPFLAGS) $< -o $(BIN)/$@ $(LDFLAGS)
	$(BIN)/$@

compute: $(SRC)/compute.cpp $(INCLUDE)/compute.hpp
	$(CPPC) $(CPPFLAGS) $(ASPYLIB) $< -o $(PYLIB)/$@$(PYLIBEXT) $(LDFLAGS)