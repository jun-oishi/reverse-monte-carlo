
CPPC = g++-11
CPPFLAGS = -std=c++20 -Wall -O3 -march=native -I./include
LDFLAGS = -lstdc++ -lm
DST = bin

clean:
	rm -rf $(DST)/*

main: main.cpp include/*.hpp
	$(CPPC) $(CPPFLAGS) $< -o $(DST)/$@ $(LDFLAGS)