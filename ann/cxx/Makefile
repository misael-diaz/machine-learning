#!/usr/bin/make
#
# Algorithms and Complexity			           January 16, 2023
# IST 4310
#
# Synopsis:
# Defines the Makefile for building the project source files with GNU make.
#
# Copyright (c) 2023 Misael Diaz-Maldonado
# This file is released under the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#

include make-inc

all: $(TESTS)

$(MAIN_EXE): $(OBJECTS)
	$(CXX) $(CXXOPT) $(OBJECTS) -o $(MAIN_EXE)

$(MAIN_OBJ): $(MAIN_CXX)
	$(CXX) $(CXXOPT) -c $(MAIN_CXX) -o $(MAIN_OBJ)

clean:
	/bin/rm -rf $(OBJECTS) $(TESTS)
