#!/usr/bin/make
#
# Machine Learning					June 16, 2023
#
# source: Makefile
# author: @misael-diaz
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

all: $(ALL)

$(MAIN_BIN): $(OBJECTS)
	$(CXX) $(CXXOPT) $(OBJECTS) -o $(MAIN_BIN)

$(LKNN_SO): $(OBJECTS)
	$(CXX) $(CXXOPT) $(OBJECTS) -o $(LKNN_SO) -shared

$(MAIN_OBJ): $(MAIN_CXX)
	$(CXX) $(CXXOPT) -c $(MAIN_CXX) -o $(MAIN_OBJ)

clean:
	/bin/rm -rf $(OBJECTS) $(TESTS)
