# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2/build

# Include any dependencies generated for this target.
include CMakeFiles/la2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/la2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/la2.dir/flags.make

CMakeFiles/la2.dir/la2.cpp.o: CMakeFiles/la2.dir/flags.make
CMakeFiles/la2.dir/la2.cpp.o: ../la2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/la2.dir/la2.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/la2.dir/la2.cpp.o -c /home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2/la2.cpp

CMakeFiles/la2.dir/la2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/la2.dir/la2.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2/la2.cpp > CMakeFiles/la2.dir/la2.cpp.i

CMakeFiles/la2.dir/la2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/la2.dir/la2.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2/la2.cpp -o CMakeFiles/la2.dir/la2.cpp.s

CMakeFiles/la2.dir/imc/MultilayerPerceptron.cpp.o: CMakeFiles/la2.dir/flags.make
CMakeFiles/la2.dir/imc/MultilayerPerceptron.cpp.o: ../imc/MultilayerPerceptron.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/la2.dir/imc/MultilayerPerceptron.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/la2.dir/imc/MultilayerPerceptron.cpp.o -c /home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2/imc/MultilayerPerceptron.cpp

CMakeFiles/la2.dir/imc/MultilayerPerceptron.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/la2.dir/imc/MultilayerPerceptron.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2/imc/MultilayerPerceptron.cpp > CMakeFiles/la2.dir/imc/MultilayerPerceptron.cpp.i

CMakeFiles/la2.dir/imc/MultilayerPerceptron.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/la2.dir/imc/MultilayerPerceptron.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2/imc/MultilayerPerceptron.cpp -o CMakeFiles/la2.dir/imc/MultilayerPerceptron.cpp.s

# Object files for target la2
la2_OBJECTS = \
"CMakeFiles/la2.dir/la2.cpp.o" \
"CMakeFiles/la2.dir/imc/MultilayerPerceptron.cpp.o"

# External object files for target la2
la2_EXTERNAL_OBJECTS =

la2: CMakeFiles/la2.dir/la2.cpp.o
la2: CMakeFiles/la2.dir/imc/MultilayerPerceptron.cpp.o
la2: CMakeFiles/la2.dir/build.make
la2: CMakeFiles/la2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable la2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/la2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/la2.dir/build: la2

.PHONY : CMakeFiles/la2.dir/build

CMakeFiles/la2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/la2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/la2.dir/clean

CMakeFiles/la2.dir/depend:
	cd /home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2 /home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2 /home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2/build /home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2/build /home/i82dianr/UCO/IMC/BackPropagationAlgorithm/la2/build/CMakeFiles/la2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/la2.dir/depend

