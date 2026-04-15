#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=================================================="
echo " 1/5: Installing Python Dependencies..."
echo "=================================================="
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Skipping pip install."
fi

echo ""
echo "=================================================="
echo " 2/5: Downloading wigxjpf header file..."
echo "=================================================="
wget -q --show-progress https://raw.githubusercontent.com/nd-nuclear-theory/wigxjpf/master/inc/wigxjpf.h -O wigxjpf.h

echo ""
echo "=================================================="
echo " 3/5: Cloning and compiling wigxjpf C-library..."
echo "=================================================="
# Remove old source folder if it exists from a previous failed run
rm -rf wigxjpf_source_temp
git clone https://github.com/nd-nuclear-theory/wigxjpf.git wigxjpf_source_temp

# Enter the cloned repo, compile it, and come back
cd wigxjpf_source_temp
make
cd ..

echo ""
echo "=================================================="
echo " 4/5: Moving library and cleaning up..."
echo "=================================================="
# Copy the compiled archive into the main project directory
cp wigxjpf_source_temp/lib/libwigxjpf.a ./

# Delete the downloaded source code to keep the folder clean
rm -rf wigxjpf_source_temp

echo "Library libwigxjpf.a successfully placed in project root."

echo ""
echo "=================================================="
echo " 5/5: Compiling Cython modules..."
echo "=================================================="
# Clean up any old broken builds
rm -rf build/
rm -f *.so

# Run the Python setup script
python setup.py build_ext --inplace

echo ""
echo "=================================================="
echo " SUCCESS: The S3 Topology pipeline is ready to run!"
echo "=================================================="