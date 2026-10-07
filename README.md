
# Installing and Compiling
For all operating systems, first make the build executable. In your terminal:

``chmod +x build_pipeline.sh``

Then you can compile by running:

``./build_pipeline.sh``

# Special Instructions for Mac Users
If you are running this code on a Mac, there are a couple of additional steps before compiling. This is because the default``C`` compiler ``clang`` is not compatible with modern Mac OS X distributions.

In your terminal, run:

```brew install llvm libomp```

Then in your ``.zshrc`` file, add the following lines:
```
export CC="/opt/homebrew/opt/llvm/bin/clang"
export CXX="/opt/homebrew/opt/llvm/bin/clang++"
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
```

Then you can install and compile the code as normal.

-------------

For basic examples of how to use the code, see the notebook ``Example_Notebook.ipynb``.

For an example script for computing multiple CMB matrices on a cluster, see ``compute_matrices.py``.

# Citations
If you use this code, please copy the citations in ``CITATION.bib``.
