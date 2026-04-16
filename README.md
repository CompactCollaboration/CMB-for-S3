Note: If you are running this code on a Mac, there are a couple of additional steps:

`brew install llvm libomp`

In your `.zshrc` file, add the following lines:
```
export CC="/opt/homebrew/opt/llvm/bin/clang"
export CXX="/opt/homebrew/opt/llvm/bin/clang++"
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
```

Then, for all operating systems, to set things up, first of all do

``chmod +x build_pipeline.sh``

Then 

``./build_pipeline.sh``

-------------

For an example of how to use the code, see the notebook ``Example_Notebook.ipynb``