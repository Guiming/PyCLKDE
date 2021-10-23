# PyCLKDE
A big data-enabled high-performance computational framework for species habitat suitability modeling and mapping

## Compatable Operating Systems
Linux/Windows/Mac

## Computing environment:
1. Install GPU drivers. If you are using NVIDIA GPUs, OpenCL support is included in the driver (https://developer.nvidia.com/opencl). If you are using AMD GPU/CPU, install appropriate OpenCL drivers from AMD (This link provides helpful pointers https://github.com/microsoft/LightGBM/issues/1342). If you are using Intel GPU/CPU, install appropriate OpenCL drivers from Intel (e.g., https://software.intel.com/en-us/articles/opencl-drivers).    
2. It's assumed that you already have Python (version 2.7 or 3.x) installed. Anaconda is recommended for installing Python https://www.anaconda.com/distribution/.
3. Install Python GDAL (https://pypi.org/project/GDAL/).
4. Once OpenCL drivers, Python and Python GDAL are properly installed, you can use pip to install the PyOpenCL package (https://pypi.org/project/pyopencl/). 
5. Run pyopencl_test.py to test if PyOpenCL is working properly: python pyopencl_test.py.

## Run PyCLKDE with sample data:
1. Run pyopencl_test.py to see a list of available OpenCL computing platforms/devices on your computer: python pyopencl_test.py
2. Change configurations in the OPENCL_CONFIG variable in utility/config.py accordingly, as well as the OPENCL_CONFIG variable in PyCLKDE_main.py.
3. Run PyCLKDE_main.py to get a sense of how to use PyCLKDE (using example data provided): python PyCLKDE_main.py

## Use PyCLKDE for your own application:
1. Prepare species occurrence data and environmental covariate data following the example data in the "data" directory
2. Change parameters (e.g., data directory, data file names, etc.) in PyCLKDE_main.py
3. Run PyCLKDE_main.py: python PyCLKDE_main.py.py

## License
Copyright 2022 Guiming Zhang. Distributed under MIT license.

## Contact
guiming.zhang@du.edu
