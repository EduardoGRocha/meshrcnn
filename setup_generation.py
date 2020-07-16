try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy

# RUN WITH: python setup_generation.py build_ext --inplace
# INSTALL: pip install trimesh

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'meshrcnn.utils.occ_net_utils.libkdtree.pykdtree.kdtree',
    sources=[
        'meshrcnn/utils/occ_net_utils/libkdtree/pykdtree/kdtree.c',
        'meshrcnn/utils/occ_net_utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
    include_dirs=[numpy_include_dir]
)

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'meshrcnn.utils.occ_net_utils.libmcubes.mcubes',
    sources=[
        'meshrcnn/utils/occ_net_utils/libmcubes/mcubes.pyx',
        'meshrcnn/utils/occ_net_utils/libmcubes/pywrapper.cpp',
        'meshrcnn/utils/occ_net_utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'meshrcnn.utils.occ_net_utils.libmesh.triangle_hash',
    sources=[
        'meshrcnn/utils/occ_net_utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir]
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'meshrcnn.utils.occ_net_utils.libmise.mise',
    sources=[
        'meshrcnn/utils/occ_net_utils/libmise/mise.pyx'
    ],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'meshrcnn.utils.occ_net_utils.libsimplify.simplify_mesh',
    sources=[
        'meshrcnn/utils/occ_net_utils/libsimplify/simplify_mesh.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'meshrcnn.utils.occ_net_utils.libvoxelize.voxelize',
    sources=[
        'meshrcnn/utils/occ_net_utils/libvoxelize/voxelize.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

'''
# DMC extensions
dmc_pred2mesh_module = CppExtension(
    'im2mesh.dmc.ops.cpp_modules.pred2mesh',
    sources=[
        'im2mesh/dmc/ops/cpp_modules/pred_to_mesh_.cpp',
    ]   
)

dmc_cuda_module = CUDAExtension(
    'im2mesh.dmc.ops._cuda_ext', 
    sources=[
        'im2mesh/dmc/ops/src/extension.cpp',
        'im2mesh/dmc/ops/src/curvature_constraint_kernel.cu',
        'im2mesh/dmc/ops/src/grid_pooling_kernel.cu',
        'im2mesh/dmc/ops/src/occupancy_to_topology_kernel.cu',
        'im2mesh/dmc/ops/src/occupancy_connectivity_kernel.cu',
        'im2mesh/dmc/ops/src/point_triangle_distance_kernel.cu',
    ]
)
'''

# Gather all extension modules
ext_modules = [
    pykdtree,
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
    #dmc_pred2mesh_module,
    #dmc_cuda_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)
