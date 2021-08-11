from fipy.solvers.solver import Solver
from fipy.matrices.scipyMatrix import _ScipyMeshMatrix
from fipy.tools import numerix
import pyamgx
pyamgx.initialize()


class PyAMGXSolver(Solver):
    """
    The PyAMGXSolver class.
    """

    def __init__(self, config_dict, *args, **kwargs):
        """
        Parameters
        ----------
        config_dict : dict
            Dictionary specifying AMGX configuration options
        """
        self.config_dict = config_dict
        self.create()

    def create(self):
        self.cfg = pyamgx.Config().create_from_dict(self.config_dict)
        self.resources = pyamgx.Resources().create_simple(self.cfg)
        self.x_gpu = pyamgx.Vector().create(self.resources)
        self.b_gpu = pyamgx.Vector().create(self.resources)
        self.A_gpu = pyamgx.Matrix().create(self.resources)
        self.solver = pyamgx.Solver().create(self.resources, self.cfg)
        return self

    @property
    def _matrixClass(self):
        return _ScipyMeshMatrix

    def _storeMatrix(self, var, matrix, RHSvector):
        self.var = var
        self.matrix = matrix
        self.RHSvector = RHSvector

        self.A_gpu.upload_CSR(self.matrix.matrix)
        self.solver.setup(self.A_gpu)

    def _solve_(self, L, x, b):

        # transfer data from CPU to GPU
        self.x_gpu.upload(x)
        self.b_gpu.upload(b)

        # solve system on GPU
        self.solver.solve(self.b_gpu, self.x_gpu)

        # download values from GPU to CPU
        self.x_gpu.download(x)

    def _solve(self):
        self._solve_(self.matrix, self.var.ravel(), numerix.array(self.RHSvector))
            
    def _canSolveAsymmetric(self):
        return False

    def destroy(self):
        self.A_gpu.destroy()
        self.b_gpu.destroy()
        self.x_gpu.destroy()
        self.solver.destroy()
        self.resources.destroy()
        self.cfg.destroy()



cfg_dict = {
    "config_version": 2, 
    "determinism_flag": 1,
    "exception_handling" : 1,
    "solver": {
        "print_grid_stats": 1, 
        "store_res_history": 1, 
        "solver": "GMRES",   # BiCGStab  GMRES
        "print_solve_stats": 1, 
        "obtain_timings": 1, 
        "preconditioner": {
            "interpolator": "D2", 
            "print_grid_stats": 1, 
            "solver": "AMG", 
            "smoother": "JACOBI_L1", 
            "presweeps": 2, 
            "selector": "PMIS", 
            "coarsest_sweeps": 2, 
            "coarse_solver": "NOSOLVER", 
            "max_iters": 1, 
            "interp_max_elements": 4, 
            "min_coarse_rows": 2, 
            "scope": "amg_solver", 
            "max_levels": 24, 
            "cycle": "V", 
            "postsweeps": 2
        }, 
        "max_iters": 100, 
        "monitor_residual": 1, 
        "gmres_n_restart": 10, 
        "convergence": "RELATIVE_INI_CORE", 
        "tolerance": 1e-06, 
        "norm": "L2"
   }
}

cfg_pcgJ_dict ={
    "config_version": 2, 
    "solver": {
        "preconditioner": {
            "print_grid_stats": 1, 
            "print_vis_data": 0, 
            "solver": "AMG", 
            "smoother": {
                "scope": "jacobi", 
                "solver": "BLOCK_JACOBI", 
                "monitor_residual": 0, 
                "print_solve_stats": 0
            }, 
            "print_solve_stats": 0, 
            "presweeps": 1, 
            "max_iters": 1, 
            "interpolator": "D2",
            "monitor_residual": 0, 
            "store_res_history": 0, 
            "scope": "amg", 
            "max_levels": 100, 
            "cycle": "F", 
            "postsweeps": 1
        }, 
        "solver": "PCG", 
        "print_solve_stats": 1, 
        "obtain_timings": 1, 
        "max_iters": 100, 
        "monitor_residual": 1, 
        "convergence": "ABSOLUTE", 
        "scope": "main", 
        "tolerance": 1e-06, 
        "norm": "L2"
    }
}