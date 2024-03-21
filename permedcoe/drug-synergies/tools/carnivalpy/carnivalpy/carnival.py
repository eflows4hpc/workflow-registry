from abc import ABC, abstractmethod
import picos as pc
import mip
import argparse
import pandas as pd
import os

class Problem(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.p = self.create_problem()
        self.setup()

    @abstractmethod
    def create_problem(self):
        pass

    @abstractmethod
    def add_constraint(self, affine_expression):
        pass

    @abstractmethod
    def create_binary(self, name: str):
        pass

    @abstractmethod
    def create_integer(self, name, lower: int=0, upper: int=1):
        pass

    @abstractmethod
    def setup(**kwargs):
        pass

    @abstractmethod
    def set_objective(self, expression):
        pass

    @abstractmethod
    def solve(self, **kwargs):
        pass

    @abstractmethod
    def export(self, file):
        pass

    @staticmethod
    def _edgename(row):
        return row.source + "_" + row.target

    def build(self, graph, measurements, perturbations, penalty=0.0001):
        # Direct translation of the V2 CARNIVAL ILP implementation
        # TODO: Support Inverse CARNIVAL (add all perturbations as parents, with both signs)
        # See https://github.com/saezlab/CARNIVAL/blob/963fbc1db2d038bfeab76abe792416908327c176/R/create_lp_formulation.R
        # Perturbations are included in C8 and C9
        vn = dict()
        # For all unique nodes in the graph
        for node in set(graph.source) | set(graph.target):
            vn[f'nU_{node}'] = self.create_binary(f'nU_{node}')
            vn[f'nD_{node}'] = self.create_binary(f'nD_{node}')
            vn[f'nX_{node}'] = self.create_integer(f'nX_{node}', lower=-1, upper=1)
            vn[f'nAc_{node}'] = self.create_integer(f'nAc_{node}', lower=-1, upper=1)
            vn[f'nDs_{node}'] = self.create_integer(f'nDs_{node}', lower=0, upper=100)
            # Add also C8 here (for all nodes)
            # https://github.com/saezlab/CARNIVAL/blob/963fbc1db2d038bfeab76abe792416908327c176/R/constraints_create_constraints_8.R#L8
            self.add_constraint(vn[f'nU_{node}'] - vn[f'nD_{node}'] + vn[f'nAc_{node}'] - vn[f'nX_{node}'] == 0)
            
        # Create variables for the measurements (to measure a mismatch which goes from 0 to 2)
        for k, v in measurements.items():
            vn[f'aD_{k}'] = self.create_integer(f'aD_{k}', lower=0, upper=2)

        # Create the variables for the edges
        for row in graph.itertuples():
            edge_name = Problem._edgename(row)
            eu = f"eU_{edge_name}"
            ed = f"eD_{edge_name}"
            if eu in vn or ed in vn:
                print(f"[WARNING] Multiple edges between {edge_name}, ignoring...")
            else:
                vn[eu] = self.create_binary(eu)
                vn[ed] = self.create_binary(ed)
                # Constraint C3
                self.add_constraint(vn[eu] + vn[ed] <= 1)

        # Add constraints for positive and negative measurements (for the objective function)
        for k, v in measurements.items():
            if v > 0:
                self.add_constraint(vn[f'nX_{k}'] - vn[f'aD_{k}'] <= 1)
                self.add_constraint(vn[f'nX_{k}'] + vn[f'aD_{k}'] >= 1)
            else:
                self.add_constraint(vn[f'nX_{k}'] - vn[f'aD_{k}'] <= -1)
                self.add_constraint(vn[f'nX_{k}'] + vn[f'aD_{k}'] >= -1)
                
        # C1 and C2
        for row in graph.itertuples():
            # Add constraints for activatory/inhibitory edges edges
            # Remove this condition just by multiplying by the interaction sign
            if row.interaction > 0:
                # C1 and C2
                self.add_constraint(vn[f'eU_{Problem._edgename(row)}'] - vn[f'nX_{row.source}'] >= 0)
                self.add_constraint(vn[f'eD_{Problem._edgename(row)}'] + vn[f'nX_{row.source}'] >= 0)
                # C3 and C4
                self.add_constraint(vn[f'eU_{Problem._edgename(row)}'] - vn[f'nX_{row.source}'] - vn[f'eD_{Problem._edgename(row)}'] <= 0)
                self.add_constraint(vn[f'eD_{Problem._edgename(row)}'] + vn[f'nX_{row.source}'] - vn[f'eU_{Problem._edgename(row)}'] <= 0)
            else:
                # C1 and C2
                self.add_constraint(vn[f'eU_{Problem._edgename(row)}'] + vn[f'nX_{row.source}'] >= 0)
                self.add_constraint(vn[f'eD_{Problem._edgename(row)}'] - vn[f'nX_{row.source}'] >= 0)
                # C3 and C4
                self.add_constraint(vn[f'eU_{Problem._edgename(row)}'] + vn[f'nX_{row.source}'] - vn[f'eD_{Problem._edgename(row)}'] <= 0)
                self.add_constraint(vn[f'eD_{Problem._edgename(row)}'] - vn[f'nX_{row.source}'] - vn[f'eU_{Problem._edgename(row)}'] <= 0)
            # Add constraints for loops
            # Re-design the constraints in the future
            self.add_constraint(101 * vn[f'eU_{Problem._edgename(row)}'] + vn[f'nDs_{row.source}'] - vn[f'nDs_{row.target}'] <= 100)
            self.add_constraint(101 * vn[f'eD_{Problem._edgename(row)}'] + vn[f'nDs_{row.source}'] - vn[f'nDs_{row.target}'] <= 100)
                
        # C6 and C7 for incoming edges
        for target in graph.target.unique():
            eU = [vn[f'eU_{Problem._edgename(row)}'] for row in graph[graph.target==target].itertuples()]
            if len(eU) > 0:
                # C6
                self.add_constraint(vn[f'nU_{target}'] <= sum(eU))
            eD = [vn[f'eD_{Problem._edgename(row)}'] for row in graph[graph.target==target].itertuples()]
            if len(eD) > 0:
                # C7
                self.add_constraint(vn[f'nD_{target}'] <= sum(eD))
            
        # Add constraints for perturbations
        # https://github.com/saezlab/CARNIVAL/blob/963fbc1db2d038bfeab76abe792416908327c176/R/create_lp_formulation.R
        for node, v in perturbations.items():
            self.add_constraint(vn[f'nU_{node}'] <= 0)
            self.add_constraint(vn[f'nD_{node}'] <= 0)
            # TODO: This can be 1 or -1 depending on the perturbation value
            self.add_constraint(vn[f'nX_{node}'] == v)
            # C8-parents, this is not only for perturbations! removed from here (TODO: validate)
            # self.add_constraint(vn[f'nX_{node}'] - vn[f'nAc_{node}'] == 0)

        # C8-parents (all nodes in the graph that are not targets)
        # See https://github.com/saezlab/CARNIVAL/blob/963fbc1db2d038bfeab76abe792416908327c176/R/constraints_create_constraints_8.R#L33
        parent_nodes = set(graph.source) - set(graph.target)
        for node in parent_nodes:
            self.add_constraint(vn[f'nX_{node}'] - vn[f'nAc_{node}'] == 0)    
            
        # C8 for unperturbed nodes
        unperturbed = (set(graph.source) | set(graph.target)) - set(perturbations.keys())
        for node in unperturbed:
            self.add_constraint(vn[f'nAc_{node}'] == 0)

        # Add objective function
        # Minimize the discrepancies of the measurements (aD vars)
        # Use a penalty on the number of active nodes
        # TODO: Change to a matrix form to accelerate PICOS processing
        obj1 = sum([abs(float(v))*vn[f'aD_{k}'] for k, v in measurements.items()])
        if penalty > 0:
            obj2u = penalty * sum([vn[f'nU_{node}'] for node in set(graph.source) | set(graph.target)])
            obj2d = penalty * sum([vn[f'nD_{node}'] for node in set(graph.source) | set(graph.target)])
            self.set_objective(sum([obj1, obj2u, obj2d]))
        else:
            self.set_objective(obj1)
        self._problem_vars = vn


class PicosProblem(Problem):
    def __init__(self) -> None:
        super().__init__()

    def create_problem(self):
        return pc.Problem()

    def add_constraint(self, affine_expression):
        self.p.add_constraint(affine_expression)

    def create_binary(self, name: str):
        return pc.BinaryVariable(name)

    def create_integer(self, name, lower: int=0, upper: int=1):
        return pc.IntegerVariable(name, lower=lower, upper=upper)

    def setup(self, **kwargs):
        if "verbosity" in kwargs:
            self.p.options["verbosity"] = kwargs["verbosity"]
        else:
            self.p.options["verbosity"] = 1
        if "max_seconds" in kwargs:
            self.p.options["timelimit"] = kwargs["max_seconds"]
        else:
            self.p.options["timelimit"] = None
        if "opt_tol" in kwargs:
            self.p.options["rel_bnb_opt_tol"] = kwargs["opt_tol"]
        if "feas_tol" in kwargs:
            self.p.options["abs_prim_fsb_tol"] = kwargs["feas_tol"]
            self.p.options["abs_dual_fsb_tol"] = kwargs["feas_tol"]
        if "int_tol" in kwargs:
            self.p.options["integrality_tol"] = kwargs["int_tol"]

    def set_objective(self, expression):
        self.p.set_objective('min', expression)

    def solve(self, **kwargs):
        self.p.solve(**kwargs)

    def export(self, file):
        l = list(zip(*list(map(lambda x: (x[0], x[1].value), self._problem_vars.items()))))
        pd.DataFrame({'variable': l[0], 'value': l[1]}).to_csv(file, index=False)

    
class MIPProblem(Problem):
    def __init__(self, solver=mip.CBC):
        self.solver_name = solver
        super().__init__()
        

    def create_problem(self):
        # Use Python-MIP only for the integrated CBC solver
        return  mip.Model("CARNIVAL-PYMIP", solver_name=self.solver_name)

    def add_constraint(self, affine_expression):
        self.p.add_constr(affine_expression)

    def create_binary(self, name: str):
        return self.p.add_var(var_type=mip.BINARY, name=name)

    def create_integer(self, name, lower: int=0, upper: int=1):
        return self.p.add_var(var_type=mip.INTEGER, name=name, lb=lower, ub=upper)
    
    def setup(self, **kwargs):
        self.p.threads = -1
        if "verbosity" in kwargs:
            self.p.verbose = kwargs["verbosity"]
        else:
            self.p.verbose = 1
        if "max_seconds" in kwargs:
            self.p.max_seconds = kwargs["max_seconds"]
        else:
            self.p.max_seconds = 1e9
        if "opt_tol" in kwargs:
            self.p.max_mip_gap = kwargs["opt_tol"]
            self.p.max_gap = kwargs["opt_tol"]
        if "int_tol" in kwargs:
            self.p.integer_tol = kwargs["int_tol"]
        if "feas_tol" in kwargs:
            self.p.infeas_tol = kwargs["feas_tol"]

    def set_objective(self, expression):
        self.p.objective = expression
        self.p.sense = mip.MINIMIZE

    def solve(self, **kwargs):
        return self.p.optimize(**kwargs)

    def export(self, file):
        # Export name of the variable and value
        l = list(zip(*list(map(lambda x: (x[0], x[1].x), self._problem_vars.items()))))
        pd.DataFrame({'variable': l[0], 'value': l[1]}).to_csv(file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CARNIVAL-PY")
    parser.add_argument('folder', type=str, help="Path to the folder with the input files: network.csv (Nx3) in the format 'source,interaction,target', \
                                                                 measurements.csv (Mx2) in the format 'id, value', \
                                                                 and perturbations.csv (Px2) in the format 'id, value'")
    parser.add_argument('--penalty', type=float, default=1e-2, help="Regularization penalty for the nodes (default 1e-2)")
    parser.add_argument('--solver', type=str, default='cbc', help="Solver name (cbc, cplex, gurobi, scip, glpk, mosek), default 'cbc'")
    parser.add_argument('--tol', type=float, default=0.01, help="MIP Gap tolerance")
    parser.add_argument('--maxtime', type=int, default=600, help="Max time in seconds")
    parser.add_argument('--export', type=str, default=None, help="Path to the file to be exported with the solution (default solution.csv)")
    args = parser.parse_args()

    export_file = args.export if args.export else f"{args.folder}/solution.csv"
    graph = pd.read_csv(os.path.join(args.folder, 'network.csv'), index_col=False)
    measurements = pd.read_csv(os.path.join(args.folder, 'measurements.csv')).set_index('id')
    # Remove measurements not in the graph
    unodes = set(graph.source).union(graph.target)
    print(f"There are {len(unodes)} unique species in the PKN")
    common_species = measurements.index.intersection(list(unodes))
    print(f"{len(common_species)} measurements included in the PKN ({measurements.shape[0] - len(common_species)} not included in the PKN and discarded)")
    measurements = measurements.loc[common_species].value.to_dict()

    # Check if perturbations is provided:
    pert_file = os.path.join(args.folder, 'perturbations.csv')
    if os.path.exists(pert_file):
        perturbations = pd.read_csv(pert_file).set_index('id').value.to_dict()
    else:
        # Select all nodes without parents as potential perturbations
        # Assume those are +1
        nodes = set(graph.source) - set(graph.target)
        print(f"No perturbations provided, adding {len(nodes)} source nodes")
        # Add +1 and -1 edges
        d1 = pd.DataFrame(dict(source=["perturbp"]*len(nodes), interaction=[1]*len(nodes), target=list(nodes)))
        d2 = pd.DataFrame(dict(source=["perturbn"]*len(nodes), interaction=[-1]*len(nodes), target=list(nodes)))
        print(f"Original PKN shape: {graph.shape}")
        df_perturb = pd.concat([d1, d2])
        # Extend original graph
        graph = pd.concat([graph, df_perturb])
        print(f"Modified PKN shape: {graph.shape}")
        perturbations = {"perturbp": 1, "perturbn": 1}
        pd.DataFrame({'id': ['perturbp', 'perturbn'], 'value': [1, 1]}).to_csv(pert_file, index=None)
        graph.to_csv(os.path.join(args.folder, 'network_with_perturbations.csv'), index=None)

    print(f"Loaded data:")
    print(f" - Network: {graph.shape}")
    print(f" - Measurements: {len(measurements)}")
    print(f" - Perturbations: {len(perturbations)}")

    # Remove strange symbols
    graph = graph[~graph.source.str.contains('_')]
    graph = graph[~graph.target.str.contains('_')]
    print(f"Network size after removing invalid symbols: {graph.shape}")
    

    if args.solver == 'cbc' or args.solver == 'gurobi_mip':
        print("Using CBC solver w/Python-MIP")
        solver = mip.CBC if args.solver == 'cbc' else mip.GRB
        backend = MIPProblem(solver=solver)
    else:
        print(f"Using {args.solver} w/PICOS")
        backend = PicosProblem()
        backend.p.options["solver"] = args.solver
    backend.setup(max_seconds=args.maxtime, opt_tol=args.tol)
    backend.build(graph, measurements, perturbations, penalty=args.penalty)
    backend.solve()
    backend.export(export_file)