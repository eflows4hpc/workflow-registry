import os
import importlib
import re
import shutil

def build_variables(values, problem):
    names = get_names(problem)
    variables_fixed = problem.get("variables-fixed")
    calls = problem.get("variables-derivate")
    variables = []
    for i in range(len(values)):
        value = {names[i]: values[i]}
        variables.append(value)
    for variable_fixed in variables_fixed:
        variables.append(variable_fixed)
    if calls:
        for name, value in calls.items():
            call = value
            head, tail = call.get("method").split(".")
            parameters = call.get("parameters")
            args = []
            for parameter in parameters:
                if re.search("eval\(", parameter):
                    s = parameter.replace('eval(', '')
                    s = s.replace(')', '')
                    res = callEval(s, variables)
                    args.append(res)
                else:
                    args.append(loop(parameter, variables))
            module = importlib.import_module(head)
            c = getattr(module, tail)(*args)
            outputs = call.get("outputs")
            for i in range(len(outputs)):
                var = {outputs[i]: c[i]}
                variables.append(var)

    return variables

def copy(src_dir, src_name, tgt_dir, tgt_name):
    src_file = os.path.join(src_dir, src_name)
    tgt_file = os.path.join(tgt_dir, tgt_name)
    shutil.copyfile(src_file, tgt_file)
    return

def create_env_simulations(mesh, sim_dir, original_name, name_sim):
    copy(mesh, original_name + ".ker.dat", sim_dir, name_sim + ".ker.dat")
    copy(mesh, original_name + ".dat", sim_dir, name_sim + ".dat")
    copy(mesh, original_name + ".dom.dat", sim_dir, name_sim + ".dom.dat")
    copy(mesh, original_name + ".fie.dat", sim_dir, name_sim + ".fie.dat")
    copy(mesh, original_name + ".post.alyadat", sim_dir, name_sim + ".post.alyadat")

def get_names(problem):
    variables = problem.get("variables-sampler")
    names = []
    for item in variables:
        for key, value in item.items():
            names.append(key)
    return names

def callEval(parameter, variables):
    groups = re.split(r'\b', parameter)
    for group in groups:
        if group != "":
            if not group.isnumeric():
                for variable in variables:
                    if group in variable:
                        var = variable.get(group)
                        parameter = parameter.replace(group, str(var))
                        break
    res = eval(parameter)
    return res

def loop(parameter, variables):
    for variable in variables:
        if parameter in variable:
            var = variable.get(parameter)
            return var
