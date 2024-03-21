library(CARNIVAL)

exportDfs <- function(df, df_network, df_perturbs, df_measures) {
    nodeVars <- df$variable[startsWith(df$variable,"nX_")]
    parts <- strsplit(nodeVars, '_')
    nodes <- sapply(parts, function(x) x[2])

    nodesUpVars <- paste0("nU_", nodes)
    nodesDownVars <- paste0("nD_", nodes)
    nodesVars <- paste0("nX_", nodes)
    nodesActStateVars <- paste0("nAc_", nodes)
    nodesDistanceVars <- paste0("nDs_", nodes)
    nodesType <- vector(mode='character', length(nodes))
    nodesType[match(df_measures$id, nodes)] <- 'M'
    nodesType[match(df_perturbs$id, nodes)] <- 'P'
    nodesDf <- data.frame(nodes, nodesVars, nodesUpVars, nodesDownVars, 
                        nodesActStateVars, nodesDistanceVars, nodesType)

    Node1 <- df_network$source
    Sign <- df_network$interaction
    Node2 <- df_network$target
    edgesUpVars <- paste0("eU_", df_network$source, '_', df_network$target)
    edgesDownVars <- paste0("eD_", df_network$source, '_', df_network$target)
    edgesDf <- data.frame(Node1, Sign, Node2, edgesUpVars, edgesDownVars)

    nodes <- df_measures$id
    value <- df_measures$value
    measurementsVars <- paste0("aD_", nodes)
    nodesVars <- paste0("nX_", nodes)

    measurementsDf <- data.frame(nodes, value, measurementsVars, nodesVars)
    variables <- list(nodesDf, edgesDf, measurementsDf)
    names(variables) <- c("nodesDf", "edgesDf", "measurementsDf")
    cops <- data.frame(solver='lpSolve')
    solVars <- as.array(df$value)
    names(solVars) <- df$variable
    #CARNIVAL:::getWeightedCollapsedSolution()
    sol <- CARNIVAL:::processSolution(as.matrix(solVars), variables, '', cops, T)
    return(sol)
}

export <- function(solution_file, network_file, perturbation_file, measure_file) {
    df <- read.csv(solution_file)
    df_network <- read.csv(network_file, colClasses=c("source"="character"))
    df_perturbs <- read.csv(perturbation_file, colClasses=c("id"="character"))
    df_measures <- read.csv(measure_file, colClasses=c("id"="character"))
    return(exportDfs(df, df_network, df_perturbs, df_measures))
}

export_folder <- function(folder) {
    solution_file <- file.path(folder, "solution.csv")
    network_file <- file.path(folder, "network.csv")
    perturbation_file <- file.path(folder, "perturbations.csv")
    measure_file <- file.path(folder, "measurements.csv")
    return(export(solution_file, network_file, perturbation_file, measure_file))
}