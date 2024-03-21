#!/usr/bin/env Rscript --vanilla
# Author: Attila Gabor

library(rhdf5)
library(CellNOptR)
library(dplyr)
library(tibble)
library(readr)
library(optparse)

#' export CellNOpt data and model to hdf5 file
#' 
#' exports the model and data structure to an HDF5 file
#' 
#' @param cnolist CNOlist data structure, see CellNOptR::CNOlist()
#' @param model CellNOpt pkn model object, use CellNOptR::readSIF() and optionally CellNOptR::preprocessing(). 
#' @param h5_model_file file name to save the data to. 
#' 
export_model_to_hdf5 <- function(cnolist, model, h5_model_file){
    
    simList = CellNOptR::prep4sim(model)
    indexList = CellNOptR::indexFinder(cnolist, model)
    
    ### save all data generated
    h5_model <- h5_model_file
    rhdf5::h5createFile(h5_model)
    
    # 1. export cnolist data structure:
    h5createGroup(h5_model,group = "cnolist")
    
    ## these are matrices
    h5write(t(cnolist@cues), h5_model,"cnolist/cues")
    h5write(t(cnolist@inhibitors), h5_model,"cnolist/inhibitors")
    h5write(t(cnolist@stimuli), h5_model,"cnolist/stimuli")
    ## vector: 
    h5write(cnolist@timepoints, h5_model,"cnolist/timepoints")
    
    # signals
    # TODO: time index should be flexible
    h5createGroup(h5_model,group = "cnolist/signals")
    h5write(t(cnolist@signals[[1]]), h5_model,"cnolist/signals/t0")
    h5write(t(cnolist@signals[[2]]), h5_model,"cnolist/signals/t1")
    
    # variances
    # TODO: time index should be flexible
    h5createGroup(h5_model,group = "cnolist/variances")
    h5write(t(cnolist@variances[[1]]), h5_model,"cnolist/variances/t0")
    h5write(t(cnolist@variances[[2]]), h5_model,"cnolist/variances/t1")
    
    
    # 2. export the network data structure:
    h5createGroup(h5_model,group = "model")
    
    ## these are scalars: 
    h5write(length(model$reacID), h5_model,"model/reacIDNum")
    h5write(length(model$namesSpecies), h5_model,"model/namesSpeciesNum")
    h5write(length(model$speciesCompressed), h5_model,"model/speciesCompressedNum")
    
    ## matrices: 
    h5write(t(model$interMat), h5_model,"model/interMat")
    h5write(t(model$notMat), h5_model,"model/notMat")
    
    ## !! two variables that are varying length of lists are not exported. 
    # I think we wont need them - they contain information about the compression. 
    
    # 3. auxilary data
    #
    
    h5createGroup(h5_model,group = "simList")
    
    # matrices
    h5write(t(simList$finalCube), h5_model,"simList/finalCube")
    h5write(t(simList$ixNeg), h5_model,"simList/ixNeg")
    h5write(t(simList$ignoreCube), h5_model,"simList/ignoreCube")
    
    # vectors
    h5write(simList$maxIx, h5_model,"simList/maxIx")
    # constant
    h5write(simList$maxInput, h5_model,"simList/maxInput")
    
    
    h5createGroup(h5_model,group = "indexList")
    
    # vectors
    h5write(indexList$signals, h5_model,"indexList/signals")
    h5write(indexList$stimulated, h5_model,"indexList/stimulated")
    h5write(indexList$inhibited, h5_model,"indexList/inhibited")
    
    H5close()
    
}


# carnival_to_cellnopt
# 
# creates a CellNOpt model and a data object from the CARNIVAL inputs. 
#' @param sif PKN
#' @param measurement data.frame, same as the input of runCARNIVAL
#' @param inputs data.frame, same as the input of runCARNIVAL
#' @return a CNOlist that represents the CARNIVAL problem as a CellNOpt object
carnival_to_cellnopt <- function(sif, measurments, inputs){
    
    # create the Model object from the prior knowledge
    tmpfile <- tempfile(fileext = ".sif")
    write_tsv(sif,file = tmpfile,col_names = FALSE)
    model <- CellNOptR::readSIF(tmpfile)
    
    
    # create the data object
    
    # in CARNIVAL: the inputs can take -1 or +1 values
    if(!is.data.frame(inputs)) stop("input should be data.frame")
    if(nrow(inputs)!=1) stop("input should have only 1 row")
    if(!all(unlist(inputs) %in% c(-1,0,1))) stop("input should be discretized. Only  values of -1, 0 and 1 are allowed.")
    
    # Checking the measurements
    if(!is.data.frame(measurments)) stop("measurments should be data.frame")
    if(nrow(measurments)!=1) stop("measurments should have only 1 row")
    
    
    
    cnodata = list(
        namesCues = colnames(inputs),
        namesStimuli = colnames(inputs),
        namesInhibitors = c(),
        namesSignals = colnames(measurments),
        timeSignals = c(0),
        valueCues = matrix(1,nrow = 1,ncol = ncol(inputs)),
        valueInhibitors = matrix(1,nrow = 1,ncol = 0),
        valueStimuli = as.matrix(inputs),
        valueSignals = list(t0 = matrix(as.numeric(measurments),nrow = 1,ncol = ncol(measurments)))
    )
    cnodata <- CNOlist(cnodata)
    
    return(list(cnodata = cnodata, model = model))
}




#' export CARNIVAL derived cnodata and model to hdf5 file
#' 
#' exports the model and data structure to an HDF5 file
#' 
#' @param cnolist CNOlist data structure, see carnival_to_cellnopt
#' @param model CellNOpt pkn model object, use CellNOptR::readSIF() and optionally CellNOptR::preprocessing(). 
#' @param h5_model_file file name to save the data to. 
#' 
export_CARNIVAL_to_hdf5 <- function(cnolist, model, h5_model_file){
    
    simList = CellNOptR::prep4sim(model)
    indexList = CellNOptR::indexFinder(cnolist, model)
    
    ### save all data generated
    h5_model <- h5_model_file
    rhdf5::h5createFile(h5_model)
    
    # 1. export cnolist data structure:
    h5createGroup(h5_model,group = "cnolist")
    
    ## these are matrices
    h5write(t(cnolist@cues), h5_model,"cnolist/cues")
    h5write(t(cnolist@inhibitors), h5_model,"cnolist/inhibitors")
    h5write(t(cnolist@stimuli), h5_model,"cnolist/stimuli")
    ## vector: 
    h5write(cnolist@timepoints, h5_model,"cnolist/timepoints")
    
    # signals (CARNIVAL: single time point only)
    # TODO: time index should be flexible
    h5createGroup(h5_model,group = "cnolist/signals")
    h5write(t(cnolist@signals[[1]]), h5_model,"cnolist/signals/t0")
    
    
    # variances
    # not for CARNIVAL
    
    # 2. export the network data structure:
    h5createGroup(h5_model,group = "model")
    
    ## these are scalars: 
    h5write(length(model$reacID), h5_model,"model/reacIDNum")
    h5write(length(model$namesSpecies), h5_model,"model/namesSpeciesNum")
    h5write(length(model$speciesCompressed), h5_model,"model/speciesCompressedNum")
    
    ## matrices: 
    h5write(t(model$interMat), h5_model,"model/interMat")
    h5write(t(model$notMat), h5_model,"model/notMat")
    
    ## !! two variables that are varying length of lists are not exported. 
    # I think we wont need them - they contain information about the compression. 
    
    # 3. auxilary data
    #
    
    h5createGroup(h5_model,group = "simList")
    
    # matrices
    h5write(t(simList$finalCube), h5_model,"simList/finalCube")
    h5write(t(simList$ixNeg), h5_model,"simList/ixNeg")
    h5write(t(simList$ignoreCube), h5_model,"simList/ignoreCube")
    
    # vectors
    h5write(simList$maxIx, h5_model,"simList/maxIx")
    # constant
    h5write(simList$maxInput, h5_model,"simList/maxInput")
    
    
    h5createGroup(h5_model,group = "indexList")
    
    # vectors
    h5write(indexList$signals, h5_model,"indexList/signals")
    h5write(indexList$stimulated, h5_model,"indexList/stimulated")
    h5write(indexList$inhibited, h5_model,"indexList/inhibited")
    
    H5close()
    
}


# maps CARNIVAL output to the PKN and generates a BitString to plot with CellNOpt
# takes the best result insteaof the average. 
carnival_res_to_bString <- function(model, result) {
    carnival_reactions <-
        result$sifAll[[1]] %>% tibble::as_tibble() %>%
        dplyr::mutate(reacID = paste0(ifelse(Sign == 1, "", "!"), Node1, "=", Node2)) %>%
        pull(reacID)
    
    bString = as.numeric(model$reacID %in% carnival_reactions)
    
    if (!all(carnival_reactions %in% model$reacID))
        stop("carnival reaction not in PKN!")
    
    return(bString)
}


parser <- OptionParser(
  usage = "usage: %prog sif_file measurements_file inputs_file output_hdf5 [options]",
  option_list = list(
    make_option(c("-v", "--verbose"), default=F, help="Verbosity (default False)")
  ),
  add_help_option = T,
  prog = "Export CARNIVAL data to hdf5 format compatible with CellNopt",
  formatter = IndentedHelpFormatter
)

arguments <- parse_args(parser, positional_arguments = T)
verbose <- arguments$options$verbose

sif <- as.data.frame(read_csv(arguments$args[1]))
measurements <- as.data.frame(read_csv(arguments$args[2]))
inputs <- data.frame(deframe(read_csv(arguments$args[3])) %>% as.list)
result <- carnival_to_cellnopt(sif, measurements, inputs)

# Export to hdf5
export_CARNIVAL_to_hdf5(result$cnodata, result$model, arguments$args[4])
