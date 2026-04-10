module BART

using Distributed
using Distributions
using LinearAlgebra
using MCMCChains
using StatsBase
using SpecialFunctions

import DecisionTree

const _VERBOSE = true

include("trees.jl")
include("models.jl")
include("proposals.jl")
include("predict.jl")
include("fit.jl")

export
    BartModel,
    Chains,
    Hypers,
    Opts,
    TrainData

export
    depth,
    fit,
    ptd,
    predict,
    update

end
