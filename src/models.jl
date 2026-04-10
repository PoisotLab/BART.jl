###############################################################################
##### Training Data type and constructors
###############################################################################

struct TrainData
    n::Int
    p::Int
    xmin::Matrix{Float64}
    xmax::Matrix{Float64}
    X::Matrix{Float64}
    ybar::Float64
    y::AbstractVector
    σhat::Float64
end

function TrainData(X::Matrix{Float64}, y::Vector{Float64})
    n = length(y)
    p = size(X, 2)
    Q = Matrix(qr(X).Q)
    ybar = mean(y)
    y = y .- ybar
    yhat = Q * Q' * y
    r = y - yhat
    σhat = p >= n ? std(y) : sqrt(dot(r, r) / (n - p))
    xmin = minimum(X; dims = 1)
    xmax = maximum(X; dims = 1)
    return TrainData(n, p, xmin, xmax, X, ybar, y, σhat)
end

function TrainData(X::Matrix{Float64}, y::Vector{Int})
    n = length(y)
    p = size(X, 2)
    xmin = minimum(X; dims = 1)
    xmax = maximum(X; dims = 1)
    ybar = mean(y)
    return TrainData(n, p, xmin, xmax, X, ybar, y, 1.0)
end

"""
    TrainData(X::Matrix{Float64}, y::V) where {V <: AbstractArray{Bool ,1}}

Generates the training data from a `BitVector` or `Vector{Bool}`, by turning the
response into an integer numbers array. This will be fit using probit BART.
"""
function TrainData(X::Matrix{Float64}, y::V) where {V <: AbstractArray{Bool ,1}}
    return TrainData(X, convert(Vector{Int}, y))
end

###############################################################################
##### MCMC options type
###############################################################################

Base.@kwdef struct Opts
    nchains::Int = 4
    nburn::Int = 2500
    ndraw::Int = 2500
    nthin::Int = 1
    S::Int64 = nburn + ndraw
end

###############################################################################
##### BART hyperparameters type
###############################################################################

struct Hypers
    m::Int64
    k::Int64
    ν::Float64
    δ::Float64
    q::Float64
    α::Float64
    β::Float64
    hard::Bool
    λmean::Float64
    λfix::Bool
    sigma_noninf::Bool
    τ::Float64
    init_trees::String
    init_depth::Vector
    sparse::Bool
    shape::Float64
    a::Float64
    b::Float64
    function Hypers(td::TrainData; m = 50, k = 2,
        ν = 3.0, q = 0.9, α = 0.95, β = 2.0,
        sigma_noninf = true,
        hard = true, λmean = 0.1, λfix = false,
        init_trees = "leaf", init_depth = ones(4),
        sparse = false, shape = 1.0, a = 0.5, b = 1.0,
    )
        if !(init_trees in ["leaf", "rf"])
            throw(ArgumentError("init_tree options are leaf or rf"))
        end
        if sigma_noninf
            ν = 0
            δ = 0
        else
            δ = 1 / quantile(InverseGamma(ν / 2, ν / (2 * td.σhat^2)), q)
        end
        if isa(td.y, Vector{Int})
            τ = (3.0 / (k * sqrt(m)))^2
        else
            τ = ((maximum(td.y) - minimum(td.y)) / (2 * k * sqrt(m)))^2
        end
        if hard
            λfix = true
            λmean = 1 / 10000
        end
        return new(m, k, ν, δ, q, α, β, hard, λmean, λfix, sigma_noninf, τ,
            init_trees, init_depth, sparse, shape, a, b,
        )
    end
end

###############################################################################
##### Bart model type and constructors
###############################################################################

struct BartModel
    hypers::Hypers
    opts::Opts
    td::TrainData
end

function BartModel(X::Matrix{Float64}, y::AbstractVector, opts::Opts; hyperargs...)
    td = TrainData(X, y)
    hypers = Hypers(td; hyperargs...)
    return BartModel(hypers, opts, td)
end

###############################################################################
##### State of sampler
###############################################################################

abstract type BartState end

mutable struct SuffStats
    Lt::Int
    Ω::Matrix{Float64}
    rhat::Vector{Float64}
end

function suffstats(rt::Vector{Float64}, S::Matrix{Float64}, bs::BartState, bm::BartModel)
    Lt = size(S, 2)
    Ω = inv(transpose(S) * S / bs.σ^2 + I / bm.hypers.τ)
    rhat = transpose(S) * rt / bs.σ^2
    return SuffStats(Lt, Ω, rhat)
end

mutable struct BartTree
    tree::Tree
    S::Matrix{Float64}
    ss::SuffStats
end

mutable struct BartEnsemble
    trees::Vector{BartTree}
end

mutable struct RegBartState <: BartState
    ensemble::BartEnsemble
    fhat::Vector{Float64}
    σ::Float64
    s::Vector{Float64}
    shape::Float64
end

function Base.convert(Node, x)
    return if isa(x, DecisionTree.Leaf)
        Leaf(x.majority)
    else
        Branch(x.featid, x.featval, convert(Node, x.left), convert(Node, x.right))
    end
end

function RegBartState(bm::BartModel)
    states = RegBartState[]
    for c in 1:(bm.opts.nchains)
        if bm.hypers.init_trees == "leaf"
            trees = [Tree(Leaf(0.0), bm.hypers.λmean) for _ in 1:(bm.hypers.m)]
        else
            rf = DecisionTree.fit!(
                DecisionTree.RandomForestRegressor(;
                    n_trees = bm.hypers.m, max_depth = bm.hypers.init_depth[c],
                    min_purity_increase = 5),
                bm.td.X, bm.td.y,
            )
            trees = Tree.(convert.(Node, rf.ensemble.trees), bm.hypers.λmean)
        end
        S = [leafprob(bm.td.X, tree) for tree in trees]
        fhats = reduce(hcat, [S[t] * getμ(trees[t]) for t in eachindex(trees)])
        yhat = vec(sum(fhats; dims = 2))
        bt = Vector{BartTree}(undef, bm.hypers.m)
        for t in eachindex(trees)
            rt = bm.td.y - sum(fhats[:, eachindex(trees) .!= t]; dims = 2)
            Ω = inv(transpose(S[t]) * S[t] / bm.td.σhat^2 + I / bm.hypers.τ)
            rhat = vec(transpose(S[t]) * rt / bm.td.σhat^2)
            bt[t] = BartTree(trees[t], S[t], SuffStats(size(S[t], 2), Ω, rhat))
        end
        push!(states,
            RegBartState(
                BartEnsemble(bt),
                yhat,
                bm.td.σhat,
                ones(bm.td.p) ./ bm.td.p,
                bm.hypers.shape,
            ))
    end
    return states
end

mutable struct ProbitBartState <: BartState
    ensemble::BartEnsemble
    fhat::Vector{Float64}
    z::Vector{Float64}
    σ::Float64
    s::Vector{Float64}
    shape::Float64
end

function ProbitBartState(bm::BartModel)
    states = []
    z = map(
        y -> if y == 1
            rand(truncated(Normal(), 0, Inf))
        else
            rand(truncated(Normal(), -Inf, 0))
        end,
        bm.td.y,
    )
    for c in 1:(bm.opts.nchains)
        if bm.hypers.init_trees == "leaf"
            trees = [Tree(Leaf(0.0), bm.hypers.λmean) for _ in 1:(bm.hypers.m)]
        else
            rf = DecisionTree.fit!(
                DecisionTree.RandomForestRegressor(;
                    n_trees = bm.hypers.m, max_depth = bm.hypers.init_depth[c],
                    min_purity_increase = 5),
                bm.td.X, z,
            )
            trees = Tree.(convert.(Node, rf.ensemble.trees), bm.hypers.λmean)
        end
        S = [leafprob(bm.td.X, tree) for tree in trees]
        fhats = reduce(hcat, [S[t] * getμ(trees[t]) for t in eachindex(trees)])
        yhat = vec(sum(fhats; dims = 2))
        bt = BartEnsemble(Vector{BartTree}(undef, bm.hypers.m))
        for t in eachindex(trees)
            rt = z - sum(fhats[:, eachindex(trees) .!= t]; dims = 2)
            Ω = inv(transpose(S[t]) * S[t] + I / bm.hypers.τ)
            rhat = vec(transpose(S[t]) * rt)
            bt.trees[t] = BartTree(trees[t], S[t], SuffStats(size(S[t], 2), Ω, rhat))
        end
        push!(states,
            ProbitBartState(bt, yhat, z, 1, ones(bm.td.p) ./ bm.td.p, bm.hypers.shape))
    end
    return states
end

###############################################################################
##### Posterior draws from BART model
###############################################################################

abstract type BartPosterior end

struct RegBartPosterior <: BartPosterior
    sdraws::Matrix{Float64}
    σdraws::Vector{Float64}
    treedraws::Vector{Vector{Tree}}
    function RegBartPosterior(bm::BartModel)
        return new(
            Matrix{Float64}(undef, bm.td.p, bm.opts.ndraw),
            Vector{Float64}(undef, bm.opts.ndraw),
            Vector{Vector{Tree}}(undef, bm.opts.ndraw),
        )
    end
end

struct ProbitBartPosterior <: BartPosterior
    sdraws::Matrix{Float64}
    zdraws::Matrix{Float64}
    treedraws::Vector{Vector{Tree}}
    function ProbitBartPosterior(bm::BartModel)
        return new(
            Matrix{Float64}(undef, bm.td.p, bm.opts.ndraw),
            Matrix{Float64}(undef, bm.td.n, bm.opts.ndraw),
            Vector{Vector{Tree}}(undef, bm.opts.ndraw),
        )
    end
end

abstract type BartChain end

struct RegBartChain <: BartChain
    bm::BartModel
    init_trees::Any
    sdraws::Array{Float64}
    treedraws::Array{Vector{Tree}}
    σdraws::Array{Float64}
end

struct ProbitBartChain <: BartChain
    bm::BartModel
    init_trees::Any
    sdraws::Array{Float64}
    treedraws::Array{Vector{Tree}}
end

struct SurvBartChain <: BartChain
    bm::BartModel
    time::Vector{Float64}
    event::Vector{Int}
    init_trees::Any
    sdraws::Array{Float64}
    treedraws::Array{Vector{Tree}}
    Sdraws::Array{Float64}
end

###############################################################################
##### MCMCChains
###############################################################################

function MCMCChains.Chains(bc::Union{ProbitBartChain, SurvBartChain})
    atd = mapslices(td -> map(x -> mean(depth.(x)), td), bc.treedraws; dims = 3)
    mpost = reshape(ptd(bc), bc.bm.opts.ndraw, 1, bc.bm.opts.nchains)
    monitor = hcat(mpost, atd)
    return Chains(mpost, ["PTD", "ATD"])
end

function MCMCChains.Chains(bc::RegBartChain)
    atd = mapslices(td -> map(x -> mean(depth.(x)), td), bc.treedraws; dims = 3)
    mpost = reshape(ptd(bc), bc.bm.opts.ndraw, 1, bc.bm.opts.nchains)
    monitor = hcat(bc.σdraws, mpost, atd)
    return Chains(monitor, ["sigma", "PTD", "ATD"])
end
