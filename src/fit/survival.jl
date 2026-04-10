function preprocess(X::Matrix{Float64}, time::Vector{Float64}, event::Vector{Int})
    n, p = size(X)
    ut = sort(unique(time))
    yij = Int[]
    tj = Float64[]
    obs = Int[]
    for i in 1:n
        for t in filter(t -> t <= time[i], ut)
            t < time[i] ? push!(yij, 0) : push!(yij, event[i])
            push!(tj, t)
            push!(obs, i)
        end
    end
    ns = counts(obs)
    Xrep = [transpose(reshape(repeat(X[i, :], ns[i]), p, ns[i])) for i in 1:n]
    Xrep = reduce(vcat, Xrep)
    return (yij, hcat(tj, Xrep))
end

function StatsBase.fit(BartModel, X::Matrix{Float64}, time::Vector{Float64},
    event::Vector{Int},
    opts = Opts(); hyperags...)
    y, X = preprocess(X, time, event)
    bm = BartModel(X, y, opts; hyperags...)
    states = ProbitBartState(bm)
    init_trees =
        deepcopy(map(state -> Tree[bt.tree for bt in state.ensemble.trees], states))
    post = pmap(bs -> sample(bs, bm), states)
     if BART._VERBOSE
        println("Processing chains...")
    end
    treedraws = reduce(hcat, pmap(p -> p.treedraws, post))
    pdraws = reduce(hcat, pmap(t -> cdf.(Normal(), predict(t, bm.td.X)), treedraws))
    ut = sort(unique(time))
    obs = [i for i in 1:length(time) for t in filter(t -> t <= time[i], ut)]
    ns = counts(obs)
    indices = [findall(obs .== i) for i in 1:(bm.td.n)]
    sdraws = [cumprod(1 .- pdraws[idx, :]; dims = 1) for idx in indices]
    sdraws = reduce(vcat, sdraws)
    return SurvBartChain(
        bm,
        time,
        event,
        init_trees,
        reshape(
            reduce(hcat, [chain.sdraws for chain in post]),
            bm.td.p,
            bm.opts.ndraw,
            bm.opts.nchains,
        ),
        reshape(
            reduce(vcat, [chain.treedraws for chain in post]),
            bm.opts.ndraw,
            1,
            bm.opts.nchains,
        ),
        sdraws,
    )
end