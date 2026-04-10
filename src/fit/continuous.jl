
function StatsBase.sample(bs::RegBartState, bm::BartModel)
    posterior = RegBartPosterior(bm)
    @time for s in 1:(bm.opts.S)
        drawtrees!(bs, bm)
        drawσ!(bs, bm)
        if bm.hypers.sparse
            draws!(bs, bm)
            drawα!(bs, bm)
        end
        if s > bm.opts.nburn
            posterior.sdraws[:, s - bm.opts.nburn] = exp.(bs.s)
            posterior.σdraws[s - bm.opts.nburn] = bs.σ
            posterior.treedraws[s - bm.opts.nburn] =
                [deepcopy(t.tree) for t in bs.ensemble.trees]
        end
        if iszero(s % 100) & BART._VERBOSE
            println("MCMC iteration $s complete.")
        end
    end
    return posterior
end

function StatsBase.fit(
    BartModel,
    X::Matrix{Float64},
    y::Vector{Float64},
    opts = Opts();
    hyperags...,
)
    bm = BartModel(X, y, opts; hyperags...)
    states = RegBartState(bm)
    init_trees =
        deepcopy(map(state -> Tree[bt.tree for bt in state.ensemble.trees], states))
    post = pmap(bs -> sample(bs, bm), states)
    if BART._VERBOSE
        println("Processing chains...")
    end
    return RegBartChain(
        bm,
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
        reshape(
            reduce(vcat, [chain.σdraws for chain in post]),
            bm.opts.ndraw,
            1,
            bm.opts.nchains,
        ),
    )
end

function update(post::RegBartChain, ndraw::Int)
    s = post.bm.opts.ndraw
    new_opts = Opts(; ndraw = ndraw, nburn = 0, nchains = post.bm.opts.nchains)
    states = RegBartState[]
    for c in 1:(post.bm.opts.nchains)
        trees = post.treedraws[s, 1, c]
        σ = post.σdraws[s, 1, c]
        S = [leafprob(post.bm.td.X, tree) for tree in trees]
        fhats = reduce(hcat, [S[t] * getμ(trees[t]) for t in eachindex(trees)])
        yhat = vec(sum(fhats; dims = 2))
        bt = Vector{BartTree}(undef, post.bm.hypers.m)
        for t in eachindex(trees)
            rt = post.bm.td.y - sum(fhats[:, eachindex(trees) .!= t]; dims = 2)
            Ω = inv(transpose(S[t]) * S[t] / σ^2 + I / post.bm.hypers.τ)
            rhat = vec(transpose(S[t]) * rt / σ^2)
            bt[t] = BartTree(trees[t], S[t], SuffStats(size(S[t], 2), Ω, rhat))
        end
        bs = RegBartState(
            BartEnsemble(bt),
            post.mdraws[:, s, c],
            σ,
            ones(post.bm.td.p) ./ post.bm.td.p,
        )
        push!(states, bs)
    end
    bm = BartModel(post.bm.hypers, new_opts, post.bm.td)
    newdraws = pmap(bs -> sample(bs, bm), states)
    return RegBartChain(
        BartModel(
            bm.hypers,
            Opts(; ndraw = ndraw + post.bm.opts.ndraw, nburn = post.bm.opts.nburn),
            bm.td,
        ),
        post.init_trees,
        hcat(post.sdraws,
            reshape(
                reduce(hcat, [chain.sdraws for chain in newdraws]),
                bm.td.p,
                bm.opts.ndraw,
                bm.opts.nchains,
            )),
        vcat(post.treedraws,
            reshape(
                reduce(vcat, [chain.treedraws for chain in newdraws]),
                bm.opts.ndraw,
                1,
                bm.opts.nchains,
            )),
        vcat(post.σdraws,
            reshape(
                reduce(vcat, [chain.σdraws for chain in newdraws]),
                bm.opts.ndraw,
                1,
                bm.opts.nchains,
            )),
    )
end