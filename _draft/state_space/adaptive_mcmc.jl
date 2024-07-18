using AdaptiveMCMC
using CairoMakie
using Distributions
using LogDensityProblems
using TransformVariables
using TransformedLogDensities
using UnPack

import MCMCChains
import PairPlots
import StatsPlots
import Random

function generate_data(n_observations; σ_p, σ_o, β, x₀)
    ε_t_dist = Normal(0, σ_p)
    η_t_dist = Normal(0, σ_o)

    ts = 1:n_observations

    s = Array{Float64}(undef, length(ts))
    s_onestep = Array{Float64}(undef, length(ts))
    x = Array{Float64}(undef, length(ts))
    y = Array{Float64}(undef, length(ts))

    for t in ts
        x_lastt = t == 1 ? x₀ : x[t-1]
        s_lastt = t == 1 ? x₀ : s[t-1]

        ε_t = rand(ε_t_dist)
        s[t] = β * s_lastt
        s_onestep[t] = β * x_lastt
        x[t] = s_onestep[t] + ε_t

        η_t = rand(η_t_dist)
        y[t] = x[t] + η_t
    end

    (; ts, s, s_onestep, x, y)
end

Random.seed!(1234)
true_solution = generate_data(100; σ_p = 1.0, σ_o = 2.0, β = 0.95, x₀ = 20.0);


struct ModelScratchMCMC{T1, T2}
    y::Vector{Float64}
    ts::UnitRange{Int64}
    prior_distributions::T1
    transform::T2
end

function ModelScratchMCMC(y, ts)
    prior_dists = (;
        β = Uniform(0, 1),
        σ_p = truncated(Normal(0.0, 1.0); lower = 0),
        σ_o = truncated(Normal(0.0, 1.0); lower = 0),
        x₀ = truncated(Normal(0.0, 10.0); lower = 0))
    transform = as((β = as𝕀, σ_p = asℝ₊, σ_o = asℝ₊, x₀ = asℝ₊))

    ModelScratchMCMC(y, ts, prior_dists, transform)
end

function (problem::ModelScratchMCMC)(θ)
    @unpack β, σ_p, σ_o, x₀ = θ
    @unpack y, ts, prior_distributions = problem

    lprior = 0.0
    lprior += logpdf(prior_distributions[:β], β)
    lprior += logpdf(prior_distributions[:σ_p], σ_o)
    lprior += logpdf(prior_distributions[:σ_p], σ_o)
    lprior += logpdf(prior_distributions[:x₀], x₀)

    llikelihood = 0.0

    if lprior > -Inf
        x = x₀
        for t in ts
            x = β * x + rand(Normal(0, σ_p))
            llikelihood += logpdf(Normal(x, σ_o), y[t])
        end
    end

    return llikelihood + lprior
end

function sample_prior(prob; transform = true)
    @unpack prior_distributions = prob
    θ = (;
        β = rand(prior_distributions[:β]),
        σ_p = rand(prior_distributions[:σ_p]),
        σ_o = rand(prior_distributions[:σ_o]),
        x₀ = rand(prior_distributions[:x₀]))

    if transform
        return inverse(prob.transform, θ)
    end

    return θ
end


problem = ModelScratchMCMC(true_solution.y, true_solution.ts)
ℓ = TransformedLogDensity(problem.transform, problem)
lposterior(x) = LogDensityProblems.logdensity(ℓ, x)
backtransform_samples(prob, x) = collect(transform(prob.transform, x))

lposterior(sample_prior(problem))

nsamples = 500_000; L = 2
nchains = 4

post_objs = []
for i in 1:nchains
    raw_post_chain = adaptive_rwm(sample_prior(problem), lposterior, nsamples;
                            b = nsamples ÷ 2, thin = 100, progress = true, L);

    post_chain = mapslices(x -> backtransform_samples(problem, x),
                           raw_post_chain.X, dims = 1)
    push!(post_objs, post_chain')
end

post = cat(post_objs..., dims = 3)


chn_mcmc = MCMCChains.Chains(post, collect(keys(problem.prior_distributions)))
StatsPlots.plot(chn_mcmc)
# PairPlots.pairplot(chn_mcmc,
#     PairPlots.Truth( (;σ_p = 1.0, σ_o = 2.0, β = 0.95, x₀ = 20.0)))
