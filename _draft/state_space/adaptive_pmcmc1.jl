using AdaptiveParticleMCMC
using CairoMakie
using Distributions
using LabelledArrays
using Statistics

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
        x[t] = (β + ε_t) * x_lastt

        η_t = rand(η_t_dist)
        y[t] = x[t] + η_t
    end

    (; ts, s, s_onestep, x, y)
end

Random.seed!(1234)
true_solution = generate_data(100; σ_p = 0.1, σ_o = 2.0, β = 0.95, x₀ = 20.0);


mutable struct Particle
    s::Float64
    Particle() = new(0.0)
end


mutable struct Param
    β::Float64
    σ_p::Float64
    σ_o::Float64
    x₀::Float64
end

struct ModelScratch
    par::Param
    y::Vector{Float64}
    ModelScratch() = new(Param(0.5, 2.0, 2.0, 10.0), true_solution.y)
end


function transition!(x, rng, k, x_prev, scratch)
    if k == 1
        x.s = (scratch.par.β + rand(rng, Normal(0, scratch.par.σ_p))) * scratch.par.x₀
    else
        x.s = (scratch.par.β + rand(rng, Normal(0, scratch.par.σ_p))) * x_prev.s
    end
end


function log_potential(k, x, scratch)
    logpdf(Normal(x.s, scratch.par.σ_o), scratch.y[k])
end

inv_logit(x) = exp(x)/(1+exp(x))
function set_param!(scratch, θ)
    scratch.par.β = inv_logit(θ.logit_beta)
    scratch.par.σ_p = exp(θ.log_sigma_p)
    scratch.par.σ_o = exp(θ.log_sigma_o)
    scratch.par.x₀ = exp(θ.log_x₀)
end

function prior(theta)
    (logpdf(Normal(0.0, 2.0), theta.logit_beta) +
     logpdf(Normal(-1.0, 2.0), theta.log_sigma_p) +
     logpdf(Normal(-1.0, 2.0), theta.log_sigma_o) +
     logpdf(Normal(0.0, 2.0), theta.log_x₀))
end

##################################################
T = length(true_solution.y)
nparticles = 100
nsamples = 200_000

theta0 = LVector(logit_beta = 0.0, log_sigma_p = 0.0, log_sigma_o = 0.0,
                  log_x₀ = 0.0)

state = SMCState(T, nparticles, Particle, ModelScratch, set_param!, log_potential, transition!);

out = adaptive_pmmh(theta0, prior, state, nsamples; thin = 1,
                    save_paths = true, b = nsamples ÷ 2, show_progress = true);

θ = deepcopy(out.Theta)
θ[1, :] = inv_logit.(out.Theta[1, :])
θ[2, :] = exp.(out.Theta[2, :])
θ[3, :] = exp.(out.Theta[3, :])
θ[4, :] = exp.(out.Theta[4, :])

chn_pmcmc = MCMCChains.Chains(θ', collect(fieldnames(Param)))
StatsPlots.plot(chn_pmcmc)
PairPlots.pairplot(chn_pmcmc,
    PairPlots.Truth( (;σ_p = 0.1, σ_o = 2.0, β = 0.95, x₀ = 20.0)))


set_theme!(
    fontsize = 18,
    Axis = (; xgridvisible = false, ygridvisible = false,
            topspinevisible = false, rightspinevisible = false),
    Legend = (; framevisible = false))


S = [out.X[j][i].s for i=1:length(out.X[1]), j=1:length(out.X)]
q95 = mapslices(x->quantile(x, [0.025, 0.975]), S, dims=2)
q5 = mapslices(x->quantile(x, [0.25, 0.75]), S, dims=2)
q_median = mapslices(median, S, dims=2)


let
    fig = Figure(size = (900, 900))

    ax = Axis(fig[1, 1]; ylabel = "value")
    band!(true_solution.ts, q95[:, 1], q95[:, 2], color = (:black, 0.2), label = "95% credible interval")
    band!(true_solution.ts, q5[:, 1], q5[:, 2], color = (:black, 0.5), label = "50% credible interval")
    lines!(true_solution.ts, q_median[:, 1], color = :black, label = "median")
    scatter!(true_solution.ts, true_solution.y, color = :steelblue4, label = "observations: y")
    lines!(true_solution.ts, true_solution.x, color = :blue, label = "true hidden state: x")
    lines!(true_solution.ts, true_solution.s, color = :red, label = "process-model state: s")
    Legend(fig[1, 2], ax)


    Axis(fig[2, 1])
    for i in 1:50
        xs = Array{Float64}(undef, length(true_solution.ts))

        p = sample(chn_pmcmc, 1).value
        x = p[var = :x₀][1]
        β = p[var = :β][1]

        for t in true_solution.ts
            x = β * x
            xs[t] = x
        end

        lines!(true_solution.ts, xs, color = (:black, 0.2))
    end
    lines!(true_solution.ts, true_solution.s, color = :red, linewidth = 5)

    Axis(fig[3, 1]; xlabel = "time", ylabel = "value")
    for i in 1:50
        xs = Array{Float64}(undef, length(true_solution.ts))

        p = sample(chn_pmcmc, 1).value
        x = p[var = :x₀][1]
        β = p[var = :β][1]
        σ_p = p[var = :σ_p][1]
        ε_t_dist = Normal(0, σ_p)
        ε = rand(ε_t_dist, length(true_solution.ts))

        for t in true_solution.ts
            x = (β + ε[t]) * x
            xs[t] = x
        end

        lines!(true_solution.ts, xs, color = (:black, 0.2))

    end
    lines!(true_solution.ts, true_solution.x, color = :blue, linewidth = 5)

    rowsize!(fig.layout, 1, Relative(0.5))
    fig
end
