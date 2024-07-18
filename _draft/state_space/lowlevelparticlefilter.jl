using AdaptiveParticleMCMC
using AdaptiveMCMC
using CairoMakie
using Distributions
using LabelledArrays
using Statistics
using AdaptiveMCMC
using CairoMakie
using LogDensityProblems
using StateSpaceModels
using TransformVariables
using TransformedLogDensities
using UnPack

import MCMCChains
import PairPlots
import StatsPlots
import Random

set_theme!(
    fontsize = 18,
    Axis = (; xgridvisible = false, ygridvisible = false,
            topspinevisible = false, rightspinevisible = false),
    Legend = (; framevisible = false))

function generate_data(n_observations; σ_p, σ_o, r, K, x₀)
    ε_t_dist = Normal(0, σ_p)
    η_t_dist = Normal(0, σ_o)

    ts = 1:n_observations

    s = Array{Float64}(undef, length(ts))
    s_onestep = Array{Float64}(undef, length(ts))
    x = Array{Float64}(undef, length(ts))
    y = Array{Float64}(undef, length(ts))

    ε = rand(Normal(0, σ_p), length(ts))
    η = rand(Normal(0, σ_o), length(ts))

    growth_rate = Array{Float64}(undef, length(ts))

    for t in ts
        s_lastt = t == 1 ? x₀ : s[t-1]
        x_lastt = t == 1 ? x₀ : x[t-1]
        s[t] = (1 + r *(1 - s_lastt/K)) * s_lastt


        growth_rate[t] = r*(1 - x_lastt/K)
        x[t] = (1 + (r*(1 - x_lastt/K) + ε[t])) * x_lastt
        y[t] = x[t] + η[t]
    end


    (; ts, s, η, ε, x, y, growth_rate, parameter = (; r, K, σ_p, σ_o, x₀))
end

Random.seed!(0)
true_solution = generate_data(1000; σ_p = 0.03, σ_o = 15.0, r = 0.01, K = 250.0, x₀ = 50.0);

let
    fig = Figure(size = (1100, 1100))
    ax = Axis(fig[1, 1]; ylabel = "population size", xticklabelsvisible = false)

    scatter!(true_solution.ts, true_solution.y, color = :red, label = "observations: y")
    lines!(true_solution.ts, true_solution.x, color = :blue, label = "true hidden state: x")
    lines!(true_solution.ts, true_solution.s, color = :grey, label = "process-model state: s")

    ylims!(-10, nothing)
    Legend(fig[1, 2], ax)

    Axis(fig[2, 1]; ylabel = "Growth rate", xticklabelsvisible = false)
    lines!(true_solution.ts, true_solution.growth_rate, color = :black)

    Axis(fig[3, 1]; ylabel = "Process error\nε ~ 𝒩(0, σₚ=$(true_solution.parameter.σ_p))\nvariation in growth rate",
        xticklabelsvisible = false)
    lines!(true_solution.ts, true_solution.ε, color = :purple)

    Axis(fig[4, 1]; ylabel = "Observation error\nη ~ 𝒩(0, σₒ=$(true_solution.parameter.σ_o))\n ", xlabel = "time")
    lines!(true_solution.ts, true_solution.η, color = :purple)

    fig
end

function my_dynamics(x, u, p, t; noise = true)
    @unpack r, K, σ_p, x₀ = p

    ε = rand(rng, Normal(0, σ_p))
    if t == 1
        return (1 + (r * (1 - x₀/K) + ε)) * x₀
    else
        return (1 + (r * (1 - x_prev.s/K) + ε)) * x_prev.s
    end
end

function my_measurement(x, u, p, t; noise = true)
    @unpack σ_o = p
    η = rand(rng, Normal(0, σ_o))
    x + η
end



using LowLevelParticleFilters, LinearAlgebra, StaticArrays, Distributions, Plots
nx = 2   # Dimension of state
nu = 2   # Dimension of input
ny = 2   # Dimension of measurements
N = 2000 # Number of particles

dg = MvNormal(ny,1.0)          # Measurement noise Distribution
df = MvNormal(nx,1.0)          # Dynamics noise Distribution
d0 = MvNormal(randn(nx),2.0)   # Initial state Distribution

A = SA[1 0.1; 0 1]
B = @SMatrix [0.0 0.1; 1 0.1]
C = @SMatrix [1.0 0; 0 1]

dynamics(x,u,p,t) = A*x .+ B*u
measurement(x,u,p,t) = C*x
pf = ParticleFilter(N, dynamics, measurement, df, dg, d0)
xs,u,y = simulate(pf,300,df)

filter_from_parameters(θ, pf = nothing) = KalmanFilter(A, B, C, 0, exp(θ[1])^2*I(nx), exp(θ[2])^2*I(ny), d0) # Works with particle filters as well

p = nothing
priors = [Normal(0,2),Normal(0,2)]
ll     = log_likelihood_fun(filter_from_parameters, priors, u, y, p)
θ₀     = log.([1.0, 1.0]) # Starting point

draw   = θ -> θ .+ 0.05 .* randn.() # This function dictates how new proposal parameters are being generated.
burnin = 200 # remove this many initial samples ("burn-in period")
@info "Starting Metropolis algorithm"
@time theta, lls = metropolis(ll, 2200, θ₀, draw) # Run PMMH for 2200  iterations
thetam = reduce(hcat, theta)'[burnin+1:end,:] # Build a matrix of the output
histogram(exp.(thetam), layout=(3,1), lab=["R1" "R2"]); Plots.plot!(lls[burnin+1:end], subplot=3, lab="log likelihood") # Visualize


@time thetalls = LowLevelParticleFilters.metropolis_threaded(burnin, ll, 2200, θ₀, draw, nthreads=2)
histogram(exp.(thetalls[:,1:2]), layout=3)
Plots.plot!(thetalls[:,3], subplot=3)

using DynamicHMC, LogDensityProblemsAD, ForwardDiff, LogDensityProblems, LinearAlgebra

struct LogTargetDensity{F}
    ll::F
    dim::Int
end
LogDensityProblems.logdensity(p::LogTargetDensity, θ) = p.ll(θ)
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()

function filter_from_parameters(θ, pf = nothing)
    T = eltype(θ)
    KalmanFilter(A, B, C, 0, exp(θ[1])^2*I(nx), exp(θ[2])^2*I(ny), MvNormal(T.(d0.μ), d0.Σ)) # It's important that the distribution of the initial state has the same element type as the parameters. DynamicHMC will use Dual numbers for differentiation, hence, we make sure that d0 has `eltype(d0) = eltype(θ)`
end
ll = log_likelihood_fun(filter_from_parameters, priors, u, y, p)

D = length(θ₀)
ℓπ = LogTargetDensity(ll, D)
∇P = ADgradient(:ForwardDiff, ℓπ)

import Random
results = mcmc_with_warmup(Random.default_rng(), ∇P, 3000)
DynamicHMC.Diagnostics.summarize_tree_statistics(results.tree_statistics)
lls = [ts.π for ts in results.tree_statistics]

histogram(exp.(results.posterior_matrix)', layout=(3,1), lab=["R1" "R2"])
Plots.plot!(lls, subplot=3, lab="log likelihood") # Visualize
